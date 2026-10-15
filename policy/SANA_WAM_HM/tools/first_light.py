"""GPU first light of the hierarchical-memory adapter WITHOUT a checkpoint: the real-dimension model with random weights
(bf16, the deploy's autocast), driven by the session for a few decisions in every deploy form the recipe allows, plus
(with ``--vae-path``) the causal-VAE streams on synthetic 320 x 512 canvases.

Checks: the forwards run on the real geometry (32 blocks, 160-token latent frames, 833-token generate windows -- 673
without ``chunk_observation`` --, K + 1 dense chunks) under bf16 autocast; per-decision latencies (commit, generate) and peak memory; finite outputs; the
streaming vs full forms on the same decision (equal up to bf16 round-off); the dense / sparse streams equal the one-shot
encode on this GPU and dtype (Sana plan section 1: "maxdiff 0 on one device / dtype / kernel path").

Usage (from policy/SANA_WAM_HM, PYTHONPATH=<XPolicyLab parent>):
  python tools/first_light.py --recipe tests/fixtures/recipe_hm_dense_sparse_videoonly_e8dde6b56.yaml [--vae-path P]
         [--decisions 4] [--steps 10] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR))

from sana_wam_hm import shared  # noqa: E402,F401
from sana_wam_hm import memory as hm  # noqa: E402
from sana_wam_hm.contract import DEPLOY_FULL, DEPLOY_STREAMING, bidirectional_view, resolve_hm_contract  # noqa: E402
from sana_wam_hm.model import build_hm_policy  # noqa: E402
from sana_wam_hm.session import Caption, HMSession  # noqa: E402
from sana_wam_hm.vae_stream import LatentStream, encode_once  # noqa: E402
from sana_wam_min.config import policy_config_from_train_config, resolve_rope_contract  # noqa: E402


def synced(fn, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def relative_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float(((a - b) ** 2).mean().sqrt() / (b**2).mean().sqrt().clamp_min(1e-8))


def run_model(args, report: dict) -> None:
    cfg = yaml.safe_load(open(args.recipe))
    contract = resolve_hm_contract(cfg)
    view = bidirectional_view(cfg)
    rope, origin, _ = resolve_rope_contract(view, None)
    config = policy_config_from_train_config(view, rope=rope, legacy_origin=origin)
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = build_hm_policy(config, contract, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                continue
            parameter.uniform_(-0.02, 0.02)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    report["contract"] = contract.describe()
    report["parameters_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
    g = torch.Generator(device="cpu").manual_seed(1)

    def randn(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=torch.bfloat16)

    length, channels = config.model_max_length, config.caption_channels
    caption = Caption(randn(1, 1, 1, length, channels), torch.ones(1, 1, length, dtype=torch.int16, device=device))
    null = Caption(randn(1, 1, 1, length, channels), torch.ones(1, 1, length, dtype=torch.int16, device=device))
    C = config.in_channels
    dense = [randn(1, C, 1, 10, 16) for _ in range(args.decisions)]
    sparse = [randn(1, C, hm.sparse_latents(contract.sparse_frames), 10, 16) for _ in range(args.decisions)]
    obs = [randn(1, C, 1, 10, 16) for _ in range(args.decisions)]
    state = randn(80)
    mask = torch.zeros(1, 32, 80, dtype=torch.bool, device=device)
    mask[..., 7:17] = True
    mask[..., 36:46] = True
    forms = [DEPLOY_STREAMING, DEPLOY_FULL]
    results = {}
    for form in forms:
        session = HMSession(
            model, caption=caption, unconditional_caption=null, steps=args.steps, flow_shift=5.0, action_flow_shift=1.0,
            video_cfg_scale=6.0, action_cfg_scale=1.0, model_fps=contract.model_fps, deploy_form=form,
        )
        torch.cuda.reset_peak_memory_stats(device)
        rows, outs = [], []
        for c in range(args.decisions):
            commit_s = None
            if session.reads_dense:
                _, commit_s = synced(session.add_dense, dense[c])
            gen = torch.Generator(device=device).manual_seed(100 + c)
            (action, video), generate_s = synced(
                session.generate, obs[c] if contract.chunk_observation else None,
                sparse[c] if session.reads_sparse else None, anchor_state=state,
                anchor_state_mask=mask[0, 0], action_mask=mask, generator=gen,
            )
            finite = bool(torch.isfinite(action).all() and torch.isfinite(video).all())
            rows.append({"decision": c, "commit_s": None if commit_s is None else round(commit_s, 3),
                         "generate_s": round(generate_s, 3), "finite": finite})
            outs.append((action.float().cpu(), video.float().cpu()))
        results[form] = outs
        report[f"form_{form}"] = {"decisions": rows, "peak_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
    if len(results) == 2:
        report["streaming_vs_full"] = [
            {"decision": c, "action_rel_rmse": relative_rmse(results[DEPLOY_STREAMING][c][0], results[DEPLOY_FULL][c][0]),
             "video_rel_rmse": relative_rmse(results[DEPLOY_STREAMING][c][1], results[DEPLOY_FULL][c][1])}
            for c in range(args.decisions)
        ]


def run_vae(args, report: dict) -> None:
    from sana_wam_min.vae import load_vae

    device = torch.device(args.device)
    bundle = load_vae(args.vae_path, device=device, dtype=torch.bfloat16)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(3, 320, 512), dtype=np.uint8)
    canvases = [torch.from_numpy(np.roll(base, shift=3 * i, axis=2).copy()) for i in range(1 + 8 * args.dense_chunks)]
    (one_shot, once_s) = synced(encode_once, bundle, canvases)
    stream = LatentStream(bundle)
    parts, times = [], []
    out, s = synced(stream.start, canvases[0])
    parts.append(out)
    times.append(s)
    for j in range(args.dense_chunks):
        encode_once(bundle, canvases[:1])          # an observation encode in between, as in deploy
        out, s = synced(stream.extend, canvases[1 + 8 * j : 9 + 8 * j])
        parts.append(out)
        times.append(s)
    streamed = torch.cat(parts, dim=2)
    sparse_frames = canvases[:17]
    sparse_once = encode_once(bundle, sparse_frames)
    sparse_stream = LatentStream(bundle)
    sparse_streamed = torch.cat(
        [sparse_stream.start(sparse_frames[0]), sparse_stream.extend(sparse_frames[1:9]), sparse_stream.extend(sparse_frames[9:17])],
        dim=2,
    )
    report["vae"] = {
        "dense_frames": len(canvases),
        "latent_shape": list(one_shot.shape),
        "dense_stream_maxdiff_vs_one_shot": float((streamed.float() - one_shot.float()).abs().max()),
        "sparse_stream_maxdiff_vs_one_shot": float((sparse_streamed.float() - sparse_once.float()).abs().max()),
        "one_shot_s": round(once_s, 3),
        "stream_step_s": [round(t, 3) for t in times],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--decisions", type=int, default=4)
    parser.add_argument("--dense-chunks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report: dict = {"device": torch.cuda.get_device_name(torch.device(args.device))}
    run_model(args, report)
    if args.vae_path:
        run_vae(args, report)
    print(json.dumps(report, indent=1))
    ok = all(row["finite"] for key, value in report.items() if key.startswith("form_") for row in value["decisions"])
    if "vae" in report:
        ok = ok and report["vae"]["dense_stream_maxdiff_vs_one_shot"] == 0.0 and report["vae"]["sparse_stream_maxdiff_vs_one_shot"] == 0.0
    print("FIRST LIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
