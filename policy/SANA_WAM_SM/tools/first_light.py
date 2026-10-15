"""GPU first light of the sparse-memory adapter WITHOUT a checkpoint: the real-dimension model with random weights (bf16,
the deploy's autocast), driven by the session for a few decisions in both deploy forms, plus (with ``--vae-path``) the
causal VAE encode of synthetic 320 x 512 canvases.

Checks: the forwards run on the real geometry (32 blocks, 160-token latent frames, the 193-token target chunk after a
prefix of 1 + n latents: n = 2 for memory_frames 16, n growing with t for a strided memory) under bf16 autocast;
per-decision latencies and peak memory; finite outputs; cached vs uncached on the same decision (equal up to bf16
round-off); the prefix encoded alone == the first n + 1 latents of the whole sample's encode on this GPU and dtype (Sana
``test_sparse_memory_vae.py`` asserts maxdiff 0 with deterministic cuDNN, which this tool sets).

Usage (from policy/SANA_WAM_SM, PYTHONPATH=<XPolicyLab parent>):
  python tools/first_light.py --recipe tests/fixtures/recipe_sm_fixed_m16_3fa3e9689.yaml [--vae-path P]
         [--decisions 4] [--first-decision 0] [--steps 10] [--device cuda]
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

from sana_wam_sm import shared  # noqa: E402,F401
from sana_wam_sm import sparse_memory as sm  # noqa: E402
from sana_wam_sm.contract import DEPLOY_CACHED, DEPLOY_UNCACHED, bidirectional_view, resolve_sm_contract  # noqa: E402
from sana_wam_sm.model import build_sm_policy  # noqa: E402
from sana_wam_sm.session import Caption, SMSession  # noqa: E402
from sana_wam_sm.vae_encode import encode_causal  # noqa: E402
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
    contract = resolve_sm_contract(cfg)
    view = bidirectional_view(cfg)
    rope, origin, _ = resolve_rope_contract(view, None)
    config = policy_config_from_train_config(view, rope=rope, legacy_origin=origin)
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = build_sm_policy(config, contract, dtype=torch.float32, device="cpu")
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
    decisions = range(args.first_decision, args.first_decision + args.decisions)
    latents = {c: sm.memory_latents(sm.decision_row(c), 0, contract.memory) for c in decisions}
    prefixes = {c: randn(1, config.in_channels, 1 + latents[c], 10, 16) for c in decisions}
    state = randn(80)
    mask = torch.zeros(1, 32, 80, dtype=torch.bool, device=device)
    mask[..., 7:17] = True
    mask[..., 36:46] = True
    results = {}
    for form in (DEPLOY_CACHED, DEPLOY_UNCACHED):
        session = SMSession(
            model, caption=caption, unconditional_caption=null, steps=args.steps, flow_shift=5.0, action_flow_shift=1.0,
            video_cfg_scale=6.0, action_cfg_scale=1.0, model_fps=contract.model_fps, deploy_form=form,
        )
        torch.cuda.reset_peak_memory_stats(device)
        rows, outs = [], []
        for c in decisions:
            gen = torch.Generator(device=device).manual_seed(100 + c)
            (action, video), decide_s = synced(
                session.decide, prefixes[c], anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask,
                generator=gen,
            )
            finite = bool(torch.isfinite(action).all() and torch.isfinite(video).all())
            rows.append({"decision": c, "memory_latents": latents[c], "decide_s": round(decide_s, 3), "finite": finite})
            outs.append((action.float().cpu(), video.float().cpu()))
        results[form] = outs
        report[f"form_{form}"] = {"decisions": rows, "peak_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
    report["cached_vs_uncached"] = [
        {"decision": c, "action_rel_rmse": relative_rmse(results[DEPLOY_CACHED][i][0], results[DEPLOY_UNCACHED][i][0]),
         "video_rel_rmse": relative_rmse(results[DEPLOY_CACHED][i][1], results[DEPLOY_UNCACHED][i][1])}
        for i, c in enumerate(decisions)
    ]


def run_vae(args, report: dict) -> None:
    from sana_wam_min.vae import load_vae

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    cfg = yaml.safe_load(open(args.recipe))
    memory = resolve_sm_contract(cfg).memory
    device = torch.device(args.device)
    bundle = load_vae(args.vae_path, device=device, dtype=torch.bfloat16)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(3, 320, 512), dtype=np.uint8)

    def canvas(row: int) -> torch.Tensor:
        return torch.from_numpy(np.roll(base, shift=3 * row, axis=2).copy())

    rows_out = []
    for c in (0, 1, 7, 40):
        t = sm.decision_row(c)
        sequence_rows = sm.sequence_rows(t, 0, memory)
        prefix_rows = sm.prefix_rows(t, 0, memory)
        sequence, sequence_s = synced(encode_causal, bundle, [canvas(row) for row in sequence_rows])
        prefix, prefix_s = synced(encode_causal, bundle, [canvas(row) for row in prefix_rows])
        n = sm.memory_latents(t, 0, memory)
        rows_out.append({
            "decision": c, "memory_latents": n, "prefix_frames": len(prefix_rows),
            "prefix_maxdiff_vs_sequence": float((prefix.float() - sequence[:, :, : n + 1].float()).abs().max()),
            "prefix_encode_s": round(prefix_s, 3), "sequence_encode_s": round(sequence_s, 3),
        })
    report["vae"] = {"memory": memory, "latent_shape": list(sequence.shape), "decisions": rows_out}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--decisions", type=int, default=4)
    parser.add_argument("--first-decision", type=int, default=0)
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
        ok = ok and all(row["prefix_maxdiff_vs_sequence"] == 0.0 for row in report["vae"]["decisions"])
    print("FIRST LIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
