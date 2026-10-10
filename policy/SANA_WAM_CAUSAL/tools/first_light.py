"""Real-weight first light of the causal adapter on one GPU, without the simulator.

Loads the runtime exactly as ``model.py`` does (causal yaml -> contract -> normalization -> bf16 causal model ->
VAE -> Gemma), then drives the per-chunk observation cycle on synthetic frames (one random image per episode, rolled by
the tick): for every chunk but the first commit the previous one (its 9 frames encoded as one clip), then encode the
chunk's start frame as its observation and generate it. Prints the load report, per-chunk latencies, the chunks read,
obs_anchor_gap (the observation vs the anchor latent of the same frame inside the next commit's clip: the causal-VAE
equality the protocol relies on), peak GPU memory and finiteness checks, and checks determinism (the same seeds
replay the same chunk 0).

Usage (from policy/SANA_WAM_CAUSAL, PYTHONPATH=<XPolicyLab parent>):
  python tools/first_light.py --checkpoint-dir DIR --text-encoder-path P --vae-path P [--allow-donor-checkpoint]
         [--chunks 4] [--steps 10] [--window-override N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR))

from sana_wam_causal import shared  # noqa: E402,F401
from sana_wam_causal.runtime import CausalRuntime, timed  # noqa: E402
from sana_wam_min.robot80 import normalize_state, normalized_gripper_bounds  # noqa: E402


def frames_at(base: np.ndarray, tick: int) -> list[np.ndarray]:
    """The three camera frames of one tick: the episode's base image rolled by the tick (per camera speed)."""

    return [np.roll(base, shift=tick * (i + 1), axis=1).copy() for i in range(3)]


def relative_rmse(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    estimate, reference = estimate.float(), reference.float()
    return float(((estimate - reference) ** 2).mean().sqrt() / (reference**2).mean().sqrt().clamp_min(1e-8))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--text-encoder-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--allow-donor-checkpoint", action="store_true")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--window-override", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    start = time.perf_counter()
    runtime = CausalRuntime.from_paths(
        args.checkpoint_dir, text_encoder_path=args.text_encoder_path, vae_path=args.vae_path, device=args.device,
        steps=args.steps, cfg_scale=6.0, action_cfg_scale=1.0, allow_donor_checkpoint=args.allow_donor_checkpoint,
    )
    print(f"[first-light] load {time.perf_counter() - start:.1f}s report={json.dumps({k: v for k, v in runtime.load_report.items() if k != 'stripped'}, default=str)}")
    contract = runtime.contract
    if args.window_override is not None:
        contract = type(contract)(**{**contract.__dict__, "sliding_window_chunks": args.window_override})
        runtime.contract = contract
    print(f"[first-light] contract {contract.describe()} | steps {runtime.frontend.steps} "
          f"cfg {runtime.frontend.video_cfg_scale}/{runtime.frontend.action_cfg_scale} "
          f"shift {runtime.frontend.flow_shift}/{runtime.frontend.action_flow_shift}")
    device = runtime.device
    normalization = runtime.frontend.normalization
    mask = torch.zeros(80, dtype=torch.bool)
    mask[7:17] = True
    mask[36:46] = True
    action_mask = mask.reshape(1, 1, 80).expand(1, contract.actions_per_chunk, 80).contiguous().to(device)
    state_raw = np.zeros(80, dtype=np.float32)
    state_raw[7:10] = (0.25, 0.0, 0.25)
    state_raw[36:39] = (0.25, 0.0, 0.25)
    state_raw[[10, 14, 39, 43]] = 1.0
    state_raw[[16, 45]] = 0.2
    anchor = normalize_state(state_raw, mask.numpy(), normalization).to(device)

    chunk, stride = contract.actions_per_chunk, contract.video_frame_stride

    def episode(seed: int, chunks: int):
        base = np.random.default_rng(seed).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        session = runtime.new_session("put the bottles into the dustbin")
        out = []
        for c in range(chunks):
            timings = {}
            gap = None
            if c:
                clip = [frames_at(base, (c - 1) * chunk + j * stride) for j in range(1 + chunk // stride)]
                (latent, anchor_latent), timings["encode"] = timed(runtime.encode_executed_chunk, clip)
                gap = relative_rmse(anchor_latent, out[-1][2])
                _, timings["commit"] = timed(session.commit_chunk, latent, out[-1][0], anchor_state=anchor,
                                             anchor_state_mask=mask.to(device), action_mask=action_mask)
            observation, timings["obs_encode"] = timed(runtime.encode_observation, frames_at(base, c * chunk))
            read, origin = session.read_chunk_ids(), session.softmax_origin()
            generator = torch.Generator(device=device).manual_seed(1000 + c)
            (action, video), timings["generate"] = timed(
                session.generate_chunk, observation, anchor_state=anchor, anchor_state_mask=mask.to(device),
                action_mask=action_mask, generator=generator, gripper_bounds=normalized_gripper_bounds(normalization),
            )
            finite = bool(torch.isfinite(action).all()) and bool(torch.isfinite(video).all())
            print(f"[first-light] chunk {c}: read={read}" + (f" origin={origin}" if origin else "") + " "
                  + " ".join(f"{k}={v:.3f}s" for k, v in timings.items())
                  + ("" if gap is None else f" obs_anchor_gap(chunk {c - 1})={gap:.2e}")
                  + f" finite={finite} |a|max={float(action.abs().max()):.3f} video_rms={float(video.float().pow(2).mean().sqrt()):.3f}")
            out.append((action, video, observation))
        return out

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    first = episode(7, args.chunks)
    if torch.cuda.is_available():
        print(f"[first-light] peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
              f"(reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB)")
    again = episode(7, 1)
    same = torch.equal(first[0][0], again[0][0]) and torch.equal(first[0][1], again[0][1])
    print(f"[first-light] chunk-0 replay deterministic: {same}")
    print("[first-light] PASS" if same and all(bool(torch.isfinite(a).all()) for a, _, _ in first) else "[first-light] FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
