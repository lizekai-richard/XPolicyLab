"""Real-weight decode check of a ``latent_action`` checkpoint on one GPU, without the simulator.

Drives the episode protocol as ``model.py`` does (the per-view streams opened at the first frame, the sink committed,
every chunk generated from its own observation latent and then committed from the frames that followed it) at the
deployment sampling (steps, video CFG / action CFG, action flow shift), and captures every chunk's denoised latent
action cells ``[1, V * 80, C_lat]`` right before ``generate_chunk`` decodes them. Per chunk it reports:

- copy spread: the RMS deviation of every cell from the mean of its value's copies (``C_lat * 80 / (C * 80)`` copies a
  frame, times V views), relative to the RMS of the cells -- small when the model writes one value into all its copies
  and the decoding groups exactly those copies. Controls, groupings that pair cells of different values: the views
  misaligned (views 1.. read with their 80 tokens permuted), the channel copies misaligned (channels C_lat / 2.. rolled
  by C / 2 steps) and the cells read token-major (token p channel c as value (p * C_lat + c) mod C * 80);
- view agreement: every view's frame decoded alone against the other views';
- smoothness: RMS of the decoded trajectory's second differences over its first differences (live non-gripper slots),
  for the decoding (values step-major, j = s * 80 + d) and for a dim-major reading (j = d * C + s) as control;
- the gripper slots before the clamp, the first action against the anchor state, the chunk's travel, and the largest
  gap between the served chunk and the decoding of the captured cells (grippers clamped, in the served dtype: 0).
``--out`` also saves every chunk's cells and served actions next to the JSON (``.pt``).

Frames: ``--frames-from`` an .npz with ``frames`` [T, 3, H, W, 3] uint8 (cam_head, cam_left_wrist, cam_right_wrist, one
frame per control tick), else synthetic frames (one random image rolled by the tick). The anchor state is fixed at
the normalized origin of the live EEF slots, grippers open: the check reads the structure of the decoded actions, not
their task.

Usage (from policy/SANA_WAM_CAUSAL, PYTHONPATH=<XPolicyLab parent>):
  python tools/latent_action_decode_check.py --checkpoint-dir DIR --text-encoder-path P --vae-path P
         [--frames-from NPZ] [--chunks 8] [--steps 10] [--video-cfg 6] [--action-cfg 1] [--action-flow-shift 1]
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

from sana_wam_causal import session as session_module  # noqa: E402
from sana_wam_causal import shared  # noqa: E402,F401
from sana_wam_causal.contract import LATENT_ACTION_TOKENS  # noqa: E402
from sana_wam_causal.model import untile_latent_actions  # noqa: E402
from sana_wam_causal.runtime import CausalRuntime  # noqa: E402
from sana_wam_min.robot80 import LEFT_GRIPPER, RIGHT_GRIPPER, normalized_gripper_bounds  # noqa: E402

LIVE = list(range(7, 17)) + list(range(36, 46))   # the two EEF arms: xyz, rot6d, gripper
GRIPPERS = (LEFT_GRIPPER, RIGHT_GRIPPER)
POSITIONS = [7, 8, 9, 36, 37, 38]


def rms(x: torch.Tensor) -> float:
    return float(x.double().pow(2).mean().sqrt())


def per_view_tokens(cells: torch.Tensor) -> torch.Tensor:
    """[1, V * 80, C_lat] -> [V, 80, C_lat]."""

    return cells[0].reshape(-1, LATENT_ACTION_TOKENS, cells.shape[-1])


def channel_major(frames: torch.Tensor) -> torch.Tensor:
    """[V, 80, C_lat] -> [V, C_lat * 80] in the tiling's cell order (cell c * 80 + p: token p, channel c)."""

    return frames.transpose(1, 2).reshape(frames.shape[0], -1)


def grouped(flat: torch.Tensor, values: int) -> tuple[torch.Tensor, float]:
    """Cells [V, N] read as value (i mod ``values``): (every value's copies' mean [values], RMS deviation of the cells
    from their value's mean relative to the RMS of the cells)."""

    flat = flat.double()
    index = torch.arange(flat.shape[1]) % values
    sums = torch.zeros(values, dtype=torch.float64).index_add(0, index.repeat(flat.shape[0]), flat.reshape(-1))
    means = sums / (torch.bincount(index, minlength=values).double() * flat.shape[0])
    return means.float(), rms(flat - means[index]) / max(rms(flat), 1e-12)


def smoothness(actions: torch.Tensor) -> float:
    """RMS of the second differences over the first differences of the live non-gripper slots of [C, 80]."""

    slots = [s for s in LIVE if s not in GRIPPERS]
    a = actions[:, slots].double()
    return rms(a[2:] - 2 * a[1:-1] + a[:-2]) / max(rms(a[1:] - a[:-1]), 1e-12)


def synthetic_frames(seed: int, ticks: int) -> np.ndarray:
    base = np.random.default_rng(seed).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    return np.stack([np.stack([np.roll(base, shift=t * (i + 1), axis=1) for i in range(3)]) for t in range(ticks)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--text-encoder-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--frames-from", default=None)
    parser.add_argument("--instruction", default="Put the bottles into the dustbin.")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--video-cfg", type=float, default=6.0)
    parser.add_argument("--action-cfg", type=float, default=1.0)
    parser.add_argument("--action-flow-shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="write the per-chunk report as JSON here")
    args = parser.parse_args()

    start = time.perf_counter()
    runtime = CausalRuntime.from_paths(
        args.checkpoint_dir, text_encoder_path=args.text_encoder_path, vae_path=args.vae_path, device=args.device,
        steps=args.steps, video_cfg_scale=args.video_cfg, action_cfg_scale=args.action_cfg,
        action_flow_shift=args.action_flow_shift,
    )
    contract = runtime.contract
    report = {k: v for k, v in runtime.load_report.items() if k != "stripped"}
    print(f"[decode-check] load {time.perf_counter() - start:.1f}s report={json.dumps(report, default=str)}")
    print(f"[decode-check] contract {contract.describe()} | steps {runtime.frontend.steps} "
          f"cfg {runtime.frontend.video_cfg_scale}/{runtime.frontend.action_cfg_scale} "
          f"shift {runtime.frontend.flow_shift}/{runtime.frontend.action_flow_shift}")
    if not contract.latent_action:
        raise SystemExit("not a latent_action checkpoint")
    if contract.first_chunk != 0 or not contract.causal_observation:
        raise SystemExit("this check drives the first-frame causal_observation form only")
    device = runtime.device
    chunk, stride = contract.actions_per_chunk, contract.video_frame_stride
    ticks = args.chunks * chunk + 1
    if args.frames_from:
        frames = np.load(args.frames_from)["frames"]
        if frames.shape[0] < ticks:
            raise SystemExit(f"{args.frames_from} holds {frames.shape[0]} ticks, {args.chunks} chunks need {ticks}")
    else:
        frames = synthetic_frames(args.seed + 7, ticks)

    mask = torch.zeros(80, dtype=torch.bool)
    mask[LIVE] = True
    action_mask = mask.reshape(1, 1, 80).expand(1, chunk, 80).contiguous().to(device)
    anchor = torch.zeros(80, device=device)   # the normalized origin of the live slots, grippers open
    bounds = normalized_gripper_bounds(runtime.frontend.normalization)
    print(f"[decode-check] gripper bounds (normalized) {bounds}")

    captured = []
    decode = session_module.untile_latent_actions

    def capture(latents, steps, dims, *rest, **kw):
        captured.append(latents.detach().to(device="cpu", dtype=torch.float32))
        return decode(latents, steps, dims, *rest, **kw)

    session_module.untile_latent_actions = capture
    session = runtime.new_session(args.instruction)
    streams = runtime.new_streams()
    sink, frozen = streams.start(runtime.view_frames(list(frames[0])))
    assert frozen is None
    session.view_shapes = streams.view_shapes if runtime.strip else None
    session.commit_sink(sink)
    g = torch.Generator().manual_seed(args.seed)
    perms = [torch.arange(LATENT_ACTION_TOKENS)] + [torch.randperm(LATENT_ACTION_TOKENS, generator=g) for _ in range(7)]
    rows, saved, previous = [], [], None
    for c in range(args.chunks):
        if c > 0:
            ticks_c = range((c - 1) * chunk + stride, c * chunk + 1, stride)
            views_t = [runtime.view_frames(list(frames[t])) for t in ticks_c]
            per_view = [[views[v] for views in views_t] for v in range(len(views_t[0]))]
            latent = streams.encode_chunk(per_view)
            session.commit_chunk(latent, previous, anchor_state=anchor, anchor_state_mask=mask.to(device),
                                 action_mask=action_mask)
        observation = sink if c == 0 else streams.encode_observation(runtime.view_frames(list(frames[c * chunk])))
        generator = torch.Generator(device=device).manual_seed(1000 + args.seed * 100 + c)
        t0 = time.perf_counter()
        action, _ = session.generate_chunk(
            anchor_state=anchor, anchor_state_mask=mask.to(device), action_mask=action_mask, generator=generator,
            gripper_bounds=bounds, observation=observation,
        )
        seconds = time.perf_counter() - t0
        previous = action
        cells = captured[-1]
        values = chunk * 80
        frames_v = per_view_tokens(cells)
        views = frames_v.shape[0]
        decoded = untile_latent_actions(cells, chunk, 80)[0]           # before the gripper clamp
        means, spread = grouped(channel_major(frames_v), values)
        assert torch.allclose(means.reshape(chunk, 80), decoded, atol=1e-5), "grouped() is not the decoding"
        misaligned_views = torch.stack([frames_v[v][perms[v % len(perms)]] for v in range(views)])
        rolled = frames_v.clone()
        half = rolled.shape[-1] // 2
        rolled[..., half:] = rolled[..., half:].roll(chunk // 2, dims=-1)   # those copies' step s read as s + C / 2
        _, spread_views = grouped(channel_major(misaligned_views), values) if views > 1 else (None, float("nan"))
        _, spread_copies = grouped(channel_major(rolled), values)
        token_means, spread_token_major = grouped(frames_v.reshape(views, -1), values)
        dim_major = means.reshape(80, chunk).T                          # value j read as (d = j // C, s = j % C)
        served = action[0].detach().float().cpu()
        signal = rms(decoded[:, LIVE])
        per_view = [untile_latent_actions(cells[:, v * LATENT_ACTION_TOKENS:(v + 1) * LATENT_ACTION_TOKENS], chunk, 80)[0]
                    for v in range(views)]
        view_gap = (max(rms((per_view[a] - per_view[b])[:, LIVE]) for a in range(views) for b in range(a + 1, views))
                    / signal) if views > 1 else float("nan")
        expected = decoded.clone()   # the decoding, its grippers clamped, in the served dtype
        for slot, (low, high) in zip(GRIPPERS, bounds):
            expected[:, slot] = expected[:, slot].clamp(float(low), float(high))
        gap = (served - expected.to(action.dtype).float()).abs()
        worst = divmod(int(gap.argmax()), 80)
        row = {
            "chunk": c, "generate_s": round(seconds, 3), "views": views, "signal_rms": signal,
            "copy_spread": spread, "copy_spread_views_misaligned": spread_views,
            "copy_spread_copies_misaligned": spread_copies, "copy_spread_token_major": spread_token_major,
            "view_gap": view_gap,
            "smoothness": smoothness(decoded), "smoothness_dim_major": smoothness(dim_major),
            "smoothness_token_major": smoothness(token_means.reshape(chunk, 80)),
            "gripper_min": float(decoded[:, list(GRIPPERS)].min()), "gripper_max": float(decoded[:, list(GRIPPERS)].max()),
            "first_action_vs_anchor_pos": rms(decoded[0, POSITIONS] - anchor.cpu()[POSITIONS]),
            "chunk_travel_pos": rms(decoded[-1, POSITIONS] - decoded[0, POSITIONS]),
            "served_gap_max": float(gap.max()), "served_gap_at": [worst[0], worst[1]],
            "served_dtype": str(action.dtype), "finite": bool(torch.isfinite(cells).all()),
        }
        rows.append(row)
        saved.append({"cells": cells, "served": served, "decoded": decoded})
        print("[decode-check] chunk {chunk}: {generate_s}s V={views} signal={signal_rms:.3f} | copy spread {copy_spread:.4f} "
              "(views misaligned {copy_spread_views_misaligned:.3f}, copies misaligned {copy_spread_copies_misaligned:.3f}, "
              "token-major {copy_spread_token_major:.3f}) | view gap {view_gap:.4f} | smoothness {smoothness:.3f} "
              "(dim-major {smoothness_dim_major:.3f}, token-major {smoothness_token_major:.3f}) | gripper "
              "[{gripper_min:.3f}, {gripper_max:.3f}] | a0-anchor pos {first_action_vs_anchor_pos:.3f} travel "
              "{chunk_travel_pos:.3f} | served gap {served_gap_max:.2e} at {served_gap_at} ({served_dtype}) | finite "
              "{finite}".format(**row), flush=True)
    session_module.untile_latent_actions = decode

    def median(key):
        return float(np.median([r[key] for r in rows]))

    summary = {key: median(key) for key in (
        "copy_spread", "copy_spread_views_misaligned", "copy_spread_copies_misaligned", "copy_spread_token_major",
        "view_gap", "smoothness", "smoothness_dim_major", "smoothness_token_major", "served_gap_max")}
    print(f"[decode-check] medians {json.dumps(summary)}")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "contract": contract.describe(), "rows": rows,
                                              "medians": summary}, indent=2))
        torch.save(saved, str(Path(args.out).with_suffix(".pt")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
