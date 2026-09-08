#!/usr/bin/env python
# ---------------------------------------------------------------------------------------------
# SANA-SIDE TOOL. This script imports the Sana repo (dev.*, diffusion.*) and is NOT part of the
# sana_wam_min import graph; it lives under tools/ only so that the Sana-free companion
# ``compare_to_sana_reference.py`` can find its output format next to it. Run it as:
#
#     cd /home/zekail/Sana/Sana-merge
#     PYTHONPATH=/home/zekail/Sana/Sana-merge \
#     HF_HOME=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/hf_cache \
#     python /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/XPolicyLab/policy/SANA_WAM/tools/sana_parity_reference.py
#
# The cwd must be the worktree root: the ``gemma-2-2b-it-local`` text-encoder path in
# diffusion/model/builder.py is relative (output/pretrained_models/text_encoder/gemma-2-2b-it).
# ---------------------------------------------------------------------------------------------
"""Produce Sana deploy-session reference predictions for the sana_wam_min parity gate.

For each selected bundle sample (and optionally synthetic observations) the Sana
``PolicyInferenceSession`` (dev/rwm/simulation_evaluation/policy_runtime/inference_session.py)
is fed one observation (frame 0 of each view, model-domain state80, action mask, wire prompt)
with a fixed diffusion seed, and everything the Sana-free side needs to replay the identical
call is saved to ``sana_reference_<sample>.pt``: frames, raw + normalized state, instruction,
the caption embeds/mask, the packed clean strip, the (re-drawn) noise tensors, the normalized
``action80`` the session returns and its raw absolute conversion.

Raw state recovery: the bundle stores ``raw_info["state80"]`` ([25, 80], un-normalized, masked)
and ``raw_info["state_mask80"]``; row 0 is the observation-time raw state. The tool re-normalizes
it with the pinned artifact through Sana's ``normalize_robot80_affine`` and records the max
abs difference against the bundle's ``initial_state80_normalized`` (expected 0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402

PORT_ROOT = Path("/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/xpolicylab_sana_wam_port_20260908")
DEFAULT_REPO = Path("/home/zekail/Sana/Sana-merge")
DEFAULT_BUNDLE_DIR = PORT_ROOT / "bundles" / "s35000"
DEFAULT_CONFIG = PORT_ROOT / "ckpt_snapshots" / "config.yaml"
DEFAULT_CHECKPOINT = PORT_ROOT / "ckpt_snapshots" / "epoch_6_step_35000"
DEFAULT_NORMALIZATION = PORT_ROOT / "ckpt_snapshots" / "normalization" / "robodojo_arx_x5_model_fps_25_f25_normalization.json"
DEFAULT_OUT_DIR = PORT_ROOT / "logs"
REFERENCE_FORMAT = "xpolicylab_sana_wam_sana_reference_v1"

SYNTHETIC_INSTRUCTION = "Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it."
SYNTHETIC_EMBODIMENT = "dual-arm RoboDojo ARX-X5 robot with parallel grippers"
SYNTHETIC_HEIGHT = 480
SYNTHETIC_WIDTH = 640
JOINT_ONLY_ACTIVE_SLOTS = (0, 1, 2, 3, 4, 5, 16, 29, 30, 31, 32, 33, 34, 45)
GRIPPER_SLOTS = (16, 45)
NUM_VIEWS = 3
FPS = 25.0
NUM_FRAMES = 25


def log(message: str) -> None:
    print(f"[sana_parity_reference] {time.strftime('%H:%M:%S')} {message}", flush=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def resolve_checkpoint_file(path: Path) -> Path:
    if path.is_dir():
        return path / "model" / "pytorch_model_fsdp.bin"
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--samples", default="0,1,2", help="comma-separated bundle sample positions ('' for none)")
    parser.add_argument("--synthetic", type=int, default=1, help="number of synthetic seeded observations (480x640 frames)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="checkpoint dir or pytorch_model_fsdp.bin")
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20_260_908, help="diffusion seed shared by every saved sample")
    parser.add_argument("--steps", type=int, default=None, help="default: train.extra.rwm_validation_steps")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--flow-shift", type=float, default=None, help="default: scheduler.inference_flow_shift")
    return parser.parse_args()


# --------------------------------------------------------------------------- observations


def observation_from_bundle(bundle: dict, artifact: dict, PolicyObservation, normalize_robot80_affine) -> dict:
    """Frame 0 of each view (uint8 CHW at the 256x320 training size), model-domain state, wire prompt."""

    frames = tuple(bundle["pixel_views_uint8"][v][0].contiguous() for v in range(NUM_VIEWS))
    state80_model = bundle["initial_state80_normalized"][0].clone().float()
    state_mask80 = bundle["initial_state_condition_mask80"][0].clone().bool()
    action_mask80 = bundle["action_mask"][0, 0].clone().bool()
    raw_info = bundle["raw_info"]
    state80_raw = torch.as_tensor(raw_info["state80"])[0].clone().float()
    raw_mask = torch.as_tensor(raw_info["state_mask80"])[0].clone().bool()
    renormalized = normalize_robot80_affine(
        state80_raw, raw_mask, artifact["state_center80"], artifact["state_scale80"], artifact["state_normalization_mask80"]
    )
    prompt_rows = [str(t) for t in bundle["prompt"]["conditional"]]
    wire = "\n\n".join(prompt_rows)
    obs = PolicyObservation(
        frames=frames,
        camera_intrinsics=torch.eye(3, dtype=torch.float64).expand(NUM_VIEWS, 3, 3).contiguous(),
        state80=state80_model,
        state_mask80=state_mask80,
        action_mask80=action_mask80,
        fps=FPS,
        instruction=wire,
        view_initial_pose=None,
        pinned_view_index=0,
    )
    return {
        "kind": "bundle",
        "name": f"sample_{int(bundle['sample_position']):03d}",
        "obs": obs,
        "frames_uint8_chw": [f.clone() for f in frames],
        "frames_note": "bundle pixel_views_uint8[v][0]: frame 0 after the training ResizeCrop (256x320), uint8 of clamp(127.5*x+127.5)",
        "state80_raw": state80_raw,
        "state_mask80": raw_mask,
        "state80_model": state80_model,
        "state_renormalized_max_abs_diff": float((renormalized - state80_model).abs().max()),
        "action_mask80": action_mask80,
        "instruction": str(raw_info.get("instruction", "")),
        "prompt_rows": prompt_rows,
        "instruction_wire": wire,
        "bundle_meta": {
            "sample_position": int(bundle["sample_position"]),
            "item_id": str(bundle["item_id"]),
            "dataset_index": int(bundle["dataset_index"]),
            "noise_seed": int(bundle["noise_seed"]),
            "reference_action_mse": float(bundle["reference"]["manifest"]["action_mse"]),
        },
    }


def synthetic_observation(index: int, artifact: dict, PolicyObservation, normalize_robot80_affine, render_prompt) -> dict:
    """Seeded random 480x640 frames and an in-range raw state; exercises the full pixel front-end."""

    generator = torch.Generator().manual_seed(20_260_908 + index)
    frames = tuple(
        (torch.rand((3, SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH), generator=generator) * 255.0).to(torch.uint8)
        for _ in range(NUM_VIEWS)
    )
    mask = torch.zeros(80, dtype=torch.bool)
    mask[list(JOINT_ONLY_ACTIVE_SLOTS)] = True
    center = torch.as_tensor(artifact["state_center80"], dtype=torch.float32)
    scale = torch.as_tensor(artifact["state_scale80"], dtype=torch.float32)
    offsets = (torch.rand(80, generator=generator) - 0.5) * 0.6  # within +-0.3 scale of the center
    state80_raw = torch.where(mask, center + offsets * scale, torch.zeros(80))
    for slot in GRIPPER_SLOTS:
        state80_raw[slot] = 0.25 + 0.5 * float(torch.rand((), generator=generator))  # closedness in [0,1]
    state80_model = normalize_robot80_affine(
        state80_raw, mask, artifact["state_center80"], artifact["state_scale80"], artifact["state_normalization_mask80"]
    ).float()
    wire = render_prompt(SYNTHETIC_EMBODIMENT, SYNTHETIC_INSTRUCTION)
    obs = PolicyObservation(
        frames=frames,
        camera_intrinsics=torch.eye(3, dtype=torch.float64).expand(NUM_VIEWS, 3, 3).contiguous(),
        state80=state80_model,
        state_mask80=mask,
        action_mask80=mask.clone(),
        fps=FPS,
        instruction=wire,
        view_initial_pose=None,
        pinned_view_index=0,
    )
    return {
        "kind": "synthetic",
        "name": f"synthetic_{index:03d}",
        "obs": obs,
        "frames_uint8_chw": [f.clone() for f in frames],
        "frames_note": "seeded torch.rand uint8 RGB frames at the 480x640 source size",
        "state80_raw": state80_raw,
        "state_mask80": mask,
        "state80_model": state80_model,
        "state_renormalized_max_abs_diff": 0.0,
        "action_mask80": mask.clone(),
        "instruction": SYNTHETIC_INSTRUCTION,
        "prompt_rows": list(wire.split("\n\n")),
        "instruction_wire": wire,
        "bundle_meta": None,
    }


# --------------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if Path.cwd().resolve() != repo:
        log(f"WARNING: cwd {Path.cwd()} != worktree {repo}; the relative gemma-2-2b-it-local path needs cwd == worktree")

    from dev.rwm.diffusion.data.robot80_normalization import (
        denormalize_robot80_affine,
        load_robot80_affine_normalization,
        normalize_robot80_affine,
    )
    from dev.rwm.diffusion.inference.policy_io import PolicyObservation
    from dev.rwm.simulation_evaluation.policy_runtime.in_process import (
        _clip_predicted_gripper_closedness,
        _reconstruct_absolute_joint_targets,
    )
    from dev.rwm.simulation_evaluation.policy_runtime.inference_session import PolicyInferenceSession
    from dev.rwm.simulation_evaluation.robodojo.common.session_contract import render_robodojo_policy_prompt
    from diffusion.model.builder import vae_encode

    device = torch.device(args.device)
    ckpt_file = resolve_checkpoint_file(args.checkpoint)
    log(f"torch {torch.__version__} device={device} ckpt={ckpt_file}")

    artifact = load_robot80_affine_normalization(args.normalization)
    normalization_sha256 = sha256_file(args.normalization)
    joint_target_mode = str(artifact["joint_target_mode"])
    log(f"normalization sha256={normalization_sha256} joint_target_mode={joint_target_mode}")

    t0 = time.time()
    session = PolicyInferenceSession.from_config(
        str(args.config),
        str(ckpt_file),
        device=device,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        load_text_encoder=True,
        vae_decode_video=False,
    )
    log(f"session ready in {time.time() - t0:.1f}s: steps={session.steps} cfg={session.cfg_scale} flow_shift={session.flow_shift}")

    items: list[dict] = []
    wanted = [int(v) for v in args.samples.split(",") if v.strip()]
    for position in wanted:
        path = args.bundle_dir / f"sample_{position:03d}.pt"
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        items.append(observation_from_bundle(bundle, artifact, PolicyObservation, normalize_robot80_affine))
    for index in range(int(args.synthetic)):
        items.append(synthetic_observation(index, artifact, PolicyObservation, normalize_robot80_affine, render_robodojo_policy_prompt))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git_head": git_head(repo),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "checkpoint_file": str(ckpt_file),
        "normalization_path": str(args.normalization),
        "normalization_sha256": normalization_sha256,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    index_rows = []
    failures = 0
    frames_count, k_actions = session.tier(FPS, NUM_FRAMES)
    latent_frames = (frames_count - 1) // 8 + 1
    for item in items:
        obs = item["obs"]
        t_item = time.time()
        try:
            with torch.inference_mode():
                # Same calls predict makes (memoized prompt embeds are reused by predict; the VAE encode
                # is deterministic, and predict's frame-0 re-pin lets us cross-check the strip below).
                caption_embeds, caption_mask, _, _ = session._embed_token_group_prompt(obs.instruction, NUM_VIEWS)
                clean_strip, view_shapes, target_sizes = session._encode_observation_views(obs, latent_frames, vae_encode)
                generator = torch.Generator(device=device).manual_seed(int(args.seed))
                result = session.predict(obs, num_frames=NUM_FRAMES, generator=generator, vae_decode_video=False)
                # Re-draw the noise in predict's order (video first in the strip dtype, then fp32 action noise).
                regen = torch.Generator(device=device).manual_seed(int(args.seed))
                video_noise = torch.randn(clean_strip.shape, device=device, dtype=clean_strip.dtype, generator=regen)
                action_noise = torch.randn((1, k_actions, 80), device=device, dtype=torch.float32, generator=regen)

            action80_model = result.action80.detach().to("cpu", torch.float32)  # [K, 80] normalized
            mask_k = np.broadcast_to(item["action_mask80"].numpy(), action80_model.shape).copy()
            action80_raw = denormalize_robot80_affine(
                action80_model, torch.from_numpy(mask_k), artifact["action_center80"], artifact["action_scale80"],
                artifact["action_normalization_mask80"],
            ).numpy()
            if joint_target_mode == "anchor_delta":
                action80_raw = _reconstruct_absolute_joint_targets(
                    action80_raw, mask_k,
                    anchor_state80_raw=item["state80_raw"].numpy(),
                    anchor_state_mask80=item["state_mask80"].numpy(),
                )
            action80_raw = _clip_predicted_gripper_closedness(np.array(action80_raw, dtype=np.float32), mask_k)
            strip_frame0_repinned = bool(torch.equal(result.video_latent[:, :, :1].cpu(), clean_strip[:, :, :1].cpu()))

            out_path = args.out_dir / f"sana_reference_{item['name']}.pt"
            payload = {
                "format": REFERENCE_FORMAT,
                "kind": item["kind"],
                "name": item["name"],
                "bundle_meta": item["bundle_meta"],
                "seed": int(args.seed),
                "steps": int(session.steps),
                "cfg_scale": float(session.cfg_scale),
                "flow_shift": float(session.flow_shift),
                "fps": FPS,
                "num_frames": int(frames_count),
                "k_actions": int(k_actions),
                "frames_uint8_chw": item["frames_uint8_chw"],
                "frames_note": item["frames_note"],
                "target_sizes_hw": [list(map(int, s)) for s in target_sizes],
                "instruction": item["instruction"],
                "instruction_wire": item["instruction_wire"],
                "prompt_rows": item["prompt_rows"],
                "state80_raw": item["state80_raw"],
                "state_mask80": item["state_mask80"],
                "state80_model": item["state80_model"],
                "state_renormalized_max_abs_diff": item["state_renormalized_max_abs_diff"],
                "action_mask80": item["action_mask80"],
                "caption_embeds": caption_embeds.detach().cpu(),
                "caption_mask": caption_mask.detach().cpu(),
                "clean_strip": clean_strip.detach().cpu(),
                "view_latent_shapes": [list(map(int, s)) for s in view_shapes],
                "video_noise": video_noise.cpu(),
                "action_noise": action_noise.cpu(),
                "action80_model": action80_model,
                "action80_raw_absolute": torch.from_numpy(np.array(action80_raw, dtype=np.float32)),
                "joint_target_mode": joint_target_mode,
                "normalization_sha256": normalization_sha256,
                "video_latent": result.video_latent.detach().cpu(),
                "strip_frame0_repinned_equal": strip_frame0_repinned,
                "meta": {k: (list(v) if isinstance(v, tuple) else v) for k, v in result.meta.items()},
                "provenance": provenance,
            }
            torch.save(payload, out_path)
            log(
                f"{item['name']}: saved {out_path.name} in {time.time() - t_item:.1f}s | state renorm max|d|="
                f"{item['state_renormalized_max_abs_diff']:.3e} | strip frame0 repinned={strip_frame0_repinned} | "
                f"action80 |mean|={action80_model.abs().mean():.4f}"
            )
            index_rows.append({"name": item["name"], "file": out_path.name, "kind": item["kind"], "bundle_meta": item["bundle_meta"]})
        except Exception:  # noqa: BLE001 - keep going, report at the end
            failures += 1
            log(f"FAILED {item['name']}:\n{traceback.format_exc()}")

    index_path = args.out_dir / "sana_reference_index.json"
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump({"format": REFERENCE_FORMAT + "_index", "provenance": provenance, "seed": int(args.seed),
                   "samples": index_rows, "failures": failures}, handle, indent=2)
    log(f"wrote {index_path}; {len(index_rows)} references, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
