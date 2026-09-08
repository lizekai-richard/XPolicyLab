#!/usr/bin/env python
"""Replay the trainer's 35-sample holdout validation from dumped bundles with sana_wam_min only.

Sana-FREE: imports nothing from the Sana repo. Each ``sample_NNN.pt`` bundle (written by the
Sana-side ``dump_validation_bundle.py``) carries the exact sampler inputs of one validation
sample plus the reference ``actions.pt``; this tool re-runs the ported model + sampler on them
and reports the masked action MSE against the reference.

Modes:
  --mode latent   feed the bundle's ``clean_video`` strip (tests model + sampler only)
  --mode pixels   re-encode ``pixel_views`` with the ported LTX2 VAE (tests the VAE front-end too)
  --mode both     run both and report both rows per sample
  --noise stored|regenerate   stored bundle noise, or a fresh CUDA generator seeded with noise_seed
  --text  stored|encode       stored Gemma embeds, or re-encode the bundle prompt rows
  --session-check             additionally run PolicyInferenceSession.predict_from_latent on the same
                              strip / noise (deploy-style data_info, rows encoded by the session) and
                              report its max|diff| against the direct replay; needs --text encode

The direct replay calls ``sample_policy`` with the bundle's trainer ``data_info`` (the exact
validation call); the session check exercises the integrator's deploy path on identical inputs.

Usage (1 GPU, sana-wam env, Sana NOT on PYTHONPATH):
    PYTHONPATH=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/XPolicyLab \\
    python holdout_replay.py --mode latent --noise stored --text stored \\
        --out .../logs/holdout_replay_latent_stored.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_ADAPTER_DIR = str(Path(__file__).resolve().parents[1])
if _ADAPTER_DIR not in sys.path:
    sys.path.insert(0, _ADAPTER_DIR)

PORT_ROOT = Path("/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/xpolicylab_sana_wam_port_20260908")
DEFAULT_BUNDLE_DIR = PORT_ROOT / "bundles" / "s35000"
DEFAULT_CHECKPOINT_DIR = PORT_ROOT / "ckpt_snapshots" / "epoch_6_step_35000"
DEFAULT_TRAIN_CONFIG = PORT_ROOT / "ckpt_snapshots" / "config.yaml"
DEFAULT_NORMALIZATION = PORT_ROOT / "ckpt_snapshots" / "normalization" / "robodojo_arx_x5_model_fps_25_f25_normalization.json"
DEFAULT_TEXT_ENCODER = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/models/Sana/text_encoder/gemma-2-2b-it"
DEFAULT_VAE = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/models/Sana/LTX-2.3-Diffusers"

# Aggregates of the reference manifests for the s35000 run (documented in spec_validation_bundle.md).
REFERENCE_AGGREGATES_S35000 = {"mean": 0.0219, "median": 0.0041}


def log(message: str) -> None:
    print(f"[holdout_replay] {time.strftime('%H:%M:%S')} {message}", flush=True)


# --------------------------------------------------------------------------- pure helpers


def masked_action_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Valid-slot-normalized action MSE of one ``[1, K, 80]`` sample, reduced exactly like the trainer.

    ``sum(((pred - target)^2) * mask) / sum(mask)`` in float32 over the flattened ``K x 80`` grid,
    then ``.mean()`` over the batch of one (``masked_action_loss(...).mean().item()``).
    """

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction.shape)}, target {tuple(target.shape)}, mask {tuple(mask.shape)}"
        )
    valid_per_sample = mask.flatten(1).sum(1)
    if bool(torch.any(valid_per_sample == 0)):
        raise ValueError("sample has no valid action slots")
    squared_error = (prediction.float() - target.float()).square()
    per_sample = (squared_error * mask).flatten(1).sum(1) / valid_per_sample
    return float(per_sample.mean().item())


def regenerate_noise(
    seed: int,
    video_shape: tuple[int, ...],
    video_dtype: torch.dtype,
    action_shape: tuple[int, ...],
    action_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw video noise first and action noise second from one seeded generator (trainer order)."""

    generator = torch.Generator(device=device).manual_seed(int(seed))
    video_noise = torch.randn(tuple(video_shape), device=device, dtype=video_dtype, generator=generator)
    action_noise = torch.randn(tuple(action_shape), device=device, dtype=action_dtype, generator=generator)
    return video_noise, action_noise


def move_data_info(data_info: dict, device: torch.device) -> dict:
    """Shallow-copy ``data_info`` with every tensor moved to ``device`` (non-tensors untouched)."""

    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data_info.items()}


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item())


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def list_bundles(bundle_dir: Path, samples: str | None) -> list[Path]:
    paths = sorted(bundle_dir.glob("sample_*.pt"))
    if samples:
        wanted = {int(v) for v in samples.split(",") if v.strip()}
        paths = [p for p in paths if int(p.stem.split("_")[1]) in wanted]
    return paths


# --------------------------------------------------------------------------- components


def build_components(args, device: torch.device) -> dict:
    from sana_wam_min.checkpoint import build_policy_model, load_policy_weights
    from sana_wam_min.config import (
        load_train_config,
        policy_config_from_train_config,
        sampling_defaults_from_train_config,
    )

    cfg = load_train_config(str(args.train_config))
    policy_config = policy_config_from_train_config(cfg)
    sampling = sampling_defaults_from_train_config(cfg)
    if args.steps is not None:
        sampling["steps"] = int(args.steps)
    if args.flow_shift is not None:
        sampling["flow_shift"] = float(args.flow_shift)
    if args.cfg_scale is not None:
        sampling["cfg_scale"] = float(args.cfg_scale)
    log(f"sampling: {sampling}")

    t0 = time.time()
    model = build_policy_model(policy_config, dtype=torch.bfloat16, device=device)
    load_report = load_policy_weights(model, str(args.checkpoint_dir), device=device)
    log(f"model ready in {time.time() - t0:.1f}s: {load_report}")

    tokenizer = text_encoder = None
    if args.text == "encode":
        from sana_wam_min.text import load_text_encoder

        t0 = time.time()
        tokenizer, text_encoder = load_text_encoder(str(args.text_encoder_path), device=device)
        log(f"text encoder ready in {time.time() - t0:.1f}s")

    vae = None
    if args.mode in ("pixels", "both"):
        from sana_wam_min.vae import load_vae

        t0 = time.time()
        vae = load_vae(str(args.vae_path), device=device, dtype=torch.bfloat16)
        log(f"VAE ready in {time.time() - t0:.1f}s")

    normalization = None
    if args.normalization_path:
        from sana_wam_min.robot80 import load_normalization

        expected = ((cfg.get("data") or {}).get("extra") or {}).get("robotwin_sft", {}).get("normalization_sha256")
        normalization = load_normalization(str(args.normalization_path), expected_sha256=expected)
        log(f"normalization sha256={normalization.sha256} joint_target_mode={normalization.joint_target_mode}")

    session = None
    if args.session_check:
        from sana_wam_min.session import PolicyInferenceSession

        session = PolicyInferenceSession(
            model=model,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            normalization=normalization,
            train_config=cfg,
            device=device,
            steps=sampling["steps"],
            cfg_scale=sampling["cfg_scale"],
            flow_shift=sampling["flow_shift"],
            checkpoint_path=str(args.checkpoint_dir),
        )
        log("session check enabled (predict_from_latent on the same components)")

    return {
        "cfg": cfg,
        "sampling": sampling,
        "model": model,
        "load_report": load_report,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "normalization": normalization,
        "session": session,
    }


def encode_pixel_views(vae, pixel_views: list[torch.Tensor], device: torch.device):
    """Re-encode ``V x [F, 3, H, W]`` float pixels in [-1, 1] into the packed strip exactly like validation."""

    from sana_wam_min.multiview import pack_multiview_latents
    from sana_wam_min.vae import encode_video

    latents = []
    for view in pixel_views:
        clip = view[None].permute(0, 2, 1, 3, 4).to(device=device, dtype=torch.bfloat16)  # [1,3,F,H,W]
        latents.append(encode_video(vae, clip))
    strip, shapes = pack_multiview_latents(latents)
    return strip, [list(map(int, s)) for s in shapes]


def encode_prompt(components: dict, rows: list[str], device: torch.device):
    from sana_wam_min.text import encode_prompt_rows

    return encode_prompt_rows([list(rows)], components["tokenizer"], components["text_encoder"], device)


def run_sampler(components: dict, device: torch.device, clean_video, video_noise, action_noise,
                clean_action, action_mask, caption_embeds, caption_mask, data_info):
    from sana_wam_min.sampler import sample_policy

    sampling = components["sampling"]
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        video, action = sample_policy(
            components["model"],
            clean_video,
            video_noise,
            action_noise,
            clean_action,
            action_mask,
            caption_embeds,
            caption_mask,
            None,
            None,
            sampling["cfg_scale"],
            data_info,
            sampling["steps"],
            sampling["flow_shift"],
        )
    return video, action


# --------------------------------------------------------------------------- per sample


def replay_sample(args, components: dict, device: torch.device, bundle_path: Path) -> list[dict]:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    position = int(bundle["sample_position"])
    seed = int(bundle["noise_seed"])
    manifest = bundle["reference"]["manifest"]
    reference_mse = float(manifest["action_mse"])
    reference_actions = bundle["reference"].get("actions_pt")

    clean_action = bundle["clean_action_normalized"].to(device)
    action_mask = bundle["action_mask"].to(device)
    data_info = move_data_info(bundle["data_info"], device)
    stored_video = bundle.get("clean_video")
    stored_shapes = bundle.get("clean_video_view_latent_shapes")

    common = {
        "sample_position": position,
        "item_id": bundle["item_id"],
        "noise_seed": seed,
        "reference_action_mse": reference_mse,
    }
    if reference_actions is not None:
        common["reference_target_matches_bundle"] = bool(
            torch.equal(reference_actions["target_normalized"].float(), bundle["clean_action_normalized"].float())
        )

    # text
    if args.text == "stored":
        caption_embeds = bundle["caption_embeds"].to(device)
        caption_mask = bundle["caption_mask"].to(device)
    else:
        caption_embeds, caption_mask = encode_prompt(components, bundle["prompt"]["conditional"], device)
        if "caption_embeds" in bundle:
            common["text_embed_max_abs_diff"] = max_abs_diff(caption_embeds, bundle["caption_embeds"])
            common["text_embed_bitwise"] = bool(torch.equal(caption_embeds.cpu(), bundle["caption_embeds"]))
            common["text_mask_equal"] = bool(torch.equal(caption_mask.cpu().long(), bundle["caption_mask"].long()))

    # noise (shape of the strip is the same for stored and re-encoded latents)
    if args.noise == "stored":
        video_noise = bundle["video_noise"].to(device)
        action_noise = bundle["action_noise"].to(device)
    else:
        video_dtype = stored_video.dtype if stored_video is not None else torch.bfloat16
        video_shape = tuple(stored_video.shape) if stored_video is not None else (1, 128, 4, 1, 240)
        video_noise, action_noise = regenerate_noise(
            seed, video_shape, video_dtype, tuple(clean_action.shape), clean_action.dtype, device
        )
        if "video_noise" in bundle:
            common["video_noise_max_abs_diff"] = max_abs_diff(video_noise, bundle["video_noise"])
            common["action_noise_max_abs_diff"] = max_abs_diff(action_noise, bundle["action_noise"])
            common["noise_bitwise"] = bool(
                torch.equal(video_noise.cpu(), bundle["video_noise"]) and torch.equal(action_noise.cpu(), bundle["action_noise"])
            )

    # state sanity: raw_info state80 row 0 normalized with the artifact must equal the model-domain state
    normalization = components["normalization"]
    raw_info = bundle.get("raw_info") or {}
    if normalization is not None and "state80" in raw_info:
        from sana_wam_min.robot80 import normalize_state

        renormalized = normalize_state(raw_info["state80"][0], raw_info["state_mask80"][0], normalization)
        common["state_renormalized_max_abs_diff"] = max_abs_diff(renormalized, bundle["initial_state80_normalized"][0])

    sources = []
    if args.mode in ("latent", "both"):
        sources.append(("latent", stored_video.to(device), stored_shapes, {}))
    if args.mode in ("pixels", "both"):
        strip, shapes = encode_pixel_views(components["vae"], bundle["pixel_views"], device)
        extra = {"view_latent_shapes": shapes}
        if stored_video is not None:
            extra["latent_max_abs_diff"] = max_abs_diff(strip, stored_video)
            extra["latent_bitwise"] = bool(torch.equal(strip.cpu(), stored_video))
            extra["latent_shapes_equal"] = shapes == [list(map(int, s)) for s in stored_shapes]
        sources.append(("pixels", strip, shapes, extra))

    rows = []
    for source, clean_video, shapes, extra in sources:
        t0 = time.time()
        video, action = run_sampler(
            components, device, clean_video, video_noise, action_noise, clean_action, action_mask,
            caption_embeds, caption_mask, data_info,
        )
        mse = masked_action_mse(action, clean_action, action_mask)
        row = dict(common)
        row.update(extra)
        row.update(
            {
                "source": source,
                "action_mse": mse,
                "action_mse_delta_vs_manifest": mse - reference_mse,
                "sampler_seconds": time.time() - t0,
                "video_finite": bool(torch.isfinite(video).all()),
            }
        )
        if reference_actions is not None:
            ref = reference_actions["sampled_normalized"].float()
            row["reference_max_abs_diff"] = max_abs_diff(action, ref)
            row["reference_bitwise"] = bool(torch.equal(action.float().cpu(), ref))
            row["reference_recomputed_mse"] = masked_action_mse(ref, reference_actions["target_normalized"], reference_actions["mask80"])
        session = components["session"]
        if session is not None:
            t1 = time.time()
            session_action, session_video = session.predict_from_latent(
                clean_video,
                [tuple(map(int, s)) for s in shapes],
                list(bundle["prompt"]["conditional"]),
                None,
                bundle["initial_state80_normalized"][0],
                bundle["initial_state_condition_mask80"][0],
                clean_action,
                action_mask,
                None,
                video_noise,
                action_noise,
            )
            row["session_action_mse"] = masked_action_mse(session_action, clean_action, action_mask)
            row["session_vs_direct_max_abs_diff"] = max_abs_diff(session_action, action)
            row["session_vs_direct_bitwise"] = bool(torch.equal(session_action.float().cpu(), action.float().cpu()))
            row["session_video_vs_direct_max_abs_diff"] = max_abs_diff(session_video, video)
            if reference_actions is not None:
                row["session_vs_reference_max_abs_diff"] = max_abs_diff(session_action, reference_actions["sampled_normalized"])
            row["session_seconds"] = time.time() - t1
        rows.append(row)
        log(
            f"sample {position:03d} [{source}] mse={mse:.6f} ref={reference_mse:.6f} "
            f"max|d|={row.get('reference_max_abs_diff', float('nan')):.3e} ({time.time() - t0:.1f}s)"
        )
    return rows


# --------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--text-encoder-path", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--vae-path", default=DEFAULT_VAE)
    parser.add_argument("--normalization-path", default=str(DEFAULT_NORMALIZATION))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("latent", "pixels", "both"), default="latent")
    parser.add_argument("--noise", choices=("stored", "regenerate"), default="stored")
    parser.add_argument("--text", choices=("stored", "encode"), default="stored")
    parser.add_argument("--samples", default=None, help="comma-separated sample positions (default all)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--session-check", action="store_true", help="also run PolicyInferenceSession.predict_from_latent (needs --text encode)")
    args = parser.parse_args()
    if args.session_check and args.text != "encode":
        parser.error("--session-check needs --text encode (predict_from_latent encodes the prompt rows itself)")
    return args


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log(f"torch {torch.__version__} device={device} mode={args.mode} noise={args.noise} text={args.text}")
    if args.noise == "regenerate" and device.type != "cuda":
        log("WARNING: regenerated noise on a non-CUDA generator cannot match the CUDA run")

    bundle_paths = list_bundles(args.bundle_dir, args.samples)
    if not bundle_paths:
        log(f"no bundles under {args.bundle_dir}")
        return 2
    components = build_components(args, device)

    rows: list[dict] = []
    errors: list[dict] = []
    for path in bundle_paths:
        try:
            rows.extend(replay_sample(args, components, device, path))
        except Exception:  # noqa: BLE001 - report every sample, fail at the end
            errors.append({"bundle": str(path), "traceback": traceback.format_exc()})
            log(f"FAILED {path.name}:\n{traceback.format_exc()}")

    reference_values = sorted({r["sample_position"]: r["reference_action_mse"] for r in rows}.items())
    report = {
        "format": "xpolicylab_sana_wam_holdout_replay_v1",
        "args": {k: str(v) for k, v in vars(args).items()},
        "device": str(device),
        "torch": torch.__version__,
        "sampling": components["sampling"],
        "load_report": {k: (v if not isinstance(v, list) else v[:16]) for k, v in components["load_report"].items()},
        "normalization_sha256": getattr(components["normalization"], "sha256", None),
        "per_source": {},
        "reference": {
            "manifest_action_mse": summarize([v for _, v in reference_values]),
            "documented_s35000": REFERENCE_AGGREGATES_S35000,
        },
        "rows": rows,
        "errors": errors,
    }
    for source in sorted({r["source"] for r in rows}):
        source_rows = [r for r in rows if r["source"] == source]
        block = {
            "action_mse": summarize([r["action_mse"] for r in source_rows]),
            "action_mse_delta_vs_manifest": summarize([r["action_mse_delta_vs_manifest"] for r in source_rows]),
        }
        if any("reference_max_abs_diff" in r for r in source_rows):
            block["reference_max_abs_diff"] = summarize([r["reference_max_abs_diff"] for r in source_rows if "reference_max_abs_diff" in r])
            block["reference_bitwise_count"] = int(sum(bool(r.get("reference_bitwise")) for r in source_rows))
        if any("latent_max_abs_diff" in r for r in source_rows):
            block["latent_max_abs_diff"] = summarize([r["latent_max_abs_diff"] for r in source_rows])
        if any("text_embed_max_abs_diff" in r for r in source_rows):
            block["text_embed_max_abs_diff"] = summarize([r["text_embed_max_abs_diff"] for r in source_rows])
        if any("video_noise_max_abs_diff" in r for r in source_rows):
            block["video_noise_max_abs_diff"] = summarize([r["video_noise_max_abs_diff"] for r in source_rows])
            block["action_noise_max_abs_diff"] = summarize([r["action_noise_max_abs_diff"] for r in source_rows])
        if any("session_vs_direct_max_abs_diff" in r for r in source_rows):
            block["session_action_mse"] = summarize([r["session_action_mse"] for r in source_rows])
            block["session_vs_direct_max_abs_diff"] = summarize([r["session_vs_direct_max_abs_diff"] for r in source_rows])
            block["session_vs_direct_bitwise_count"] = int(sum(bool(r.get("session_vs_direct_bitwise")) for r in source_rows))
        report["per_source"][source] = block

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    log(f"wrote {args.out}")
    for source, block in report["per_source"].items():
        log(f"[{source}] action_mse {block['action_mse']} | reference {report['reference']['manifest_action_mse']}")
        if "reference_max_abs_diff" in block:
            log(f"[{source}] vs reference actions.pt: {block['reference_max_abs_diff']} bitwise={block['reference_bitwise_count']}")
        if "session_vs_direct_max_abs_diff" in block:
            log(f"[{source}] session.predict_from_latent vs direct: {block['session_vs_direct_max_abs_diff']} bitwise={block['session_vs_direct_bitwise_count']}")
    if errors:
        log(f"{len(errors)} sample(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
