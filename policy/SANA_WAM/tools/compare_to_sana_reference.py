#!/usr/bin/env python
"""Run the sana_wam_min session on Sana reference inputs and gate each stage against the Sana values.

Sana-FREE companion of ``sana_parity_reference.py``: loads each ``sana_reference_*.pt``, builds
``PolicyInferenceSession.from_paths`` once, feeds the identical frames / raw state / instruction /
diffusion seed, and compares every stage against the Sana values.

Gate semantics:

* bitwise gates (must be exactly equal for PASS): prompt rows and wire string, Gemma text embeds
  and mask, the packed observation strip (LTX2 causal VAE), regenerated video/action noise,
  the action mask and the view latent shapes;
* ``kernel_level`` comparison: the final normalized ``action80`` after the Euler loop. The
  transformer forward is NOT bitwise across attention kernels (Sana runs flash/fla kernels, this
  package runs SDPA), so the sampled action carries kernel-level drift. It is reported with
  max|d| and FAILs only above ``--action-tol`` (default 5e-2 normalized units); the raw absolute
  action max|d| is reported for information only. Functional parity of the transformer is
  established by holdout MSE parity (``tools/holdout_replay.py``), not by this comparison.

Overall PASS when every gate of every reference passes; exit code 0 on PASS, 1 on FAIL or error.

Usage (1 GPU, sana-wam env, Sana NOT on PYTHONPATH):
    PYTHONPATH=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/XPolicyLab \\
    python compare_to_sana_reference.py --out .../logs/compare_to_sana_reference.json
"""

from __future__ import annotations

import argparse
import inspect
import json
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
DEFAULT_REFERENCE_DIR = PORT_ROOT / "logs"
DEFAULT_CHECKPOINT_DIR = PORT_ROOT / "ckpt_snapshots" / "epoch_6_step_35000"
DEFAULT_TRAIN_CONFIG = PORT_ROOT / "ckpt_snapshots" / "config.yaml"
DEFAULT_NORMALIZATION = PORT_ROOT / "ckpt_snapshots" / "normalization" / "robodojo_arx_x5_model_fps_25_f25_normalization.json"
DEFAULT_TEXT_ENCODER = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/models/Sana/text_encoder/gemma-2-2b-it"
DEFAULT_VAE = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/models/Sana/LTX-2.3-Diffusers"
REFERENCE_FORMAT = "xpolicylab_sana_wam_sana_reference_v1"
IMAGE_SIZE = 320


def log(message: str) -> None:
    print(f"[compare_to_sana_reference] {time.strftime('%H:%M:%S')} {message}", flush=True)


# --------------------------------------------------------------------------- pure helpers


def bitwise_gate(ours: torch.Tensor, reference: torch.Tensor) -> dict:
    """Bitwise gate record for one tensor pair: PASS only when shape, dtype-cast values and every element agree."""

    a = ours.detach().cpu()
    b = reference.detach().cpu()
    if a.shape != b.shape:
        return {"gate": "bitwise", "verdict": "FAIL", "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    equal = bool(torch.equal(a, b))
    max_abs = float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0
    return {"gate": "bitwise", "verdict": "PASS" if equal else "FAIL", "bitwise": equal, "max_abs_diff": max_abs}


def kernel_level_compare(ours: torch.Tensor, reference: torch.Tensor, tol: float | None) -> dict:
    """Kernel-level record: max|d| always reported; gated against ``tol`` only when ``tol`` is given."""

    a = ours.detach().float().cpu()
    b = reference.detach().float().cpu()
    if a.shape != b.shape:
        return {"gate": "kernel_level", "verdict": "FAIL", "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    max_abs = float((a - b).abs().max().item()) if a.numel() else 0.0
    record = {
        "gate": "kernel_level",
        "bitwise": bool(torch.equal(ours.detach().cpu(), reference.detach().cpu())),
        "max_abs_diff": max_abs,
        "ref_max_abs": float(b.abs().max().item()) if b.numel() else 0.0,
    }
    if tol is not None:
        record["tol"] = float(tol)
        record["verdict"] = "PASS" if max_abs <= tol else "FAIL"
    else:
        record["verdict"] = "INFO"
    return record


def bool_gate(equal: bool) -> dict:
    return {"gate": "bitwise", "verdict": "PASS" if equal else "FAIL"}


def all_pass(records: list[dict]) -> str:
    """PASS when every gated record passes; INFO records do not count."""

    return "PASS" if all(r.get("verdict") in ("PASS", "INFO") for r in records) else "FAIL"


def frames_chw_to_hwc(frames_chw: list[torch.Tensor]) -> list[np.ndarray]:
    return [np.ascontiguousarray(f.permute(1, 2, 0).numpy()) for f in frames_chw]


# --------------------------------------------------------------------------- session


def build_session(args, device: torch.device):
    from sana_wam_min.session import PolicyInferenceSession

    kwargs = {
        "normalization_path": str(args.normalization_path),
        "device": str(device),
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "flow_shift": args.flow_shift,
    }
    accepted = inspect.signature(PolicyInferenceSession.from_paths).parameters
    if "train_config_path" in accepted:
        kwargs["train_config_path"] = str(args.train_config)
    elif "config_path" in accepted:
        kwargs["config_path"] = str(args.train_config)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return PolicyInferenceSession.from_paths(str(args.checkpoint_dir), str(args.text_encoder_path), str(args.vae_path), **kwargs)


def intermediate_checks(session, reference: dict, frames_hwc: list[np.ndarray], device: torch.device) -> dict:
    """Bitwise gates on the prompt rows, text embeds/mask, observation strip and regenerated noise."""

    from sana_wam_min.text import render_token_group_rows

    out: dict = {}
    rows = list(render_token_group_rows(reference["instruction"]))
    out["prompt_rows"] = bool_gate(rows == list(reference["prompt_rows"]))
    out["instruction_wire"] = bool_gate("\n\n".join(rows) == reference["instruction_wire"])

    with torch.inference_mode():
        y, mask, _, _ = session.encode_rows(rows, None)
    out["text_embeds"] = bitwise_gate(y, reference["caption_embeds"])
    out["text_mask"] = bool_gate(bool(torch.equal(mask.cpu().long(), reference["caption_mask"].long())))

    latent_frames = (int(reference["num_frames"]) - 1) // 8 + 1
    with torch.inference_mode():
        strip, shapes = session.encode_observation(frames_hwc, latent_frames)
    out["strip"] = bitwise_gate(strip, reference["clean_strip"])
    out["view_latent_shapes"] = bool_gate([list(map(int, s)) for s in shapes] == list(reference["view_latent_shapes"]))

    if device.type == "cuda":
        generator = torch.Generator(device=device).manual_seed(int(reference["seed"]))
        video_noise = torch.randn(tuple(reference["clean_strip"].shape), device=device, dtype=reference["clean_strip"].dtype, generator=generator)
        action_noise = torch.randn((1, int(reference["k_actions"]), 80), device=device, dtype=torch.float32, generator=generator)
        out["video_noise"] = bitwise_gate(video_noise, reference["video_noise"])
        out["action_noise"] = bitwise_gate(action_noise, reference["action_noise"])
    return out


def compare_one(session, path: Path, device: torch.device, action_tol: float) -> dict:
    reference = torch.load(path, map_location="cpu", weights_only=False)
    if reference.get("format") != REFERENCE_FORMAT:
        raise ValueError(f"{path} has format {reference.get('format')!r}, expected {REFERENCE_FORMAT}")
    frames_hwc = frames_chw_to_hwc(reference["frames_uint8_chw"])
    row = {"name": reference["name"], "kind": reference["kind"], "file": str(path), "bundle_meta": reference.get("bundle_meta")}
    row["intermediate"] = intermediate_checks(session, reference, frames_hwc, device)

    generator = torch.Generator(device=device).manual_seed(int(reference["seed"]))
    t0 = time.time()
    result = session.predict(
        frames_hwc,
        reference["state80_raw"].numpy(),
        reference["state_mask80"].numpy(),
        reference["instruction"],
        generator,
    )
    row["predict_seconds"] = time.time() - t0
    action_model = torch.as_tensor(result.action80_model).reshape(-1, 80)
    action_abs = torch.as_tensor(result.action80_raw_absolute).reshape(-1, 80)
    row["action80_model"] = kernel_level_compare(action_model, reference["action80_model"].reshape(-1, 80), action_tol)
    row["action80_raw_absolute"] = kernel_level_compare(action_abs, reference["action80_raw_absolute"].reshape(-1, 80), None)
    our_mask = torch.as_tensor(result.action_mask).reshape(-1, 80).bool()
    row["action_mask"] = bool_gate(bool(torch.equal(our_mask[0].cpu(), reference["action_mask80"].bool())))
    video_latent = getattr(result, "video_latent", None)
    if video_latent is not None:
        row["video_latent"] = kernel_level_compare(torch.as_tensor(video_latent), reference["video_latent"], None)
    receipt = getattr(result, "receipt", None)
    if isinstance(receipt, dict):
        row["receipt"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in receipt.items()}
    gated = list(row["intermediate"].values()) + [row["action80_model"], row["action_mask"]]
    row["verdict"] = all_pass(gated)
    bitwise_failed = [k for k, v in row["intermediate"].items() if v.get("verdict") == "FAIL"]
    log(
        f"{reference['name']}: {row['verdict']} | action80_model kernel_level max|d|="
        f"{row['action80_model'].get('max_abs_diff', float('nan')):.3e} (tol {action_tol:g}) "
        f"| absolute max|d|={row['action80_raw_absolute'].get('max_abs_diff', float('nan')):.3e} "
        f"| bitwise gates failed: {bitwise_failed or 'none'}"
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR, help="dir holding sana_reference_*.pt")
    parser.add_argument("--reference", nargs="*", default=None, help="explicit reference .pt files (overrides --reference-dir)")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--text-encoder-path", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--vae-path", default=DEFAULT_VAE)
    parser.add_argument("--normalization-path", default=str(DEFAULT_NORMALIZATION))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument(
        "--action-tol", type=float, default=5e-2,
        help="max|d| ceiling on the final normalized action80 (kernel-level drift across attention kernels)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    paths = [Path(p) for p in args.reference] if args.reference else sorted(args.reference_dir.glob("sana_reference_*.pt"))
    if not paths:
        log("no sana_reference_*.pt found")
        return 2
    log(f"torch {torch.__version__} device={device} references={len(paths)}")
    t0 = time.time()
    session = build_session(args, device)
    log(f"session ready in {time.time() - t0:.1f}s")

    rows, errors = [], []
    for path in paths:
        try:
            rows.append(compare_one(session, path, device, args.action_tol))
        except Exception:  # noqa: BLE001
            errors.append({"file": str(path), "traceback": traceback.format_exc()})
            log(f"FAILED {path.name}:\n{traceback.format_exc()}")

    overall = "PASS" if rows and not errors and all(r["verdict"] == "PASS" for r in rows) else "FAIL"
    report = {
        "format": "xpolicylab_sana_wam_compare_to_sana_reference_v2",
        "args": {k: str(v) for k, v in vars(args).items()},
        "torch": torch.__version__,
        "device": str(device),
        "action_tol": args.action_tol,
        "overall": overall,
        "rows": rows,
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    log(f"wrote {args.out}; overall {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
