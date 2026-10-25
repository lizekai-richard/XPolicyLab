#!/usr/bin/env python3
"""Decode the ``prediction_latent_dir`` dumps of one RoboDojo unit attempt of SANA_WAM_CAUSAL into side-by-side MP4s.

Every decision of the adapter (``prediction_latent_dir`` set) leaves ``ep<E>_env00_chunk<C>.npz``: the chunk's generated
target latents after its observation latent (its start frame encoded alone), still packed and normalized, plus the
control tick the chunk starts at; the next decision adds ``ep<E>_env00_chunk<C>_observed.npz``, the same with the
chunk's streamed (real) target latents -- the latents the memory commits and ``world_model_rel_rmse`` compares against.
This tool decodes them with the LTX-2.3 VAE IN THE EPISODE'S STREAM: the decoder needs the temporal context the encoder
had (a chunk's latent was encoded after the whole episode before it), so chunk c is decoded as the last latent of [sink,
observed chunks 0 .. c-1, X_c] with X_c its observed latents (the VAE's reconstruction) or its generated ones (the
prediction) -- the same context for both. A chunk without the observed chunks before it (the last chunk's successor, a
padded window's frozen chunk, memory_mode none) falls back to [its observation latent, X_c]. The frames are paired with
the evaluator's own rollout videos (``client/*/upstream_eval_result/episode_<k>_cam_<view>_<success|fail>.mp4``):

    one output frame every ``video_frame_stride`` ticks (6.25 fps for f33fps8); row = camera (head, left wrist, right
    wrist); columns = the real frame at that tick | the VAE's reconstruction of the observed chunk (the ceiling a
    perfect prediction would reach) | the world model's prediction. Chunk c, generated at tick t_c, covers the ticks
    t_c + s .. t_c + C. The frame title carries the chunk's latent relative RMSE (prediction vs observed latent, the
    adapter's world_model_rel_rmse) and the per-frame PSNR of the prediction and of the reconstruction against the real
    frame (head camera).

Usage (GPU node, sana-wam env):
    python tools/decode_prediction_latents.py --attempt-dir <RUN_ROOT>/units/<uid>/a1 --out <dir> \
        [--vae-path $SANA_WAM_VAE_PATH] [--device cuda] [--max-episodes N]
Writes ``<out>/<task>_layout<L>_<success|fail>.mp4`` and ``<out>/<task>_layout<L>_<success|fail>_stats.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from sana_wam_causal import shared  # noqa: E402,F401  (sys.path bridge to policy/SANA_WAM)
from sana_wam_min.multiview import unpack_multiview_latents  # noqa: E402
from sana_wam_min.robodojo_io import ROBODOJO_VIEW_ORDER  # noqa: E402
from sana_wam_min.vae import decode_latent, load_vae, video_to_uint8  # noqa: E402

PANEL_H, PANEL_W = 256, 320          # every panel is shown at the strip's per-view size
HEADER = 22


def load_chunks(pred_dir: Path) -> dict[int, list[dict]]:
    """{episode: [chunk, ...]} with every chunk's predicted dump and, when present, its observed dump."""

    episodes: dict[int, dict[int, dict]] = defaultdict(dict)
    for path in sorted(pred_dir.glob("ep*_env*_chunk*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as d:
            data = {k: d[k] for k in d.files}
        slot = episodes[int(data["episode"])].setdefault(int(data["chunk"]), {})
        slot["observed" if bool(data.get("observed", False)) else "predicted"] = data
    out = {}
    for ep, chunks in sorted(episodes.items()):
        out[ep] = [chunks[c] for c in sorted(chunks) if "predicted" in chunks[c]]
    return out


def client_episodes(attempt_dir: Path) -> tuple[str, list[dict], Path | None]:
    """(task, per-episode details in client order, the upstream_eval_result dir)."""

    results = sorted(glob.glob(str(attempt_dir / "client" / "*" / "upstream_eval_result" / "_result.json")))
    if not results:
        return "unknown", [], None
    res_path = Path(results[0])
    task = re.sub(r"^\d+_", "", res_path.parent.parent.name)
    details = json.loads(res_path.read_text()).get("details", {})
    ordered = [details[k] for k in sorted(details, key=int)]
    return task, ordered, res_path.parent


def real_frames(video_dir: Path, k: int, view: str, ticks: set[int]) -> dict[int, np.ndarray]:
    import imageio.v3 as iio
    from PIL import Image

    hits = sorted(glob.glob(str(video_dir / f"episode_{k:07d}_{view}_*.mp4")))
    if not hits:
        return {}
    out = {}
    for i, frame in enumerate(iio.imiter(hits[0], plugin="pyav")):
        if i in ticks:
            out[i] = np.asarray(Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.BILINEAR))
    return out


@torch.no_grad()
def decode(bundle, latent: np.ndarray, shapes, device: str, frames: int) -> list[np.ndarray]:
    """Per view the LAST ``frames`` decoded frames, uint8 [frames, H, W, 3], of the packed latent run [1, C, F, h, W]
    (the decoder yields 1 + (F - 1) * 8 frames: the first latent's frame, then 8 per later latent)."""

    latent = torch.from_numpy(latent).to(device=device, dtype=torch.bfloat16)
    views = unpack_multiview_latents(latent, shapes) if len(shapes) > 1 else (latent,)
    clips = []
    for view in views:
        pixels = decode_latent(bundle, view.contiguous())[0]                 # [3, T, H, W] in [-1, 1]
        clip = video_to_uint8(pixels.permute(1, 2, 3, 0).float()).numpy()   # [T, H, W, 3]
        clips.append(clip[-frames:])
    return clips


def stream_context(chunks: list[dict], c: int) -> np.ndarray | None:
    """[sink, observed targets of chunks 0 .. c-1] (packed, fp16) when every one of them was dumped, else None."""

    sink = chunks[0]["predicted"]["video_latent"][:, :, :1]      # chunk 0's observation latent: the sink
    parts = [sink]
    for k in range(c):
        if "observed" not in chunks[k]:
            return None
        parts.append(chunks[k]["observed"]["video_latent"][:, :, 1:])
    return np.concatenate(parts, axis=2)


def panel(img: np.ndarray | None) -> np.ndarray:
    from PIL import Image

    if img is None:
        return np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    if img.shape[:2] != (PANEL_H, PANEL_W):
        img = np.asarray(Image.fromarray(img).resize((PANEL_W, PANEL_H), Image.BILINEAR))
    return img


def psnr(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    mse = float(((panel(a).astype(np.float64) - panel(b).astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def compose(rows: list[tuple], title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = np.zeros((HEADER + PANEL_H * len(rows), 3 * PANEL_W, 3), dtype=np.uint8)
    for r, cells in enumerate(rows):
        y = HEADER + r * PANEL_H
        for col, img in enumerate(cells):
            canvas[y:y + PANEL_H, col * PANEL_W:(col + 1) * PANEL_W] = panel(img)
    img = Image.fromarray(canvas)
    ImageDraw.Draw(img).text((6, 4), title, fill=(255, 255, 255))
    return np.asarray(img)


def relative_rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate, reference = estimate.astype(np.float32), reference.astype(np.float32)
    return float(np.sqrt(((estimate - reference) ** 2).mean()) / max(np.sqrt((reference ** 2).mean()), 1e-8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae-path", default=os.environ.get("SANA_WAM_VAE_PATH") or os.environ.get("SANA_POLICY_VAE_PRETRAINED"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-episodes", type=int, default=0)
    a = ap.parse_args()

    attempt = Path(a.attempt_dir)
    episodes = load_chunks(attempt / "pred_latents")
    if not episodes:
        sys.exit(f"no prediction latents under {attempt / 'pred_latents'}")
    task, details, video_dir = client_episodes(attempt)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_vae(a.vae_path, device=a.device, dtype=torch.bfloat16)
    done = 0
    for k, (ep, chunks) in enumerate(episodes.items()):
        if a.max_episodes and done >= a.max_episodes:
            break
        info = details[k] if k < len(details) else {}
        layout = info.get("layout_id", f"ep{ep}")
        outcome = "success" if info.get("success") else "fail"
        first = chunks[0]["predicted"]
        stride = int(first["video_frame_stride"]) or 1
        strip = str(first["visual_layout"]) == "three_view_strip"
        plan = []                                          # (tick, chunk index, decoded frame index)
        for ci, ch in enumerate(chunks):
            t0, ex = int(ch["predicted"]["tick"]), int(ch["predicted"]["executed"])
            for j in range(1, ex // stride + 1):
                plan.append((t0 + j * stride, ci, j))
        ticks = {t for t, _, _ in plan}
        views = ROBODOJO_VIEW_ORDER if strip else ROBODOJO_VIEW_ORDER[:1]
        reals = {view: real_frames(video_dir, k, view, ticks) for view in views} if video_dir else {}
        shapes = [tuple(int(v) for v in s) for s in np.asarray(first["view_latent_shapes"]).reshape(-1, 2)]
        f_targets = int(first["video_latent"].shape[2]) - 1
        n_frames = 8 * f_targets
        predicted, observed, contexts = [], [], []
        for c, ch in enumerate(chunks):
            context = stream_context(chunks, c)
            if context is None or int(ch["predicted"]["chunk"]) != c:
                context = ch["predicted"]["video_latent"][:, :, :1]   # fall back: the chunk's observation latent alone
            contexts.append(int(context.shape[2]))
            run = np.concatenate([context, ch["predicted"]["video_latent"][:, :, 1:]], axis=2)
            predicted.append(decode(bundle, run, shapes, a.device, n_frames))
            if "observed" in ch:
                run = np.concatenate([context, ch["observed"]["video_latent"][:, :, 1:]], axis=2)
                observed.append(decode(bundle, run, shapes, a.device, n_frames))
            else:
                observed.append(None)
        latent_err = [
            relative_rmse(ch["predicted"]["video_latent"][:, :, 1:], ch["observed"]["video_latent"][:, :, 1:])
            if "observed" in ch else float("nan") for ch in chunks
        ]
        frames, stats = [], []
        for t, ci, j in plan:
            rows = []
            for v, view in enumerate(views):
                real = reals.get(view, {}).get(t)
                rec = None if observed[ci] is None else observed[ci][v][j - 1]
                rows.append((real, rec, predicted[ci][v][j - 1]))
            p_pred, p_rec = psnr(rows[0][0], rows[0][2]), psnr(rows[0][0], rows[0][1])
            stats.append({"tick": t, "chunk": int(chunks[ci]["predicted"]["chunk"]), "latent_rel_rmse": latent_err[ci],
                          "psnr_pred_vs_real_head": p_pred, "psnr_recon_vs_real_head": p_rec,
                          "decode_context_latents": contexts[ci]})
            title = (f"{task} layout {layout} {outcome} | tick {t} chunk {int(chunks[ci]['predicted']['chunk'])} | "
                     f"real | VAE recon of real | predicted | latent relRMSE {latent_err[ci]:.3f} | "
                     f"PSNR head pred {p_pred:.1f} dB, recon {p_rec:.1f} dB")
            frames.append(compose(rows, title))
        if not frames:
            continue
        dst = out_dir / f"{task}_layout{layout}_{outcome}.mp4"
        import imageio.v2 as iio2                      # the ffmpeg writer (the pyav plugin cannot size its codec here)
        with iio2.get_writer(dst, fps=25.0 / stride, codec="libx264", quality=7, macro_block_size=2) as writer:
            for frame in frames:
                writer.append_data(frame)
        (out_dir / f"{task}_layout{layout}_{outcome}_stats.json").write_text(json.dumps(stats, indent=1))
        print(f"wrote {dst} ({len(frames)} frames, {len(chunks)} chunks)", flush=True)
        done += 1


if __name__ == "__main__":
    main()
