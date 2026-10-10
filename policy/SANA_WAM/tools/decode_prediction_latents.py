#!/usr/bin/env python3
"""Decode the ``prediction_latent_dir`` dumps of one RoboDojo unit attempt into side-by-side MP4s.

Each inference of the adapter (``prediction_latent_dir`` set) leaves ``ep<E>_env<V>_chunk<C>.npz``: the sampled, still
packed and normalized video latent of that chunk plus the control tick it was predicted at and how many targets were
executed. This tool decodes every chunk with the LTX-2.3 VAE and pairs it with the evaluator's own rollout videos
(``client/*/upstream_eval_result/episode_<k>_cam_<view>_<success|fail>.mp4``, one frame per control tick):

    one output frame every ``video_frame_stride`` ticks (6.25 fps for f33fps8); row = camera (head, left wrist,
    right wrist), left = the real frame at that tick, right = the world model's prediction of it. Chunk c, predicted from
    the observation at tick t_c, supplies the ticks t_c + s .. t_c + executed (its decoded frames 1 .. executed / s;
    decoded frame 0 is only the reconstruction of the observation it was conditioned on).

Usage (GPU node, sana-wam env, run from policy/SANA_WAM):
    python tools/decode_prediction_latents.py --attempt-dir <RUN_ROOT>/units/<uid>/a1 --out <dir> \
        [--vae-path $SANA_POLICY_VAE_PRETRAINED] [--device cuda] [--max-episodes N]
Writes ``<out>/<task>_layout<L>_<success|fail>.mp4`` (task and layout from the client's ``_result.json``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from sana_wam_min.multiview import unpack_multiview_latents  # noqa: E402
from sana_wam_min.robodojo_io import ROBODOJO_VIEW_ORDER  # noqa: E402
from sana_wam_min.vae import decode_latent, load_vae, video_to_uint8  # noqa: E402

PANEL_H, PANEL_W = 256, 320          # every panel is shown at the strip's per-view size
HEADER = 22


def load_chunks(pred_dir: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(pred_dir.glob("ep*_env*_chunk*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as d:
            episodes[int(d["episode"])].append({k: d[k] for k in d.files})
    for chunks in episodes.values():
        chunks.sort(key=lambda c: int(c["chunk"]))
    return dict(sorted(episodes.items()))


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
def decode_chunk(bundle, chunk: dict, device: str) -> list[np.ndarray]:
    """Per view a uint8 [T, H, W, 3] clip (T = (F - 1) * temporal + 1)."""
    latent = torch.from_numpy(chunk["video_latent"]).to(device=device, dtype=torch.bfloat16)
    shapes = [tuple(int(v) for v in s) for s in np.asarray(chunk["view_latent_shapes"]).reshape(-1, 2)]
    views = unpack_multiview_latents(latent, shapes) if len(shapes) > 1 else (latent,)
    clips = []
    for view in views:
        pixels = decode_latent(bundle, view.contiguous())[0]                 # [3, T, H, W] in [-1, 1]
        clips.append(video_to_uint8(pixels.permute(1, 2, 3, 0).float()).numpy())   # [T, H, W, 3]
    return clips


def panel(img: np.ndarray) -> np.ndarray:
    from PIL import Image

    if img.shape[:2] != (PANEL_H, PANEL_W):
        img = np.asarray(Image.fromarray(img).resize((PANEL_W, PANEL_H), Image.BILINEAR))
    return img


def compose(rows: list[tuple[np.ndarray | None, np.ndarray]], title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = np.zeros((HEADER + PANEL_H * len(rows), 2 * PANEL_W, 3), dtype=np.uint8)
    for r, (real, pred) in enumerate(rows):
        y = HEADER + r * PANEL_H
        if real is not None:
            canvas[y:y + PANEL_H, :PANEL_W] = panel(real)
        canvas[y:y + PANEL_H, PANEL_W:] = panel(pred)
    img = Image.fromarray(canvas)
    ImageDraw.Draw(img).text((6, 4), title, fill=(255, 255, 255))
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae-path", default=os.environ.get("SANA_POLICY_VAE_PRETRAINED") or os.environ.get("SANA_WAM_VAE_PATH"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-episodes", type=int, default=0)
    a = ap.parse_args()
    import imageio.v3 as iio

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
        stride = int(chunks[0]["video_frame_stride"]) or 1
        layout_name = str(chunks[0]["visual_layout"])
        plan = []                                          # (tick, chunk index, decoded frame index)
        for ci, ch in enumerate(chunks):
            t0, ex = int(ch["tick"]), int(ch["executed"])
            for j in range(1, ex // stride + 1):
                plan.append((t0 + j * stride, ci, j))
        ticks = {t for t, _, _ in plan}
        reals = {view: real_frames(video_dir, k, view, ticks) for view in ROBODOJO_VIEW_ORDER} if video_dir else {}
        decoded = [decode_chunk(bundle, ch, a.device) for ch in chunks]
        frames = []
        for t, ci, j in plan:
            clips = decoded[ci]
            if layout_name == "three_view_strip":
                rows = [(reals.get(view, {}).get(t), clips[v][j]) for v, view in enumerate(ROBODOJO_VIEW_ORDER)]
            else:                                          # one canvas: show it once, with the head camera as reference
                rows = [(reals.get(ROBODOJO_VIEW_ORDER[0], {}).get(t), clips[0][j])]
            title = f"{task} layout {layout} {outcome} | tick {t} chunk {int(chunks[ci]['chunk'])} | left real, right predicted"
            frames.append(compose(rows, title))
        if not frames:
            continue
        dst = out_dir / f"{task}_layout{layout}_{outcome}.mp4"
        import imageio.v2 as iio2                      # the ffmpeg writer (the pyav plugin cannot size its codec here)
        with iio2.get_writer(dst, fps=25.0 / stride, codec="libx264", quality=7, macro_block_size=2) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"wrote {dst} ({len(frames)} frames, {len(chunks)} chunks)", flush=True)
        done += 1


if __name__ == "__main__":
    main()
