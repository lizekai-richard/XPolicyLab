"""The OpenWAM L-shape canvas: the MoT policy's single composited visual stream.

Port of Sana ``dev/rwm/diffusion/data/openwam_multiview_layout.py`` (``assemble_openwam_canvas`` / ``_stretch_resize``)
and the canvas dataset's clip transform (``ToTensorVideo -> Normalize(0.5, 0.5)``, no resize-crop). The complete RGB
frame is 384 x 320 (height x width): ``cam_head`` stretched to the 256 x 320 top slot, ``cam_left_wrist`` /
``cam_right_wrist`` to the 128 x 160 bottom-left / bottom-right slots, every slot a PIL BILINEAR resize with no
aspect-ratio preservation. Frames arrive as decoded RGB uint8 arrays (the policy server decodes; the training-time
OpenCV JPEG-decode convention concerned the raw HDF5 JPEGs only) and are never channel-swapped.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from sana_wam_min.pixels import ToTensorVideo

OPENWAM_CAMERA_LAYOUT: tuple[str, ...] = ("cam_head", "cam_left_wrist", "cam_right_wrist")
OPENWAM_CANVAS_HEIGHT = 384
OPENWAM_CANVAS_WIDTH = 320
OPENWAM_TOP_HEIGHT_RATIO = 2.0 / 3.0
OPENWAM_LAYOUT_ID = "openwam_lshape_rgb_v1"
COMPOSITE_VIEW_KEY = "openwam_canvas"


def slot_geometry(
    out_h: int = OPENWAM_CANVAS_HEIGHT,
    out_w: int = OPENWAM_CANVAS_WIDTH,
    top_height_ratio: float = OPENWAM_TOP_HEIGHT_RATIO,
    camera_layout: Sequence[str] = OPENWAM_CAMERA_LAYOUT,
) -> dict[str, tuple[int, int, int, int]]:
    """Return ``{camera: (y, x, height, width)}`` of the three slots (top, bottom-left, bottom-right)."""

    top_h = int(round(out_h * top_height_ratio))
    bottom_h = out_h - top_h
    half_w = out_w // 2
    right_w = out_w - half_w
    top, left, right = camera_layout
    return {top: (0, 0, top_h, out_w), left: (top_h, 0, bottom_h, half_w), right: (top_h, half_w, bottom_h, right_w)}


def _to_pil(frame, camera: str) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB") if frame.mode != "RGB" else frame
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"{camera}: canvas slots take uint8 HxWx3 RGB frames, got shape {array.shape} dtype {array.dtype}")
    return Image.fromarray(np.ascontiguousarray(array))


def _stretch_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """BILINEAR resize to the slot size with no aspect-ratio preservation (``_stretch_resize`` of the training layout)."""

    return image.resize((target_width, target_height), Image.BILINEAR)


def assemble_openwam_canvas(
    frames_by_camera: Mapping[str, object],
    camera_layout: Sequence[str] = OPENWAM_CAMERA_LAYOUT,
    out_h: int = OPENWAM_CANVAS_HEIGHT,
    out_w: int = OPENWAM_CANVAS_WIDTH,
    top_height_ratio: float = OPENWAM_TOP_HEIGHT_RATIO,
) -> Image.Image:
    """Composite three camera frames into the OpenWAM L-shape canvas (PIL RGB image of size ``(out_w, out_h)``)."""

    if len(camera_layout) != 3:
        raise ValueError(f"the OpenWAM canvas layout takes exactly 3 cameras, got {len(camera_layout)}")
    missing = [camera for camera in camera_layout if camera not in frames_by_camera]
    if missing:
        raise KeyError(f"missing required camera frame(s) for the OpenWAM canvas: {missing}")

    top_h = int(round(out_h * top_height_ratio))
    bottom_h = out_h - top_h
    half_w = out_w // 2
    right_w = out_w - half_w

    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    canvas.paste(_stretch_resize(_to_pil(frames_by_camera[camera_layout[0]], camera_layout[0]), top_h, out_w), (0, 0))
    canvas.paste(_stretch_resize(_to_pil(frames_by_camera[camera_layout[1]], camera_layout[1]), bottom_h, half_w), (0, top_h))
    canvas.paste(_stretch_resize(_to_pil(frames_by_camera[camera_layout[2]], camera_layout[2]), bottom_h, right_w), (half_w, top_h))
    return canvas


def canvas_from_frames(frames_rgb: Sequence[np.ndarray], camera_layout: Sequence[str] = OPENWAM_CAMERA_LAYOUT) -> np.ndarray:
    """Composite the frames given in ``camera_layout`` order into one uint8 ``[384, 320, 3]`` RGB canvas."""

    if len(frames_rgb) != len(camera_layout):
        raise ValueError(f"expected {len(camera_layout)} frames in order {tuple(camera_layout)}, got {len(frames_rgb)}")
    canvas = assemble_openwam_canvas(dict(zip(camera_layout, frames_rgb)), camera_layout)
    return np.asarray(canvas, dtype=np.uint8)


def canvas_transform() -> T.Compose:
    """The canvas dataset's clip transform: ``ToTensorVideo -> Normalize(0.5, 0.5)`` (no resize-crop; the canvas is the bucket)."""

    return T.Compose([ToTensorVideo(), T.Normalize([0.5] * 3, [0.5] * 3, inplace=True)])


def canvas_to_model_tensor(canvas_uint8_hwc: np.ndarray) -> torch.Tensor:
    """Map one uint8 ``[H, W, 3]`` canvas to float32 ``[3, H, W]`` in ``[-1, 1]`` on CPU, as the training dataset does."""

    frame = torch.as_tensor(np.asarray(canvas_uint8_hwc))
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != torch.uint8:
        raise ValueError(f"canvas must be uint8 [H, W, 3], got {tuple(frame.shape)} {frame.dtype}")
    chw = frame.detach().to(device="cpu").permute(2, 0, 1).contiguous()
    return canvas_transform()(chw.unsqueeze(0))[0]


def canvas_latent_hw(spatial_compression: int = 32, out_h: int = OPENWAM_CANVAS_HEIGHT, out_w: int = OPENWAM_CANVAS_WIDTH) -> tuple[int, int]:
    """The native latent grid of the canvas (12 x 10 for the LTX-2.3 VAE's 32x spatial compression)."""

    return out_h // int(spatial_compression), out_w // int(spatial_compression)


__all__ = [
    "COMPOSITE_VIEW_KEY",
    "OPENWAM_CAMERA_LAYOUT",
    "OPENWAM_CANVAS_HEIGHT",
    "OPENWAM_CANVAS_WIDTH",
    "OPENWAM_LAYOUT_ID",
    "OPENWAM_TOP_HEIGHT_RATIO",
    "assemble_openwam_canvas",
    "canvas_from_frames",
    "canvas_latent_hw",
    "canvas_to_model_tensor",
    "canvas_transform",
    "slot_geometry",
]
