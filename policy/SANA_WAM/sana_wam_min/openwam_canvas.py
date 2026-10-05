"""OpenWAM composite-canvas front-end of the SANA policy (the ``rwm/openwam`` canvas line).

Ports of Sana ``dev/rwm/diffusion/data/openwam_multiview_layout.py`` (``assemble_openwam_canvas``, itself
OpenWAM's ``assemble_multiview_layout``: a PIL BILINEAR stretch of every camera straight into its slot, no
aspect-preserving crop, no gaps) and of the canvas dataset's pixel transform and one-row prompt
(``datasets/robodojo_openwam_canvas_sft_data.py``: ``ToTensorVideo -> Normalize(0.5, 0.5)`` over the whole
canvas, ``_rebuild_openwam_prompt``). The three RoboDojo cameras become ONE 384 x 320 RGB frame -- head
256 x 320 on top, left / right wrist 128 x 160 below -- that the VAE encodes once into a ``[128, F, 12, 10]``
latent; the policy sees 480 video tokens on the native 12 x 10 grid and one prompt shared by the video and
action tokens. Live frames arrive as decoded RGB uint8 arrays and receive no colour conversion (the training
reader's ``cv2`` decode of the raw HDF5 JPEGs only undoes a swap the recorder stored; see the Sana module).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .pixels import to_tensor
from .text import ROBODOJO_EMBODIMENT, prompt_field

# (top, bottom-left, bottom-right) slots; the same camera names the observation carries.
OPENWAM_CAMERA_LAYOUT: tuple[str, ...] = ("cam_head", "cam_left_wrist", "cam_right_wrist")
OPENWAM_CANVAS_HEIGHT = 384
OPENWAM_CANVAS_WIDTH = 320
OPENWAM_TOP_HEIGHT_RATIO = 2.0 / 3.0
# Model-facing metadata of the one composite stream (the dataset's ``_rewrite_openwam_data_info``).
OPENWAM_VIEW_KEY = "openwam_canvas"
OPENWAM_VIEW_SLOT_IDS = (0,)
# Observation View text of the composite-view row (the dataset's ``_OPENWAM_COMPOSITE_DESCRIPTOR``), by era:
#   composite_view -- rwm/openwam and zekai-merge up to 2a10d0d75 (the canvas class's G = 1 row, then the G = 2 view row);
#   l_shape        -- zekai-merge 2a10d0d75 .. 92b9e64d1 (2026-09-22/23, G = 1);
#   two_rows       -- zekai-merge 92b9e64d1 / rwm/mot 3ea1f9af9 on (G = 1; "L shape" was only OpenWAM's historical name for
#                     the filled head-over-wrists rectangle).
OPENWAM_COMPOSITE_VIEW_TEXT = "a composite view combining the head camera above the left and right wrist cameras"
OPENWAM_L_SHAPE_TEXT = (
    "one image compositing the robot's cameras in an L shape, with the head camera across the top two "
    "thirds and the left and right wrist cameras side by side below it"
)
OPENWAM_TWO_ROWS_TEXT = (
    "one image stacking the robot's cameras in two rows, with the head camera across the full width of "
    "the top two thirds and the left and right wrist cameras side by side in the bottom third"
)
OPENWAM_PROMPT_TEXTS = {
    "composite_view": OPENWAM_COMPOSITE_VIEW_TEXT,
    "l_shape": OPENWAM_L_SHAPE_TEXT,
    "two_rows": OPENWAM_TWO_ROWS_TEXT,
}


def canvas_slot_boxes(
    out_h: int = OPENWAM_CANVAS_HEIGHT,
    out_w: int = OPENWAM_CANVAS_WIDTH,
    top_height_ratio: float = OPENWAM_TOP_HEIGHT_RATIO,
    camera_layout: Sequence[str] = OPENWAM_CAMERA_LAYOUT,
) -> dict[str, tuple[int, int, int, int]]:
    """Pixel box ``(top, left, height, width)`` of every camera slot, with OpenWAM's rounding
    (top ``round(H * ratio)`` rows over the full width, the rest split ``W // 2`` / ``W - W // 2``)."""

    if len(camera_layout) != 3:
        raise ValueError(f"the OpenWAM canvas layout takes exactly 3 cameras, got {len(camera_layout)}")
    top_h = int(round(out_h * top_height_ratio))
    bottom_h = out_h - top_h
    half_w = out_w // 2
    right_w = out_w - half_w
    return {
        camera_layout[0]: (0, 0, top_h, out_w),
        camera_layout[1]: (top_h, 0, bottom_h, half_w),
        camera_layout[2]: (top_h, half_w, bottom_h, right_w),
    }


def _as_pil_rgb(frame: Any, camera: str) -> Image.Image:
    """A PIL RGB image of one camera frame: a uint8 HxWx3 array (no channel or value change) or an RGB image."""

    if isinstance(frame, Image.Image):
        if frame.mode != "RGB":
            raise ValueError(f"{camera}: PIL frames must be RGB, got mode {frame.mode!r}")
        return frame
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"{camera}: expected a uint8 HxWx3 RGB frame, got shape {array.shape} dtype {array.dtype}")
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def _stretch_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """BILINEAR resize to the slot size with no aspect-ratio preservation (OpenWAM ``_stretch_resize``)."""

    return image.resize((target_width, target_height), Image.BILINEAR)


def assemble_openwam_canvas(
    frames_by_camera: Mapping[str, Any],
    camera_layout: Sequence[str] = OPENWAM_CAMERA_LAYOUT,
    out_h: int = OPENWAM_CANVAS_HEIGHT,
    out_w: int = OPENWAM_CANVAS_WIDTH,
    top_height_ratio: float = OPENWAM_TOP_HEIGHT_RATIO,
) -> np.ndarray:
    """Composite three camera frames into the OpenWAM L-shape canvas, returned as uint8 ``[out_h, out_w, 3]`` RGB.

    Every camera named in ``camera_layout`` (top, bottom-left, bottom-right) is required: a missing one raises
    instead of becoming a black slot, as the training reader does. Pixel-identical to Sana's
    ``assemble_openwam_canvas`` / OpenWAM's ``assemble_multiview_layout`` for the same Pillow.
    """

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
    top, left, right = (_as_pil_rgb(frames_by_camera[camera], camera) for camera in camera_layout)
    canvas.paste(_stretch_resize(top, top_h, out_w), (0, 0))
    canvas.paste(_stretch_resize(left, bottom_h, half_w), (0, top_h))
    canvas.paste(_stretch_resize(right, bottom_h, right_w), (half_w, top_h))
    return np.asarray(canvas, dtype=np.uint8)


_CANVAS_NORMALIZE = T.Normalize([0.5] * 3, [0.5] * 3, inplace=True)


def canvas_to_model_tensor(canvas_uint8_hwc: Any, canvas_hw: tuple[int, int] = (OPENWAM_CANVAS_HEIGHT, OPENWAM_CANVAS_WIDTH)) -> torch.Tensor:
    """Map one uint8 ``[H, W, 3]`` canvas to float32 ``[3, H, W]`` in ``[-1, 1]``: ``ToTensorVideo -> Normalize``, no
    resize or crop (the canvas IS the training bucket), on CPU exactly like the dataset's ``_CANVAS_TRANSFORM``."""

    frame = canvas_uint8_hwc if torch.is_tensor(canvas_uint8_hwc) else torch.as_tensor(np.asarray(canvas_uint8_hwc))
    expected = (int(canvas_hw[0]), int(canvas_hw[1]), 3)
    if tuple(frame.shape) != expected or frame.dtype != torch.uint8:
        raise ValueError(f"canvas must be uint8 {expected}, got {tuple(frame.shape)} {frame.dtype}")
    clip = to_tensor(frame.detach().to(device="cpu").permute(2, 0, 1).contiguous().unsqueeze(0))
    return _CANVAS_NORMALIZE(clip)[0]


def expected_canvas_latent_hw(spatial_compression: int = 32) -> tuple[int, int]:
    """The latent grid of the canvas at the VAE's spatial stride: ``(384 // s, 320 // s)`` = ``(12, 10)`` at 32."""

    stride = int(spatial_compression)
    return OPENWAM_CANVAS_HEIGHT // stride, OPENWAM_CANVAS_WIDTH // stride


def render_canvas_prompt_rows(
    instruction: str,
    *,
    action_mode_text: str,
    include_instruction: bool = True,
    embodiment: str = ROBODOJO_EMBODIMENT,
    view_text: str = OPENWAM_COMPOSITE_VIEW_TEXT,
    groups: int = 1,
) -> tuple[str, ...]:
    """The token-group rows of a canvas (a composited single visual stream), ``build_token_group_prompts`` with ONE view.

    ``groups=1`` (the rwm/openwam canvas class; both canvas modes since 2026-09-22): the composite view's row alone --
    ``Embodiment Type`` / ``Action Mode`` / ``Observation View`` / ``Instruction`` lines joined by ``"\\n"`` -- shared
    by the video and action tokens, the view-less robot row dropped. ``groups=2`` (zekai-merge 2026-09-21 .. 22): the
    standard payload of one view, (composite-view row, robot row). ``include_instruction=False`` renders the CFG
    unconditional rows. ``action_mode_text`` has no default on purpose: it must be the sentence of the checkpoint's
    own action / target modes (``text.action_mode_text``); ``view_text`` is the era's descriptor.
    """

    if int(groups) not in (1, 2):
        raise ValueError(f"a canvas prompt carries G = 1 or G = 2 rows, got {groups!r}")
    common = [prompt_field("Embodiment Type", embodiment), prompt_field("Action Mode", action_mode_text)]
    instruction_lines = [prompt_field("Instruction", instruction)] if include_instruction else []
    view_row = "\n".join(common + [prompt_field("Observation View", view_text)] + instruction_lines)
    if int(groups) == 1:
        return (view_row,)
    return (view_row, "\n".join(common + instruction_lines))


__all__ = [
    "OPENWAM_CAMERA_LAYOUT",
    "OPENWAM_CANVAS_HEIGHT",
    "OPENWAM_CANVAS_WIDTH",
    "OPENWAM_COMPOSITE_VIEW_TEXT",
    "OPENWAM_L_SHAPE_TEXT",
    "OPENWAM_PROMPT_TEXTS",
    "OPENWAM_TOP_HEIGHT_RATIO",
    "OPENWAM_TWO_ROWS_TEXT",
    "OPENWAM_VIEW_KEY",
    "OPENWAM_VIEW_SLOT_IDS",
    "assemble_openwam_canvas",
    "canvas_slot_boxes",
    "canvas_to_model_tensor",
    "expected_canvas_latent_hw",
    "render_canvas_prompt_rows",
]
