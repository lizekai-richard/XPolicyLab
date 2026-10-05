"""The ``sana_pixel`` multiview canvas: the semantic 2x2 quadrants composited in PIXEL space (2026-09-22).

Ports of Sana ``dev/rwm/diffusion/data/sana_pixel_multiview_layout.py`` (``assemble_sana_pixel_canvas``) and of the
canvas dataset's front-end (``datasets/robodojo_sana_pixel_canvas_sft_data.py``, identical on rwm/zekai-merge and
rwm/mot). Training: the base multiview loader decodes every camera at the CANVAS size -- the recipe's
``aspect_ratio_type: ASPECT_RATIO_SANA_PIXEL_2X2_320_480`` is a one-bucket table, so each view goes through the
standard clip transform ``ToTensorVideo -> ResizeCrop(320, 480) -> Normalize(0.5, 0.5)`` -- then every view is
bilinearly resized (``F.interpolate``, align_corners False, no antialias: an exact 2x2 average) into its 160 x 240
quadrant of ONE 320 x 480 canvas on the same semantic slots the ``sana_latent`` spatial RoPE uses (slot ``s`` at
``divmod(s, 2)``: head top-left, left wrist bottom-left, right wrist bottom-right) and the unclaimed top-right
quadrant is filled with -1.0, black in the normalized domain (user ruling 2026-09-23). The canvas is encoded once
into a ``[128, F, 10, 15]`` latent, ``view_count == 1``, and the policy runs its single-view path (plain mRoPE over the
10 x 15 grid) with ONE prompt row naming the tiling. Unlike the OpenWAM canvas the pixels come from the converted
package's per-view clips, so live RGB frames follow exactly the per-view transform of the strip line.

The recipe's ``data.aspect_ratio_type`` picks the canvas (Sana ``SANA_PIXEL_CANVASES``, zekai-merge 88a22ba0c, 2026-09-23):
320 x 480 (160 x 240 tiles, ``[128, F, 10, 15]``) or 320 x 512 (160 x 256 tiles on whole 32 x 32 VAE cells,
``[128, F, 10, 16]``); every step above is the same with the other size, and an unknown type is refused.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .pixels import frame_to_model_tensor

SANA_PIXEL_CANVAS_HEIGHT = 320
SANA_PIXEL_CANVAS_WIDTH = 480
# data.aspect_ratio_type -> canvas (height, width), as Sana's SANA_PIXEL_CANVASES (zekai-merge 88a22ba0c)
SANA_PIXEL_ASPECT_320_480 = "ASPECT_RATIO_SANA_PIXEL_2X2_320_480"
SANA_PIXEL_ASPECT_320_512 = "ASPECT_RATIO_SANA_PIXEL_2X2_320_512"
SANA_PIXEL_CANVASES = {
    SANA_PIXEL_ASPECT_320_480: (SANA_PIXEL_CANVAS_HEIGHT, SANA_PIXEL_CANVAS_WIDTH),
    SANA_PIXEL_ASPECT_320_512: (320, 512),
}
SANA_PIXEL_GRID = (2, 2)
SANA_PIXEL_LAYOUT_ID = "sana_pixel_2x2_rgb_v1"
SANA_PIXEL_FILL_VALUE = -1.0
# Model-facing metadata of the one composite stream (the dataset's ``_rewrite_sana_pixel_data_info``).
SANA_PIXEL_VIEW_KEY = "sana_pixel_canvas"
SANA_PIXEL_VIEW_SLOT_IDS = (0,)
# Observation View text of the composite-view row (``_SANA_PIXEL_COMPOSITE_DESCRIPTOR``), by era:
#   composite_view -- the G = 2 payload of zekai-merge b71474ad1 .. e4bb170ad / rwm/mot b978b9ef0 .. 2b4a4dc8d (smokes);
#   tiling         -- the ONE shared row from zekai-merge e4bb170ad / rwm/mot 2b4a4dc8d on.
SANA_PIXEL_COMPOSITE_VIEW_TEXT = (
    "a composite view with the head camera top left and the left and right wrist cameras below"
)
SANA_PIXEL_TILING_TEXT = (
    "one image tiling the robot's cameras in a two by two grid, with the head camera in the top left "
    "quadrant, the left wrist camera in the bottom left quadrant, the right wrist camera in the bottom "
    "right quadrant, and the remaining top right quadrant filled black"
)
SANA_PIXEL_PROMPT_TEXTS = {
    "composite_view": SANA_PIXEL_COMPOSITE_VIEW_TEXT,
    "tiling": SANA_PIXEL_TILING_TEXT,
}


def sana_pixel_canvas_hw(aspect_ratio_type: Optional[str]) -> tuple[int, int]:
    """The canvas ``(height, width)`` a recipe's ``data.aspect_ratio_type`` names (Sana ``sana_pixel_canvas_hw``);
    ``None`` = 320 x 480, the only canvas before 88a22ba0c. A type outside ``SANA_PIXEL_CANVASES`` is refused."""

    if aspect_ratio_type is None:
        return SANA_PIXEL_CANVASES[SANA_PIXEL_ASPECT_320_480]
    if aspect_ratio_type not in SANA_PIXEL_CANVASES:
        raise ValueError(
            f"multiview sana_pixel needs data.aspect_ratio_type in {sorted(SANA_PIXEL_CANVASES)}, got {aspect_ratio_type!r}"
        )
    return SANA_PIXEL_CANVASES[aspect_ratio_type]


def sana_pixel_tile_shape(
    canvas_height: int = SANA_PIXEL_CANVAS_HEIGHT, canvas_width: int = SANA_PIXEL_CANVAS_WIDTH
) -> tuple[int, int]:
    """Return the ``(height, width)`` of one quadrant of the canvas."""

    rows, columns = SANA_PIXEL_GRID
    if int(canvas_height) % rows or int(canvas_width) % columns:
        raise ValueError(
            f"the canvas {(canvas_height, canvas_width)} must split into a {rows}x{columns} grid of equal tiles"
        )
    return int(canvas_height) // rows, int(canvas_width) // columns


def assemble_sana_pixel_canvas(
    views: Sequence[torch.Tensor],
    slot_ids: Sequence[int],
    canvas_height: int = SANA_PIXEL_CANVAS_HEIGHT,
    canvas_width: int = SANA_PIXEL_CANVAS_WIDTH,
    fill_value: float = SANA_PIXEL_FILL_VALUE,
) -> torch.Tensor:
    """Tile per-view ``[F, C, H, W]`` clips into one ``[F, C, canvas_height, canvas_width]`` canvas clip.

    Line-for-line port of the training ``assemble_sana_pixel_canvas``: each view is bilinearly resized (in float32,
    cast back) into the quadrant of its slot (``0`` top-left, ``1`` top-right, ``2`` bottom-left, ``3`` bottom-right);
    quadrants no view claims keep ``fill_value``.
    """

    clips = list(views)
    slots = [int(slot) for slot in slot_ids]
    if not clips or len(clips) != len(slots):
        raise ValueError(f"one slot per view is required, got {len(clips)} views and {len(slots)} slots")
    if len(set(slots)) != len(slots) or any(slot < 0 or slot > 3 for slot in slots):
        raise ValueError(f"the 2x2 canvas needs unique slots in [0, 3], got {slots}")
    frames, channels = clips[0].shape[0], clips[0].shape[1]
    for clip in clips:
        if clip.ndim != 4 or clip.shape[0] != frames or clip.shape[1] != channels:
            raise ValueError(
                f"every view clip must be [F, C, H, W] with F={frames} and C={channels}, got {tuple(clip.shape)}"
            )
    tile_height, tile_width = sana_pixel_tile_shape(canvas_height, canvas_width)
    canvas = clips[0].new_full((frames, channels, int(canvas_height), int(canvas_width)), float(fill_value))
    for clip, slot in zip(clips, slots, strict=True):
        tile = clip
        if tile.shape[-2:] != (tile_height, tile_width):
            tile = F.interpolate(
                clip.to(torch.float32), size=(tile_height, tile_width), mode="bilinear", align_corners=False
            ).to(clip.dtype)
        row, column = divmod(slot, 2)
        top, left = row * tile_height, column * tile_width
        canvas[:, :, top : top + tile_height, left : left + tile_width] = tile
    return canvas


def sana_pixel_canvas_from_frames(
    frames_rgb: Sequence[np.ndarray],
    slot_ids: Sequence[int],
    canvas_hw: tuple[int, int] = (SANA_PIXEL_CANVAS_HEIGHT, SANA_PIXEL_CANVAS_WIDTH),
    view_resize: str = "stretch",
) -> torch.Tensor:
    """RGB uint8 HxWx3 frames (view order) -> float32 ``[3, H, W]`` canvas in ``[-1, 1]`` (``canvas_hw``, 320 x 480 by
    default, 320 x 512 for the ``..._320_512`` recipes), on CPU.

    Every frame takes the training clip transform at the canvas bucket (``StretchResize`` of the whole frame -- or, for
    legacy ``view_resize="crop"`` checkpoints, ``ResizeCrop`` to fill the canvas -- then ``Normalize``), exactly as the
    base loader decodes each view of a sana_pixel recipe, and the quadrants are tiled by
    :func:`assemble_sana_pixel_canvas`.
    """

    if len(frames_rgb) != len(slot_ids):
        raise ValueError(f"expected one frame per slot {tuple(slot_ids)}, got {len(frames_rgb)} frames")
    target = (int(canvas_hw[0]), int(canvas_hw[1]))
    views = [frame_to_model_tensor(frame, target, view_resize).unsqueeze(0) for frame in frames_rgb]
    return assemble_sana_pixel_canvas(views, slot_ids, target[0], target[1])[0]


def expected_sana_pixel_latent_hw(
    spatial_compression: int = 32, canvas_hw: tuple[int, int] = (SANA_PIXEL_CANVAS_HEIGHT, SANA_PIXEL_CANVAS_WIDTH)
) -> tuple[int, int]:
    """The latent grid of the canvas at the VAE's spatial stride: ``(10, 15)`` for 320 x 480, ``(10, 16)`` for 320 x 512."""

    stride = int(spatial_compression)
    return int(canvas_hw[0]) // stride, int(canvas_hw[1]) // stride


__all__ = [
    "SANA_PIXEL_ASPECT_320_480",
    "SANA_PIXEL_ASPECT_320_512",
    "SANA_PIXEL_CANVASES",
    "SANA_PIXEL_CANVAS_HEIGHT",
    "SANA_PIXEL_CANVAS_WIDTH",
    "SANA_PIXEL_COMPOSITE_VIEW_TEXT",
    "SANA_PIXEL_FILL_VALUE",
    "SANA_PIXEL_GRID",
    "SANA_PIXEL_LAYOUT_ID",
    "SANA_PIXEL_PROMPT_TEXTS",
    "SANA_PIXEL_TILING_TEXT",
    "SANA_PIXEL_VIEW_KEY",
    "SANA_PIXEL_VIEW_SLOT_IDS",
    "assemble_sana_pixel_canvas",
    "expected_sana_pixel_latent_hw",
    "sana_pixel_canvas_from_frames",
    "sana_pixel_canvas_hw",
    "sana_pixel_tile_shape",
]
