"""Pixel front-end of the policy: the training-time RGB transform and the latent observation window.

Ports of ``diffusion/data/transforms.py`` (``ToTensorVideo -> ResizeCrop -> Normalize``), the
robot aspect-ratio buckets, and the deployment session's ``tier`` / ``_pixel_frame`` /
``_encode_observation_views`` window-forming logic. Everything runs on CPU in float32; the
caller casts the result to the VAE dtype and device.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torchvision import transforms as T

# Single-bucket tables: every input resolution resolves to the one entry.
ASPECT_RATIO_VIDEO_320_ROBOT = {"0.8": [256.0, 320.0]}
ASPECT_RATIO_VIDEO_480_ROBOT = {"0.75": [480.0, 640.0]}
ROBOT_ASPECT_RATIO_TABLES = {
    320: ASPECT_RATIO_VIDEO_320_ROBOT,
    480: ASPECT_RATIO_VIDEO_480_ROBOT,
}

# Trained (fps -> pixel-frame counts) tiers; the longest option is the default horizon.
DEFAULT_MULTI_FPS = {25: [25]}


def get_closest_ratio(height: float, width: float, ratios: dict) -> tuple[list, float]:
    """Return the bucket ``(size, key)`` whose aspect ratio is closest to ``height / width``."""

    aspect_ratio = height / width
    max_key = max(float(k) for k in ratios.keys())
    if max_key >= 100:
        closest_ratio = min(ratios.keys(), key=lambda r: abs((float(r) % 100) - aspect_ratio))
    else:
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


def target_size_hw(image_size: int = 320, frame_hw: Optional[tuple[int, int]] = None) -> tuple[int, int]:
    """Return the ``(H, W)`` pixel bucket for ``image_size`` (320 -> (256, 320); 480 -> (480, 640))."""

    table = ROBOT_ASPECT_RATIO_TABLES[int(image_size)]
    height, width = (frame_hw if frame_hw is not None else (1, 1))
    size, _ = get_closest_ratio(int(height), int(width), table)
    return int(size[0]), int(size[1])


def to_tensor(clip: torch.Tensor) -> torch.Tensor:
    """Convert a uint8 ``[T, C, H, W]`` clip to float32 in ``[0, 1]`` (no permute)."""

    if clip.dtype != torch.uint8:
        raise TypeError(f"clip tensor should have data type uint8. Got {clip.dtype}")
    return clip.float() / 255.0


def resize(clip: torch.Tensor, target_size: tuple[int, int], interpolation_mode: str) -> torch.Tensor:
    """Resize ``[N, C, H, W]`` with ``F.interpolate`` (align_corners=False, antialias off)."""

    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=False)


def crop(clip: torch.Tensor, i: int, j: int, h: int, w: int) -> torch.Tensor:
    """Crop the ``h x w`` window at ``(i, j)`` from a ``[..., H, W]`` clip."""

    return clip[..., i : i + h, j : j + w]


def resize_crop_geometry(h: int, w: int, th: int, tw: int) -> tuple[int, int, int, int]:
    """Return ``(sh, sw, i, j)``: the fill-resize size and the crop offset for ``h x w -> th x tw``.

    ``int(round(sw - tw) / 2.0)`` rounds the difference, halves, then truncates (odd
    differences truncate down, e.g. 21 -> 10); ``round`` is Python banker's rounding.
    """

    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        i = 0
        j = int(round(sw - tw) / 2.0)
    else:
        sh, sw = round(h * rw), tw
        i = int(round(sh - th) / 2.0)
        j = 0
    return sh, sw, i, j


def resize_crop_to_fill(clip: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Bilinear-resize ``[T, C, H, W]`` to fill ``target_size`` and center-crop the overflow."""

    h, w = clip.size(-2), clip.size(-1)
    th, tw = target_size[0], target_size[1]
    sh, sw, i, j = resize_crop_geometry(h, w, th, tw)
    clip = resize(clip, (sh, sw), "bilinear")
    assert i + th <= clip.size(-2) and j + tw <= clip.size(-1)
    return crop(clip, i, j, th, tw)


class ToTensorVideo:
    """Callable wrapper of :func:`to_tensor`."""

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return to_tensor(clip)


class ResizeCrop:
    """Callable wrapper of :func:`resize_crop_to_fill` with a fixed target size."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = (int(size), int(size)) if isinstance(size, (int, float)) else size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return resize_crop_to_fill(clip, self.size)


def video_transform(target_hw: tuple[int, int]) -> T.Compose:
    """The training clip transform ``ToTensorVideo -> ResizeCrop -> Normalize(0.5, 0.5)``."""

    return T.Compose(
        [
            ToTensorVideo(),
            ResizeCrop(tuple(int(v) for v in target_hw)),
            T.Normalize([0.5] * 3, [0.5] * 3, inplace=True),
        ]
    )


def frame_to_model_tensor(rgb_uint8_hwc, target_hw: tuple[int, int]) -> torch.Tensor:
    """Map one RGB uint8 ``[H, W, 3]`` frame (numpy or torch) to float32 ``[3, H_t, W_t]`` in ``[-1, 1]``.

    Runs on CPU in float32 exactly like the training dataset transform; the frame must be
    true RGB (BGR input silently inverts colours downstream).
    """

    frame = torch.as_tensor(np.asarray(rgb_uint8_hwc) if not torch.is_tensor(rgb_uint8_hwc) else rgb_uint8_hwc)
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != torch.uint8:
        raise ValueError(f"frame must be uint8 [H, W, 3], got {tuple(frame.shape)} {frame.dtype}")
    chw = frame.detach().to(device="cpu").permute(2, 0, 1).contiguous()
    clip = video_transform(target_hw)(chw.unsqueeze(0))
    return clip[0]


def frames_to_vae_input(frames_chw: torch.Tensor) -> torch.Tensor:
    """Reshape a float ``[F, 3, H, W]`` clip to the VAE layout ``[1, 3, F, H, W]``."""

    return frames_chw.permute(1, 0, 2, 3).unsqueeze(0)


def latent_frame_count(frames: int, temporal_compression: int = 8) -> int:
    """Return the latent frame count ``(frames - 1) // t + 1`` of a ``frames``-frame clip."""

    return (int(frames) - 1) // int(temporal_compression) + 1


def observation_window(latent_frame0: torch.Tensor, latent_frames: int) -> torch.Tensor:
    """Place the ``[1, C, 1, h, w]`` observation latent at frame 0 of a zero ``[1, C, latent_frames, h, w]`` window.

    Frames 1.. are placeholders only (the sampler starts them from noise and re-pins frame 0).
    """

    if latent_frame0.ndim != 5 or latent_frame0.shape[0] != 1 or latent_frame0.shape[2] != 1:
        raise ValueError(f"observation latent must be [1, C, 1, h, w], got {tuple(latent_frame0.shape)}")
    window = latent_frame0.new_zeros(
        (1, latent_frame0.shape[1], int(latent_frames), latent_frame0.shape[-2], latent_frame0.shape[-1])
    )
    window[:, :, :1] = latent_frame0
    return window


def view_latent_shape_tensor(view_shapes, device="cpu") -> torch.Tensor:
    """Return the ``[1, V, 2]`` int64 per-view latent ``(h, w)`` tensor the model consumes."""

    return torch.tensor([[list(shape) for shape in view_shapes]], dtype=torch.int64, device=device)


def tier(fps: float, num_frames: Optional[int] = None, multi_fps: Optional[dict] = None) -> tuple[int, int]:
    """Map ``fps`` to ``(pixel frames, K action rows)``; default = the longest trained horizon."""

    tiers = DEFAULT_MULTI_FPS if multi_fps is None else {int(k): list(v) for k, v in multi_fps.items()}
    key = int(round(float(fps)))
    if key not in tiers:
        raise ValueError(f"fps {fps} is not a trained tier; multi_fps tiers: {sorted(tiers)}")
    options = tiers[key]
    frames = options[-1] if num_frames is None else int(num_frames)
    if frames not in options:
        raise ValueError(f"num_frames={num_frames} is not a trained horizon for fps {key}; trained: {options}")
    return frames, frames - 1


__all__ = [
    "ASPECT_RATIO_VIDEO_320_ROBOT",
    "ASPECT_RATIO_VIDEO_480_ROBOT",
    "DEFAULT_MULTI_FPS",
    "ResizeCrop",
    "ToTensorVideo",
    "frame_to_model_tensor",
    "frames_to_vae_input",
    "get_closest_ratio",
    "latent_frame_count",
    "observation_window",
    "resize_crop_geometry",
    "resize_crop_to_fill",
    "target_size_hw",
    "tier",
    "video_transform",
    "view_latent_shape_tensor",
]
