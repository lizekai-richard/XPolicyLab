"""Strided video frames: the window keeps every source row for the actions while the video samples every s-th row.

Port of Sana ``dev/rwm/diffusion/data/video_frame_stride.py`` (branch ``rwm/strided_video``, design doc
``dev/rwm/docs/strided_video_design.md``). The training yaml declares ``data.extra.robot_sft.video_fps``: the
number of video frames sampled AFTER the observation frame of a ``rows``-row window (row 0 is not one of them). The
stride is ``s = (rows - 1) / video_fps`` (an integer or the config is refused) and the video of the window is the
frames at rows ``0, s, 2s, ..., rows - 1`` -- ``video_fps + 1`` frames, which must be ``1 + 8k`` for the causal VAE.
The action rows stay ``rows - 1`` (dense), the normalization artifact and the prompt stay keyed on rows.

At deployment only the observation frame is real, so a strided checkpoint changes two things in this adapter: the
observation encode fills a SHORTER latent window (``1 + video_fps / 8`` frames instead of ``1 + (rows - 1) / 8``)
and the batch carries ``data_info["video_frame_stride"]``, which the policy port reads to expect ``(F - 1) * 8 * s``
action rows and to switch the robot tail to the independent action RoPE (``policy_model/model.py``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

VIDEO_FPS_KEY = "video_fps"
VIDEO_FRAME_STRIDE_KEY = "video_frame_stride"


def parse_video_fps(value) -> Optional[int]:
    """The number of video frames sampled after the observation frame (a positive integer); None keeps every frame."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{VIDEO_FPS_KEY} must be a positive integer (sampled frames per window), got {value!r}")
    fps = int(value)
    if fps < 1:
        raise ValueError(f"{VIDEO_FPS_KEY} must be a positive integer (sampled frames per window), got {value!r}")
    return fps


def video_frame_stride_from_video_fps(window_rows: int, video_fps: int) -> int:
    """The frame stride that samples ``video_fps`` frames after the observation frame of a ``window_rows``-row window:
    ``(window_rows - 1) // video_fps``; frame j of the video is row ``j * stride``. ``window_rows - 1`` must be a
    positive multiple of ``video_fps``."""

    rows, fps = int(window_rows), int(video_fps)
    if rows < 2 or fps < 1:
        raise ValueError(f"window_rows must be >= 2 and {VIDEO_FPS_KEY} >= 1, got {window_rows} and {video_fps}")
    if (rows - 1) % fps:
        raise ValueError(
            f"a window of {rows} rows cannot be sampled at {VIDEO_FPS_KEY} {fps}: rows - 1 = {rows - 1} "
            f"must be a multiple of the sampled frame count"
        )
    return (rows - 1) // fps


def strided_video_frames(window_rows: int, stride: int, temporal_stride: int) -> int:
    """How many video frames a ``window_rows``-row window yields when frame j is row ``j * stride``:
    ``1 + (window_rows - 1) // stride``, which must be ``1 + k * temporal_stride`` (``k >= 1``) for the causal VAE.
    Stride 1 is the dense window (the historical contract)."""

    rows, stride, temporal = int(window_rows), int(stride), int(temporal_stride)
    if rows < 2 or stride < 1 or temporal < 1:
        raise ValueError(
            f"window_rows must be >= 2 and the strides >= 1, got rows={rows} stride={stride} temporal={temporal}"
        )
    if (rows - 1) % stride:
        raise ValueError(
            f"a window of {rows} rows does not tile a video frame stride of {stride}: rows - 1 must be a multiple of the stride"
        )
    frames = 1 + (rows - 1) // stride
    if frames < 1 + temporal or (frames - 1) % temporal:
        raise ValueError(
            f"a window of {rows} rows at video frame stride {stride} gives {frames} video frames; "
            f"the causal VAE needs 1 + k * {temporal} frames with k >= 1"
        )
    return frames


__all__ = [
    "VIDEO_FPS_KEY",
    "VIDEO_FRAME_STRIDE_KEY",
    "parse_video_fps",
    "strided_video_frames",
    "video_frame_stride_from_video_fps",
]
