"""The view resize of rwm/zekai-merge 1061b16f0 / 77cf81fbf (rwm/mot 034e55dca / f06615b2f): since then an SFT view is
STRETCHED whole to its bucket instead of scaled-to-cover and centre-cropped. The adapter defaults to stretch and keeps crop
for legacy checkpoints (``view_resize``); both, and the sana_pixel canvas built from either, are checked here against the
live tree bit for bit -- the
dataset's clip transform (``ToTensorVideo -> ResizeCrop | StretchResize -> Normalize``), the deploy ``pixel_clip`` and
``assemble_sana_pixel_canvas``. Point ``SANA_REPO`` at a zekai-merge checkout at or after 77cf81fbf (default ~/zekail/Sana).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from torchvision import transforms as T

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_TESTS_DIR))
SANA_REPO = os.environ.get("SANA_REPO", os.path.expanduser("~/zekail/Sana"))
if SANA_REPO not in sys.path and os.path.isdir(SANA_REPO):
    sys.path.insert(0, SANA_REPO)
os.environ.setdefault("DISABLE_XFORMERS", "1")

live_resize = pytest.importorskip("dev.rwm.diffusion.data.view_resize", reason="no Sana tree with view_resize.py (set SANA_REPO)")
live_transforms = pytest.importorskip("diffusion.data.transforms")
live_layout = pytest.importorskip("dev.rwm.diffusion.data.sana_pixel_multiview_layout")
live_deploy = pytest.importorskip("dev.rwm.deploy.common")

from sana_wam_min import pixels  # noqa: E402
from sana_wam_min.sana_pixel_canvas import sana_pixel_canvas_from_frames  # noqa: E402

SOURCES = [(480, 640), (240, 320), (360, 640)]
TARGETS = [(256, 320), (320, 512), (320, 480)]


def _frame(hw, seed):
    return np.random.default_rng(seed).integers(0, 256, size=(*hw, 3), dtype=np.uint8)


def _live_clip(frame, target, resize_cls):
    """The live dataset's per-view clip transform on one uint8 HWC frame -> [3, H_t, W_t]."""

    clip = torch.from_numpy(frame).permute(2, 0, 1)[None].contiguous()
    return T.Compose([live_transforms.ToTensorVideo(), resize_cls(target), T.Normalize([0.5] * 3, [0.5] * 3)])(clip)[0]


@pytest.mark.parametrize("hw", SOURCES)
@pytest.mark.parametrize("target", TARGETS)
def test_a_stretched_view_is_the_live_stretch_resize(hw, target):
    frame = _frame(hw, 1)
    ours = pixels.frame_to_model_tensor(frame, target, "stretch")
    assert torch.equal(ours, _live_clip(frame, target, live_resize.StretchResize))
    assert torch.equal(ours, pixels.frame_to_model_tensor(frame, target))          # stretch is the default


@pytest.mark.parametrize("hw", SOURCES)
@pytest.mark.parametrize("target", TARGETS)
def test_legacy_crop_is_the_live_resize_crop(hw, target):
    frame = _frame(hw, 2)
    ours = pixels.frame_to_model_tensor(frame, target, "crop")
    assert torch.equal(ours, _live_clip(frame, target, live_transforms.ResizeCrop))
    if hw[0] * target[1] != hw[1] * target[0]:            # different aspect: the two modes really differ
        assert not torch.equal(ours, pixels.frame_to_model_tensor(frame, target))


def test_stretch_is_the_live_deploy_pixel_clip():
    frame = _frame((480, 640), 3)
    chw = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
    live = live_deploy.pixel_clip(chw, (256, 320), aspect_ratios=pixels.ASPECT_RATIO_VIDEO_320_ROBOT)
    assert torch.equal(pixels.frame_to_model_tensor(frame, (256, 320), "stretch"), live[0])


@pytest.mark.parametrize("canvas_hw", [(320, 512), (320, 480)])
@pytest.mark.parametrize("mode", ["crop", "stretch"])
def test_the_sana_pixel_canvas_is_the_live_tiling_of_the_resized_views(canvas_hw, mode):
    frames = [_frame((480, 640), seed) for seed in (11, 12, 13)]
    slots = list(live_layout.SANA_PIXEL_VIEW_SLOTS)
    resize_cls = live_resize.StretchResize if mode == "stretch" else live_transforms.ResizeCrop
    views = tuple(_live_clip(frame, canvas_hw, resize_cls)[None] for frame in frames)
    live = live_layout.assemble_sana_pixel_canvas(views, slots, *canvas_hw)[0]
    ours = sana_pixel_canvas_from_frames(frames, slots, canvas_hw, view_resize=mode)
    assert ours.shape == live.shape and torch.equal(ours, live)


def test_view_resize_values_are_checked():
    assert pixels.validate_view_resize(" Stretch ") == "stretch"
    with pytest.raises(ValueError):
        pixels.validate_view_resize("pad")
