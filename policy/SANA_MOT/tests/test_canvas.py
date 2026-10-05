"""The OpenWAM L-shape canvas: slot geometry, PIL parity with Sana's compositor, the clip transform."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
import _tiny_mot  # noqa: E402,F401  (sys.path)

from sana_mot_min.canvas import (  # noqa: E402
    OPENWAM_CAMERA_LAYOUT,
    OPENWAM_CANVAS_HEIGHT,
    OPENWAM_CANVAS_WIDTH,
    assemble_openwam_canvas,
    canvas_from_frames,
    canvas_latent_hw,
    canvas_to_model_tensor,
    slot_geometry,
)
from sana_wam_min.robodojo_io import ROBODOJO_VIEW_ORDER  # noqa: E402

SANA_MOT_REPO = os.environ.get("SANA_MOT_REPO", os.path.expanduser("~/zekail/Sana_mot"))


def solid_frames(shape=(480, 640, 3)) -> list[np.ndarray]:
    colors = ((200, 10, 10), (10, 200, 10), (10, 10, 200))
    return [np.full(shape, color, dtype=np.uint8) for color in colors]


def test_slot_geometry_is_the_openwam_l_shape():
    geometry = slot_geometry()
    assert tuple(OPENWAM_CAMERA_LAYOUT) == tuple(ROBODOJO_VIEW_ORDER)
    assert geometry["cam_head"] == (0, 0, 256, 320)
    assert geometry["cam_left_wrist"] == (256, 0, 128, 160)
    assert geometry["cam_right_wrist"] == (256, 160, 128, 160)
    assert (OPENWAM_CANVAS_HEIGHT, OPENWAM_CANVAS_WIDTH) == (384, 320) and canvas_latent_hw() == (12, 10)


def test_canvas_places_each_camera_in_its_slot():
    canvas = canvas_from_frames(solid_frames())
    assert canvas.shape == (384, 320, 3) and canvas.dtype == np.uint8
    assert np.all(canvas[:256, :] == (200, 10, 10))
    assert np.all(canvas[256:, :160] == (10, 200, 10))
    assert np.all(canvas[256:, 160:] == (10, 10, 200))
    # arbitrary source sizes are stretched into the slots; wrong counts / dtypes are refused
    assert canvas_from_frames(solid_frames((97, 123, 3))).shape == (384, 320, 3)
    with pytest.raises(ValueError, match="expected 3 frames"):
        canvas_from_frames(solid_frames()[:2])
    with pytest.raises(ValueError, match="uint8"):
        canvas_from_frames([f.astype(np.float32) for f in solid_frames()])
    with pytest.raises(KeyError):
        assemble_openwam_canvas({"cam_head": solid_frames()[0]})


def test_canvas_is_rgb_and_never_swapped():
    frames = solid_frames()
    canvas = canvas_from_frames(frames)
    assert tuple(canvas[0, 0]) == (200, 10, 10)          # head is red, red stays in channel 0
    assert tuple(canvas[300, 300]) == (10, 10, 200)      # right wrist is blue, blue stays in channel 2


def test_canvas_to_model_tensor_range_and_layout():
    frames = solid_frames()
    tensor = canvas_to_model_tensor(canvas_from_frames(frames))
    assert tensor.shape == (3, 384, 320) and tensor.dtype == torch.float32
    assert torch.isclose(tensor[0, 0, 0], torch.tensor(200 / 255 * 2 - 1)) and torch.isclose(tensor[1, 0, 0], torch.tensor(10 / 255 * 2 - 1))
    assert tensor.min() >= -1 and tensor.max() <= 1
    with pytest.raises(ValueError):
        canvas_to_model_tensor(np.zeros((384, 320, 3), dtype=np.float32))


def test_pixel_parity_with_sana_compositor():
    if SANA_MOT_REPO not in sys.path and os.path.isdir(SANA_MOT_REPO):
        sys.path.insert(0, SANA_MOT_REPO)
    live = pytest.importorskip("dev.rwm.diffusion.data.openwam_multiview_layout", reason="Sana rwm/mot checkout (and cv2) not importable")
    from PIL import Image

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(3)]
    ours = canvas_from_frames(frames)
    reference = np.asarray(
        live.assemble_openwam_canvas({cam: Image.fromarray(f) for cam, f in zip(live.OPENWAM_CAMERA_LAYOUT, frames)})
    )
    assert tuple(live.OPENWAM_CAMERA_LAYOUT) == tuple(OPENWAM_CAMERA_LAYOUT)
    assert (live.OPENWAM_CANVAS_HEIGHT, live.OPENWAM_CANVAS_WIDTH) == (OPENWAM_CANVAS_HEIGHT, OPENWAM_CANVAS_WIDTH)
    assert np.array_equal(ours, reference)
