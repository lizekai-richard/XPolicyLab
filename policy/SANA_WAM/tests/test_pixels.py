"""CPU tests for sana_wam_min.pixels against the training transform recipe re-derived with torch ops."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ADAPTER_DIR = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/XPolicyLab/policy/SANA_WAM"
if ADAPTER_DIR not in sys.path:
    sys.path.insert(0, ADAPTER_DIR)

from sana_wam_min import pixels  # noqa: E402


def _gradient_frame(h: int, w: int) -> np.ndarray:
    ys = np.linspace(0, 255, h, dtype=np.float64)[:, None]
    xs = np.linspace(0, 255, w, dtype=np.float64)[None, :]
    r = np.broadcast_to(xs, (h, w))
    g = np.broadcast_to(ys, (h, w))
    b = (xs + ys) / 2.0
    return np.stack([r, g, np.broadcast_to(b, (h, w))], axis=-1).round().astype(np.uint8)


def _reference(frame_hwc: np.ndarray, th: int, tw: int) -> torch.Tensor:
    """ToTensorVideo -> ResizeCrop -> Normalize(0.5, 0.5) written out with raw torch ops."""

    clip = torch.from_numpy(frame_hwc).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    h, w = clip.shape[-2:]
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        i, j = 0, int(round(sw - tw) / 2.0)
    else:
        sh, sw = round(h * rw), tw
        i, j = int(round(sh - th) / 2.0), 0
    clip = F.interpolate(clip, size=(sh, sw), mode="bilinear", align_corners=False)
    clip = clip[..., i : i + th, j : j + tw]
    return ((clip - 0.5) / 0.5)[0]


def test_target_size_single_bucket():
    assert pixels.target_size_hw(320) == (256, 320)
    assert pixels.target_size_hw(320, frame_hw=(480, 640)) == (256, 320)
    assert pixels.target_size_hw(320, frame_hw=(1080, 1920)) == (256, 320)
    assert pixels.target_size_hw(480) == (480, 640)
    size, key = pixels.get_closest_ratio(480, 640, pixels.ASPECT_RATIO_VIDEO_320_ROBOT)
    assert size == [256.0, 320.0] and key == 0.8


def test_resize_crop_geometry_offsets():
    assert pixels.resize_crop_geometry(480, 640, 256, 320) == (256, 341, 0, 10)
    assert pixels.resize_crop_geometry(720, 1280, 256, 320) == (256, 455, 0, 67)
    assert pixels.resize_crop_geometry(480, 848, 256, 320) == (256, 452, 0, 66)
    assert pixels.resize_crop_geometry(240, 320, 256, 320) == (256, 341, 0, 10)
    assert pixels.resize_crop_geometry(256, 320, 256, 320) == (256, 320, 0, 0)
    assert pixels.resize_crop_geometry(480, 640, 480, 640) == (480, 640, 0, 0)


def test_frame_to_model_tensor_matches_reference_640x480():
    frame = _gradient_frame(480, 640)
    out = pixels.frame_to_model_tensor(frame, (256, 320))
    assert out.shape == (3, 256, 320)
    assert out.dtype == torch.float32
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0
    assert torch.equal(out, _reference(frame, 256, 320))
    # The crop drops 10 px on the left: column 0 of the output is column 10 of the 256x341 resize.
    resized = F.interpolate(
        torch.from_numpy(frame).permute(2, 0, 1)[None].float() / 255.0,
        size=(256, 341),
        mode="bilinear",
        align_corners=False,
    )
    assert torch.equal(out[:, :, 0], ((resized - 0.5) / 0.5)[0, :, :, 10])
    assert torch.equal(out[:, :, -1], ((resized - 0.5) / 0.5)[0, :, :, 329])
    assert not torch.equal(out[:, :, 0], ((resized - 0.5) / 0.5)[0, :, :, 0])


def test_frame_to_model_tensor_torch_input_and_identity_size():
    frame = torch.from_numpy(_gradient_frame(256, 320))
    out = pixels.frame_to_model_tensor(frame, (256, 320))
    assert torch.equal(out, (frame.permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5)
    # The caller's uint8 frame is not mutated by the in-place Normalize.
    assert frame.dtype == torch.uint8


def test_frame_to_model_tensor_zeros_and_ones():
    zeros = np.zeros((480, 640, 3), dtype=np.uint8)
    assert torch.equal(pixels.frame_to_model_tensor(zeros, (256, 320)), torch.full((3, 256, 320), -1.0))
    ones = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert torch.equal(pixels.frame_to_model_tensor(ones, (256, 320)), torch.full((3, 256, 320), 1.0))


def test_frame_to_model_tensor_rejects_wrong_layout():
    with pytest.raises(ValueError):
        pixels.frame_to_model_tensor(np.zeros((3, 480, 640), dtype=np.uint8), (256, 320))
    with pytest.raises((ValueError, TypeError)):
        pixels.frame_to_model_tensor(np.zeros((480, 640, 3), dtype=np.float32), (256, 320))


def test_frames_to_vae_input_layout():
    clip = torch.randn(25, 3, 256, 320)
    vae_in = pixels.frames_to_vae_input(clip)
    assert vae_in.shape == (1, 3, 25, 256, 320)
    assert torch.equal(vae_in[0, :, 7], clip[7])


def test_observation_window_zero_fill():
    latent = torch.randn(1, 128, 1, 8, 10, dtype=torch.bfloat16)
    window = pixels.observation_window(latent, pixels.latent_frame_count(25))
    assert window.shape == (1, 128, 4, 8, 10)
    assert window.dtype == torch.bfloat16
    assert torch.equal(window[:, :, :1], latent)
    assert torch.count_nonzero(window[:, :, 1:]) == 0
    with pytest.raises(ValueError):
        pixels.observation_window(torch.zeros(1, 128, 4, 8, 10), 4)


def test_latent_frame_count():
    assert pixels.latent_frame_count(25) == 4
    assert pixels.latent_frame_count(1) == 1
    assert pixels.latent_frame_count(33) == 5


def test_tier():
    assert pixels.tier(25) == (25, 24)
    assert pixels.tier(25.0, num_frames=25) == (25, 24)
    assert pixels.tier(24.6) == (25, 24)
    with pytest.raises(ValueError):
        pixels.tier(30)
    with pytest.raises(ValueError):
        pixels.tier(25, num_frames=33)
    assert pixels.tier(30, multi_fps={30: [17, 33]}) == (33, 32)


def test_view_latent_shape_tensor():
    t = pixels.view_latent_shape_tensor(((8, 10), (8, 10), (8, 10)))
    assert t.shape == (1, 3, 2) and t.dtype == torch.int64
    assert t.tolist() == [[[8, 10], [8, 10], [8, 10]]]


def test_package_import_paths():
    import importlib

    a = importlib.import_module("sana_wam_min.pixels")
    b = importlib.import_module("XPolicyLab.policy.SANA_WAM.sana_wam_min.pixels")
    assert a.target_size_hw(320) == b.target_size_hw(320) == (256, 320)
