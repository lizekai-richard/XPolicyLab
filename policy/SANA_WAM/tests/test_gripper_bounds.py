"""The 2026-09-20 gripper normalization vs Sana's [0, 1] clamp of the NORMALIZED gripper slots.

Sana's sampler clamps action slots 16 / 45 to [0, 1] after the loop -- the raw closedness range while the grippers were
identity-mapped. The 2026-09-20 artifacts map raw [0, 1] to [-1, 1] (center 0.5, scale 0.5), so that clamp cut the
whole open half off: the first two checkpoints on the scheme (mot_jointabs_f33fps8_openwam,
sanavideo_eefabs_f33fps8_sana_pixel) never commanded an opening above 0.5 and collapsed in the simulator. The session
now clamps to the model-domain image of raw [0, 1] (robot80.normalized_gripper_bounds): identical for the old artifacts.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from test_sampler import SHIFT, STEPS, _make_inputs, _StubModel  # noqa: E402  (puts policy/SANA_WAM on sys.path)
from test_session_cpu import fake_frames, fake_state, make_session  # noqa: E402

from sana_wam_min import session as session_mod  # noqa: E402
from sana_wam_min.robodojo_io import opening_from_closedness, state80_from_obs  # noqa: E402
from sana_wam_min.robot80 import denormalize_action, load_normalization, normalized_gripper_bounds  # noqa: E402
from sana_wam_min.sampler import LEFT_GRIPPER, RIGHT_GRIPPER, sample_policy  # noqa: E402
from sana_wam_min.session import PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256  # noqa: E402


def _stamped(norm):
    """The packaged artifact with the 2026-09-20 gripper statistics (center 0.5, scale 0.5) on slots 16 / 45."""

    stamped = copy.copy(norm)
    for kind in ("state", "action"):
        mask = np.array(getattr(norm, f"{kind}_normalization_mask80"), copy=True)
        center = np.array(getattr(norm, f"{kind}_center80"), copy=True)
        scale = np.array(getattr(norm, f"{kind}_scale80"), copy=True)
        mask[[16, 45]], center[[16, 45]], scale[[16, 45]] = True, 0.5, 0.5
        object.__setattr__(stamped, f"{kind}_normalization_mask80", mask)
        object.__setattr__(stamped, f"{kind}_center80", center)
        object.__setattr__(stamped, f"{kind}_scale80", scale)
    return stamped


def _sample_to_gripper(value: float, **kwargs) -> torch.Tensor:
    clean_video, clean_action, action_mask, cond, cond_mask, uncond, uncond_mask, data_info = _make_inputs()
    g = torch.Generator().manual_seed(5)
    video_noise = torch.randn(clean_video.shape, generator=g).to(clean_video.dtype)
    action_noise = torch.randn(clean_action.shape, generator=g)
    video_target = torch.zeros_like(video_noise)
    action_target = torch.zeros_like(action_noise)
    action_target[..., [LEFT_GRIPPER, RIGHT_GRIPPER]] = value
    stub = _StubModel("to_target", cond, video_target, action_target, video_noise, action_noise)
    _, action = sample_policy(
        stub, clean_video, video_noise, action_noise, clean_action, action_mask,
        cond, cond_mask, uncond, uncond_mask, 1.0, data_info, STEPS, SHIFT, **kwargs,
    )
    return action[..., [LEFT_GRIPPER, RIGHT_GRIPPER]]


def test_the_sampler_clamp_follows_the_given_bounds():
    # Sana's default: the normalized gripper is clamped to [0, 1]
    assert torch.all(_sample_to_gripper(-0.8) == 0)
    # the model-domain image of raw [0, 1] under the 2026-09-20 statistics keeps the open half
    torch.testing.assert_close(_sample_to_gripper(-0.8, gripper_bounds=((-1.0, 1.0), (-1.0, 1.0))), torch.full((1, 4, 2), -0.8), atol=1e-5, rtol=0)
    assert torch.all(_sample_to_gripper(-1.7, gripper_bounds=((-1.0, 1.0), (-1.0, 1.0))) == -1.0)
    torch.testing.assert_close(_sample_to_gripper(-1.7, gripper_bounds=None), torch.full((1, 4, 2), -1.7), atol=1e-5, rtol=0)


def test_normalized_gripper_bounds_of_old_and_new_artifacts():
    packaged = load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256)
    assert normalized_gripper_bounds(packaged) == ((0.0, 1.0), (0.0, 1.0))        # identity grippers: Sana's clamp
    stamped = _stamped(packaged)
    assert normalized_gripper_bounds(stamped) == ((-1.0, 1.0), (-1.0, 1.0))
    # the bounds denormalize to raw closedness 0 / 1, i.e. fully open / fully closed
    rows = torch.zeros(2, 80)
    rows[0, [16, 45]], rows[1, [16, 45]] = -1.0, 1.0
    mask = torch.zeros(2, 80, dtype=torch.bool)
    mask[:, [16, 45]] = True
    raw = denormalize_action(rows, mask, stamped)
    assert raw[0, 16].item() == 0.0 and raw[1, 45].item() == 1.0
    assert opening_from_closedness(raw[0, 16].item()) == 1.0
    # the old clamp: normalized 0 is raw 0.5, an opening of at most 0.5
    half = torch.zeros(1, 80)
    assert opening_from_closedness(denormalize_action(half, mask[:1], stamped)[0, 16].item()) == 0.5


def test_the_session_passes_its_artifacts_bounds(monkeypatch):
    seen = []
    real = session_mod.sample_policy

    def spy(*args, **kwargs):
        seen.append(kwargs.get("gripper_bounds"))
        return real(*args, **kwargs)

    monkeypatch.setattr(session_mod, "sample_policy", spy)
    state80, mask80 = state80_from_obs(fake_state())
    session = make_session()
    session.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))
    new = make_session()
    new.normalization = _stamped(new.normalization)
    new.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))
    assert seen == [((0.0, 1.0), (0.0, 1.0)), ((-1.0, 1.0), (-1.0, 1.0))]
