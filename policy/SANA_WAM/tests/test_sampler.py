"""CPU tests for sana_wam_min.sampler: schedule parity, loop invariants, CFG gating, determinism."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_SANA_WAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(_SANA_WAM_DIR, "..", "..", ".."))
for _p in (_SANA_WAM_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sana_wam_min import sampler as plain_sampler  # noqa: E402
from sana_wam_min.sampler import LEFT_GRIPPER, RIGHT_GRIPPER, make_scheduler, sample_policy  # noqa: E402

STEPS = 50
SHIFT = 3.5
N_TRAIN = 1000


def test_import_through_xpolicylab_namespace():
    import XPolicyLab.policy.SANA_WAM.sana_wam_min.sampler as ns_sampler

    assert ns_sampler.sample_policy.__doc__ == plain_sampler.sample_policy.__doc__
    assert ns_sampler.LEFT_GRIPPER == 16 and ns_sampler.RIGHT_GRIPPER == 45
    forbidden = ("dev.", "diffusion.", "sana.", "tqdm")
    with open(plain_sampler.__file__) as f:
        src = f.read()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert not any(tok in stripped for tok in forbidden), stripped


def _reference_schedule(steps: int, shift: float):
    """Spec section 3 math: float64 linspace in t, shift, cast to float32, terminal zero appended."""
    t_init = np.linspace(1, N_TRAIN, N_TRAIN, dtype=np.float32)[::-1]
    sig = t_init / N_TRAIN
    sig = shift * sig / (1 + (shift - 1) * sig)
    sigma_max, sigma_min = float(sig[0]), float(sig[-1])
    ts = np.linspace(sigma_max * N_TRAIN, sigma_min * N_TRAIN, steps)
    sigs = ts / N_TRAIN
    sigs = shift * sigs / (1 + (shift - 1) * sigs)
    sigmas = torch.from_numpy(sigs).to(torch.float32)
    timesteps = sigmas * N_TRAIN
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    return ts, timesteps, sigmas


def test_scheduler_timesteps_match_diffusers_and_spec_math():
    from diffusers import FlowMatchEulerDiscreteScheduler

    ref = FlowMatchEulerDiscreteScheduler(shift=SHIFT)
    ref.set_timesteps(STEPS, device="cpu")
    ours = make_scheduler(STEPS, SHIFT, torch.device("cpu"))
    assert ours.timesteps.dtype == torch.float32
    assert torch.equal(ours.timesteps, ref.timesteps)
    assert torch.equal(ours.sigmas, ref.sigmas)
    assert ours.config.num_train_timesteps == N_TRAIN

    ts, timesteps, sigmas = _reference_schedule(STEPS, SHIFT)
    assert torch.equal(ours.timesteps, timesteps)
    assert torch.equal(ours.sigmas, sigmas)
    assert len(ours.timesteps) == STEPS and len(ours.sigmas) == STEPS + 1
    print(
        f"\nlinspace(t) first/last = {ts[0]:.7f} -> {ts[-1]:.7f}; "
        f"timesteps first/last = {ours.timesteps[0].item():.4f} -> {ours.timesteps[-1].item():.4f}; "
        f"sigmas[-2:] = {ours.sigmas[-2].item():.6f}, {ours.sigmas[-1].item():.1f}"
    )
    assert ts[0] == 1000.0
    assert abs(ts[-1] - 3.4912718) < 1e-6
    assert ours.timesteps[0].item() == 1000.0
    expected_head = torch.tensor([1000.0, 994.1038, 988.0312, 981.7741, 975.3240])
    expected_tail = torch.tensor([244.9772, 194.4078, 139.2076, 78.7099, 12.1137])
    assert torch.allclose(ours.timesteps[:5], expected_head, atol=5e-4)
    assert torch.allclose(ours.timesteps[45:50], expected_tail, atol=5e-4)
    assert ours.sigmas[-1].item() == 0.0


class _StubModel:
    """Deterministic velocity oracle counting conditional / unconditional calls.

    ``mode='zero'`` returns zero velocities; ``mode='to_target'`` returns the constant
    velocity ``noise - target`` (training convention), so an exact Euler integration
    from sigma=1 to sigma=0 lands on ``target``.
    """

    def __init__(self, mode: str, cond_embeds, video_target=None, action_target=None,
                 video_noise=None, action_noise=None):
        self.mode = mode
        self.cond_embeds = cond_embeds
        self.video_target = video_target
        self.action_target = action_target
        self.video_noise = video_noise
        self.action_noise = action_noise
        self.cond_calls = 0
        self.uncond_calls = 0
        self.seen_timesteps: list[torch.Tensor] = []
        self.seen_action_timesteps: list[torch.Tensor] = []

    def __call__(self, x, timestep, y, mask=None, data_info=None):
        if y is self.cond_embeds:
            self.cond_calls += 1
        else:
            self.uncond_calls += 1
        assert timestep.shape == (x.shape[0], 1, x.shape[2])
        assert data_info["rwm_task"] == "policy"
        assert data_info["action_timestep"].shape == data_info["action80"].shape[:2]
        assert torch.all(data_info["action_timestep"] == timestep[0, 0, 1])
        assert torch.all(timestep[0, 0, 1:] == timestep[0, 0, 1])
        assert timestep[0, 0, 0].item() == 0.0
        self.seen_timesteps.append(timestep[0, 0, 1].clone())
        self.seen_action_timesteps.append(data_info["action_timestep"][0, 0].clone())
        action = data_info["action80"]
        if self.mode == "zero":
            return {"x": torch.zeros_like(x), "action_pred": torch.zeros_like(action)}
        return {
            "x": (self.video_noise - self.video_target).to(x.dtype),
            "action_pred": (self.action_noise - self.action_target).to(action.dtype),
        }


def _make_inputs(seed: int = 0, video_dtype=torch.bfloat16, k: int = 4):
    g = torch.Generator().manual_seed(seed)
    clean_video = torch.randn(1, 4, 3, 1, 6, generator=g).to(video_dtype)
    clean_action = torch.randn(1, k, 80, generator=g)
    action_mask = torch.zeros(1, k, 80, dtype=torch.bool)
    action_mask[..., list(range(0, 6)) + [16] + list(range(29, 35)) + [45]] = True
    cond = torch.randn(1, 4, 1, 8, 16, generator=g)
    cond_mask = torch.ones(1, 4, 1, 1, 8, dtype=torch.int64)
    uncond = torch.randn(1, 4, 1, 8, 16, generator=g)
    data_info = {"view_count": torch.tensor([3]), "model_fps": torch.tensor([25.0])}
    return clean_video, clean_action, action_mask, cond, cond_mask, uncond, cond_mask.clone(), data_info


def test_sample_policy_zero_velocity_invariants():
    clean_video, clean_action, action_mask, cond, cond_mask, uncond, uncond_mask, data_info = _make_inputs()
    g = torch.Generator().manual_seed(123)
    video_noise = torch.randn(clean_video.shape, generator=g).to(clean_video.dtype)
    action_noise = torch.randn(clean_action.shape, generator=g)
    stub = _StubModel("zero", cond)
    video, action = sample_policy(
        stub, clean_video, video_noise, action_noise, clean_action, action_mask,
        cond, cond_mask, uncond, uncond_mask, 1.0, data_info, STEPS, SHIFT,
    )
    assert video.shape == clean_video.shape and video.dtype == clean_video.dtype
    assert action.shape == clean_action.shape and action.dtype == clean_action.dtype
    assert torch.equal(video[:, :, :1], clean_video[:, :, :1])
    # Zero velocity leaves the noisy frames exactly at their noise values.
    assert torch.equal(video[:, :, 1:], video_noise[:, :, 1:])
    assert torch.all(action[~action_mask] == 0)
    # Zero velocity leaves the unmasked, non-gripper slots at their noise values.
    non_gripper = action_mask.clone()
    non_gripper[..., [LEFT_GRIPPER, RIGHT_GRIPPER]] = False
    assert torch.equal(action[non_gripper], action_noise[non_gripper])
    grip = action[..., [LEFT_GRIPPER, RIGHT_GRIPPER]]
    assert torch.all(grip >= 0) and torch.all(grip <= 1)
    assert torch.equal(grip, action_noise[..., [LEFT_GRIPPER, RIGHT_GRIPPER]].clamp(0, 1))
    assert stub.cond_calls == STEPS
    assert stub.uncond_calls == 0
    # Raw 0..1000 timesteps, frame 0 pinned at 0, action rows share the video timestep.
    sched = make_scheduler(STEPS, SHIFT, torch.device("cpu"))
    assert torch.equal(torch.stack(stub.seen_timesteps), sched.timesteps)
    assert torch.equal(torch.stack(stub.seen_action_timesteps), sched.timesteps)
    # Inputs are not mutated in place.
    assert torch.all(action_noise[~action_mask] != 0)


def test_sample_policy_constant_velocity_reaches_target():
    """v = noise - x0 integrated from sigma 1 to 0 must land on x0 (checks the -prediction sign)."""
    clean_video, clean_action, action_mask, cond, cond_mask, uncond, uncond_mask, data_info = _make_inputs(
        video_dtype=torch.float32
    )
    g = torch.Generator().manual_seed(7)
    video_noise = torch.randn(clean_video.shape, generator=g)
    action_noise = torch.randn(clean_action.shape, generator=g)
    video_target = torch.randn(clean_video.shape, generator=g)
    action_target = torch.rand(clean_action.shape, generator=g)  # in [0,1] so the gripper clamp is a no-op
    stub = _StubModel("to_target", cond, video_target, action_target, video_noise, action_noise)
    video, action = sample_policy(
        stub, clean_video, video_noise, action_noise, clean_action, action_mask,
        cond, cond_mask, uncond, uncond_mask, 1.0, data_info, STEPS, SHIFT,
    )
    assert torch.equal(video[:, :, :1], clean_video[:, :, :1])
    assert torch.allclose(video[:, :, 1:], video_target[:, :, 1:], atol=2e-5)
    assert torch.allclose(action[action_mask], action_target[action_mask], atol=2e-5)
    assert torch.all(action[~action_mask] == 0)


def test_reference_euler_matches_diffusers_per_token_step():
    """Spec 4.4 dependency-free Euler step reproduces one diffusers per-token step on the action stream."""
    sched = make_scheduler(STEPS, SHIFT, torch.device("cpu"))
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 4, 80, generator=g)
    v = torch.randn(1, 4, 80, generator=g)
    for i in (0, 17, STEPS - 1):
        sched_i = make_scheduler(STEPS, SHIFT, torch.device("cpu"))
        t = sched.timesteps[i]
        tok = t.expand(1, 4)
        out = plain_sampler._action_step(sched_i, v, t, x, tok)
        sig_cur = tok / N_TRAIN
        s = sched.sigmas.view(-1, 1, 1)
        lower = ((s < sig_cur[None] - 1e-6) * s).max(dim=0)[0]
        dt = (sig_cur - lower)[..., None]
        ref = x.float() - dt * v
        assert torch.equal(out, ref)
        assert torch.allclose(lower[0, 0], sched.sigmas[i + 1])


def test_cfg_gating_and_errors():
    clean_video, clean_action, action_mask, cond, cond_mask, uncond, uncond_mask, data_info = _make_inputs()
    g = torch.Generator().manual_seed(1)
    video_noise = torch.randn(clean_video.shape, generator=g).to(clean_video.dtype)
    action_noise = torch.randn(clean_action.shape, generator=g)
    stub = _StubModel("zero", cond)
    sample_policy(
        stub, clean_video, video_noise, action_noise, clean_action, action_mask,
        cond, cond_mask, uncond, uncond_mask, 2.0, data_info, 5, SHIFT,
    )
    assert stub.cond_calls == 5 and stub.uncond_calls == 5
    with pytest.raises(ValueError):
        sample_policy(
            stub, clean_video, video_noise, action_noise, clean_action, action_mask,
            cond, cond_mask, None, None, 2.0, data_info, 5, SHIFT,
        )
    with pytest.raises(ValueError):
        sample_policy(
            stub, clean_video, video_noise, action_noise, clean_action, action_mask.float(),
            cond, cond_mask, None, None, 1.0, data_info, 5, SHIFT,
        )
    with pytest.raises(ValueError):
        sample_policy(
            stub, clean_video, video_noise, action_noise, clean_action, action_mask,
            cond, cond_mask, None, None, 1.0, data_info, 0, SHIFT,
        )
    with pytest.raises(ValueError):
        sample_policy(
            stub, clean_video, video_noise[:, :, :2], action_noise, clean_action, action_mask,
            cond, cond_mask, None, None, 1.0, data_info, 5, SHIFT,
        )

    class _NaNModel(_StubModel):
        def __call__(self, x, timestep, y, mask=None, data_info=None):
            out = super().__call__(x, timestep, y, mask=mask, data_info=data_info)
            out["action_pred"] = out["action_pred"] + float("nan")
            return out

    with pytest.raises(FloatingPointError):
        sample_policy(
            _NaNModel("zero", cond), clean_video, video_noise, action_noise, clean_action, action_mask,
            cond, cond_mask, None, None, 1.0, data_info, 2, SHIFT,
        )


def test_determinism_and_noise_draw_order():
    clean_video, clean_action, action_mask, cond, cond_mask, uncond, uncond_mask, data_info = _make_inputs()

    def run(seed):
        g = torch.Generator().manual_seed(seed)
        return sample_policy(
            _StubModel("zero", cond), clean_video, None, None, clean_action, action_mask,
            cond, cond_mask, None, None, 1.0, data_info, 8, SHIFT, generator=g,
        )

    v1, a1 = run(42)
    v2, a2 = run(42)
    v3, a3 = run(43)
    assert torch.equal(v1, v2) and torch.equal(a1, a2)
    assert not torch.equal(v1, v3) or not torch.equal(a1, a3)

    # Explicit noise drawn video-first then action from the same seed equals the generator path.
    g = torch.Generator().manual_seed(42)
    video_noise = torch.randn(clean_video.shape, dtype=clean_video.dtype, generator=g)
    action_noise = torch.randn(clean_action.shape, dtype=clean_action.dtype, generator=g)
    v4, a4 = sample_policy(
        _StubModel("zero", cond), clean_video, video_noise, action_noise, clean_action, action_mask,
        cond, cond_mask, None, None, 1.0, data_info, 8, SHIFT,
    )
    assert torch.equal(v1, v4) and torch.equal(a1, a4)


class _SplitModel(_StubModel):
    """Constant velocities that DIFFER between the conditional and unconditional rows, so guidance is observable.

    Conditional: video 1, action 1; unconditional: video 0.5, action 0.5 -> a guided stream integrates
    ``0.5 + s * 0.5`` per unit sigma, an unguided one exactly ``1``.
    """

    def __call__(self, x, timestep, y, mask=None, data_info=None):
        out = super().__call__(x, timestep, y, mask=mask, data_info=data_info)
        level = 1.0 if y is self.cond_embeds else 0.5
        return {"x": torch.full_like(out["x"], level), "action_pred": torch.full_like(out["action_pred"], level)}


def _run_split(video_cfg_scale, action_cfg_scale, cfg_scale=1.0, steps=6):
    clean_video, clean_action, action_mask, cond, cond_mask, uncond, uncond_mask, data_info = _make_inputs(
        video_dtype=torch.float32
    )
    g = torch.Generator().manual_seed(11)
    video_noise = torch.randn(clean_video.shape, generator=g)
    action_noise = torch.randn(clean_action.shape, generator=g)
    stub = _SplitModel("zero", cond)
    video, action = sample_policy(
        stub, clean_video, video_noise, action_noise, clean_action, action_mask,
        cond, cond_mask, uncond, uncond_mask, cfg_scale, data_info, steps, SHIFT,
        video_cfg_scale=video_cfg_scale, action_cfg_scale=action_cfg_scale,
    )
    return video, action, stub


def test_per_stream_cfg_scales_guide_only_their_own_stream():
    # Both streams unguided (the historical cfg_scale=1 path): a single forward per step.
    v_plain, a_plain, stub = _run_split(None, None, cfg_scale=1.0)
    assert stub.cond_calls == 6 and stub.uncond_calls == 0
    # Both guided through the shared knob.
    v_both, a_both, stub = _run_split(None, None, cfg_scale=6.0)
    assert stub.cond_calls == 6 and stub.uncond_calls == 6
    assert not torch.equal(v_both[:, :, 1:], v_plain[:, :, 1:]) and not torch.equal(a_both, a_plain)
    # Video-only guidance: the unconditional forward still runs (one transformer for both streams), the video
    # matches the fully guided run and the action is BIT-IDENTICAL to the unguided one.
    v_vid, a_vid, stub = _run_split(None, 1.0, cfg_scale=6.0)
    assert stub.cond_calls == 6 and stub.uncond_calls == 6
    assert torch.equal(v_vid, v_both)
    assert torch.equal(a_vid, a_plain)
    # The same through the explicit video knob with cfg_scale left at 1.
    v_vid2, a_vid2, stub = _run_split(6.0, None, cfg_scale=1.0)
    assert stub.uncond_calls == 6 and torch.equal(v_vid2, v_both) and torch.equal(a_vid2, a_plain)
    # Action-only guidance: video unguided, action guided.
    v_act, a_act, stub = _run_split(1.0, 6.0, cfg_scale=1.0)
    assert stub.uncond_calls == 6
    assert torch.equal(v_act, v_plain) and torch.equal(a_act, a_both)
    # Different scales per stream are honoured independently (video at 2 is neither the 1 nor the 6 result).
    v_mix, a_mix, _ = _run_split(2.0, 6.0, cfg_scale=1.0)
    assert torch.equal(a_mix, a_both)
    assert not torch.equal(v_mix, v_plain) and not torch.equal(v_mix, v_both)
    # A guided action stream needs the unconditional rows even when cfg_scale itself is 1.
    clean_video, clean_action, action_mask, cond, cond_mask, _, _, data_info = _make_inputs()
    with pytest.raises(ValueError):
        sample_policy(
            _StubModel("zero", cond), clean_video, None, None, clean_action, action_mask,
            cond, cond_mask, None, None, 1.0, data_info, 2, SHIFT,
            generator=torch.Generator().manual_seed(0), action_cfg_scale=6.0,
        )
    with pytest.raises(ValueError):
        sample_policy(
            _StubModel("zero", cond), clean_video, None, None, clean_action, action_mask,
            cond, cond_mask, None, None, 1.0, data_info, 2, SHIFT,
            generator=torch.Generator().manual_seed(0), video_cfg_scale=float("nan"),
        )
