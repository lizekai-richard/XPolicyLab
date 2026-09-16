"""CPU tests for the XPolicyLab adapter policy/SANA_WAM/model.py with a stubbed inference session (no weights)."""

from __future__ import annotations

import os
import shutil
import sys
import types
import warnings

import numpy as np
import pytest
import torch

_SANA_WAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(_SANA_WAM_DIR, "..", "..", ".."))
for _p in (_SANA_WAM_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# XPolicyLab.utils.load_file imports h5py at module import; the policy test env does not need HDF5.
try:
    import h5py  # noqa: F401
except ImportError:
    sys.modules["h5py"] = types.ModuleType("h5py")

import XPolicyLab.policy.SANA_WAM.model as adapter  # noqa: E402
from sana_wam_min.robodojo_io import ROBODOJO_NATIVE_ACTION_KEYS, ROBOT80_JOINT_SLOTS_12  # noqa: E402
from sana_wam_min.session import PredictResult  # noqa: E402

# The resolved training yaml of the RoboDojo 320px joint-only line, as the trainer dumps it next to a
# checkpoint; the header of tests/fixtures/config.yaml states its provenance.
SNAPSHOT_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "config.yaml")
K = 24


class StubSession:
    """Stands in for PolicyInferenceSession: hold-position rows plus a per-row ramp on the active slots."""

    from_paths_kwargs: dict = {}

    def __init__(self, ramp: float = 0.01):
        self.device = torch.device("cpu")
        self.model = None
        self.steps, self.cfg_scale, self.flow_shift = 50, 1.0, 3.5
        self.video_cfg_scale, self.action_cfg_scale = 1.0, 1.0
        self.ramp = ramp
        self.calls: list[dict] = []

    @classmethod
    def from_paths(cls, **kwargs):
        cls.from_paths_kwargs = dict(kwargs)
        return cls()

    def predict(self, frames, state80_raw, state_mask80, instruction, generator=None):
        self.calls.append(
            {
                "frames": [np.array(f, copy=True) for f in frames],
                "state80_raw": np.array(state80_raw, copy=True),
                "state_mask80": np.array(state_mask80, copy=True),
                "instruction": instruction,
                "seed": None if generator is None else int(generator.initial_seed()),
            }
        )
        mask = torch.as_tensor(state_mask80).reshape(1, 80).expand(K, 80).contiguous()
        rows = torch.as_tensor(state80_raw, dtype=torch.float32).reshape(1, 80).repeat(K, 1)
        rows = rows + self.ramp * torch.arange(K, dtype=torch.float32)[:, None]
        rows = rows.masked_fill(~mask, 0.0)
        rows[:, [16, 45]] = rows[:, [16, 45]].clamp(0, 1)
        return PredictResult(
            action80_raw_absolute=rows, action80_model=rows.clone(), action_mask=mask, video_latent=torch.zeros(1), receipt={}
        )


def fake_obs(env_idx: int = 0, instruction: str = "language instruction", shape=(480, 640, 3), dtype=np.uint8) -> dict:
    cams = {}
    for i, cam in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist")):
        color = np.full(shape, 10 * (i + 1), dtype=np.uint8).astype(dtype)
        cams[cam] = {"color": color, "depth": np.zeros(shape, dtype=np.uint8), "shape": shape[:2]}
    return {
        "vision": cams,
        "instruction": instruction,
        "state": {
            "mobile": {"base_pose": [0.0] * 7, "base_twist": [0.0] * 6},
            "left_arm_joint_state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            "left_ee_joint_state": np.array([1.0], dtype=np.float32),
            "left_ee_pose": np.ones(7, dtype=np.float32),
            "right_arm_joint_state": np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], dtype=np.float32),
            "right_ee_joint_state": np.array([0.0], dtype=np.float32),
        },
        "additional_info": {"frequency": 30},
        "data_format_version": "v1.0",
        "env_idx": env_idx,
    }


@pytest.fixture
def ckpt_dir(tmp_path):
    ckpt = tmp_path / "ckpt"
    (ckpt / "model").mkdir(parents=True)
    shutil.copy(SNAPSHOT_CONFIG, ckpt / "config.yaml")
    return ckpt


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(adapter, "get_robot_action_dim_info", lambda env: {"arm_dim": [6, 6], "ee_dim": [1, 1]})
    monkeypatch.setattr(adapter, "PolicyInferenceSession", StubSession)
    StubSession.from_paths_kwargs = {}
    return StubSession


def base_cfg(ckpt_dir, **overrides) -> dict:
    cfg = {
        "policy_name": "SANA_WAM",
        "protocol": "ws",
        "host": "localhost",
        "port": 12345,
        "bench_name": "RoboDojo",
        "task_name": "stack_bowls",
        "ckpt_name": "dummy",
        "env_cfg_type": "arx_x5",
        "seed": 0,
        "action_type": "joint",
        "gpu_id": 0,
        "eval_batch": False,
        "checkpoint_dir": str(ckpt_dir),
        "text_encoder_path": "/stub/gemma",
        "vae_path": "/stub/vae",
        "device": "cpu",
        "weight_dtype": "bfloat16",
        "sampling_steps": 50,
        "cfg_scale": 1.0,
        "flow_shift": 3.5,
        "joint_limit_mode": "clip",
        "diffusion_seed_base": 20260802,
        "strict_image_size": False,
    }
    cfg.update(overrides)
    return cfg


def test_init_resolves_checkpoint_and_forwards_session_kwargs(patched, ckpt_dir):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = adapter.Model(base_cfg(ckpt_dir))
    assert model.ckpt_dir == ckpt_dir.resolve()
    kw = patched.from_paths_kwargs
    assert kw["checkpoint_dir"] == str(ckpt_dir.resolve()) and kw["steps"] == 50 and kw["cfg_scale"] == 1.0
    assert kw["flow_shift"] == 3.5 and kw["text_encoder_path"] == "/stub/gemma" and kw["vae_path"] == "/stub/vae"
    assert kw["device"] == torch.device("cpu") and kw["normalization_path"] is None
    assert kw["video_cfg_scale"] is None and kw["action_cfg_scale"] is None      # null inherits cfg_scale
    assert any("joint_lower" in str(w.message) for w in caught)
    assert model.joint_lower is None


def test_per_stream_cfg_knobs_are_forwarded_as_floats(patched, ckpt_dir):
    # deploy.yml overrides arrive as strings from the policy-server command line ("action_cfg_scale=1")
    adapter.Model(base_cfg(ckpt_dir, cfg_scale="6", action_cfg_scale="1", video_cfg_scale="null"))
    kw = patched.from_paths_kwargs
    assert kw["cfg_scale"] == 6.0 and kw["action_cfg_scale"] == 1.0 and kw["video_cfg_scale"] is None


def test_get_action_returns_24_native_dicts(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir))
    obs = fake_obs()
    model.update_obs(obs)
    actions = model.get_action()
    assert len(actions) == K
    for action in actions:
        assert tuple(action) == ROBODOJO_NATIVE_ACTION_KEYS
        assert action["left_arm_joint_state"].shape == (6,) and action["right_arm_joint_state"].shape == (6,)
        assert action["left_ee_joint_state"].shape == (1,) and action["right_ee_joint_state"].shape == (1,)
        assert all(v.dtype == np.float32 and v.ndim == 1 and v.flags["C_CONTIGUOUS"] for v in action.values())
    # Row 0 is the hold-position row: joints echo the state, grippers invert closedness back to opening.
    np.testing.assert_allclose(actions[0]["left_arm_joint_state"], obs["state"]["left_arm_joint_state"], atol=1e-6)
    np.testing.assert_allclose(actions[0]["right_arm_joint_state"], obs["state"]["right_arm_joint_state"], atol=1e-6)
    assert actions[0]["left_ee_joint_state"][0] == pytest.approx(1.0)
    assert actions[0]["right_ee_joint_state"][0] == pytest.approx(0.0)
    call = model.session.calls[0]
    assert call["instruction"] == "language instruction"
    assert call["state80_raw"][16] == pytest.approx(0.0) and call["state80_raw"][45] == pytest.approx(1.0)
    assert [f.shape for f in call["frames"]] == [(480, 640, 3)] * 3
    assert call["frames"][1][0, 0].tolist() == [20, 20, 20]


def test_frames_are_never_decoded_or_channel_swapped(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir))
    obs = fake_obs()
    obs["vision"]["cam_head"]["color"] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs["vision"]["cam_head"]["color"][..., 0] = 200
    obs["vision"]["cam_head"]["color"][..., 2] = 5
    model.update_obs(obs)
    model.get_action()
    frame = model.session.calls[0]["frames"][0]
    assert frame.dtype == np.uint8 and frame[0, 0].tolist() == [200, 0, 5]
    assert frame.flags["C_CONTIGUOUS"]


def test_non_uint8_frames_are_rejected_naming_the_camera(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir))
    for dtype in (np.float32, np.float64, np.uint16):
        obs = fake_obs()
        obs["vision"]["cam_left_wrist"]["color"] = np.full((480, 640, 3), 0.5, dtype=dtype)
        model.update_obs(obs)
        with pytest.raises(ValueError, match="cam_left_wrist"):
            model.get_action()
    assert model.session.calls == []


def test_strict_image_size_rejects_other_shapes(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir, strict_image_size=True))
    model.update_obs(fake_obs(shape=(240, 320, 3)))
    with pytest.raises(ValueError):
        model.get_action()
    lenient = adapter.Model(base_cfg(ckpt_dir))
    lenient.update_obs(fake_obs(shape=(240, 320, 3)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert len(lenient.get_action()) == K
    assert any("resized" in str(w.message) for w in caught)


def test_reset_and_counters_drive_distinct_seeds(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir))
    model.update_obs(fake_obs())
    model.get_action()
    model.update_obs(fake_obs())
    model.get_action()
    assert model._chunk_index == 2 and model._episode_index == 0
    model.reset()
    assert model._chunk_index == 0 and model._episode_index == 1 and model._obs == {} and model._order == []
    model.update_obs(fake_obs())
    model.get_action()
    seeds = [c["seed"] for c in model.session.calls]
    assert len(set(seeds)) == 3 and all(0 <= s < 2**53 for s in seeds)
    assert seeds[0] == adapter.derive_diffusion_seed(20260802, 0, 0, 0)
    assert seeds[2] == adapter.derive_diffusion_seed(20260802, 0, 1, 0)


def test_get_action_batch_loops_over_env_indices(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir))
    model.update_obs_batch([fake_obs(env_idx=3, instruction="a"), fake_obs(env_idx=7, instruction="b")])
    batch = model.get_action_batch([7, 3])
    assert len(batch) == 2 and all(len(chunk) == K for chunk in batch)
    assert [c["instruction"] for c in model.session.calls] == ["b", "a"]
    assert len(model.get_action_batch(np.array([3]))) == 1
    with pytest.raises(ValueError):
        model.get_action_batch([99])


def test_missing_instruction_falls_back_to_default(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir, default_instruction="do the task"))
    obs = fake_obs()
    del obs["instruction"]
    model.update_obs(obs)
    model.get_action()
    assert model.session.calls[0]["instruction"] == "do the task"
    obs = fake_obs()
    obs["instruction"] = None
    obs["instructions"] = ["Stack the bowls.", "other"]
    model.update_obs(obs)
    model.get_action()
    assert model.session.calls[1]["instruction"] == "Stack the bowls."


def test_joint_limits_clip_and_reject(patched, ckpt_dir):
    lower, upper = [-0.35] * 12, [0.35] * 12
    model = adapter.Model(base_cfg(ckpt_dir, joint_lower=lower, joint_upper=upper))
    model.update_obs(fake_obs())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actions = model.get_action()
    assert any("clipped" in str(w.message) for w in caught)
    left = np.stack([a["left_arm_joint_state"] for a in actions])
    right = np.stack([a["right_arm_joint_state"] for a in actions])
    assert left.max() <= 0.35 and right.min() >= -0.35 and left[0, 0] == pytest.approx(0.1)
    rejecting = adapter.Model(base_cfg(ckpt_dir, joint_lower=lower, joint_upper=upper, joint_limit_mode="reject"))
    rejecting.update_obs(fake_obs())
    with pytest.raises(ValueError):
        rejecting.get_action()
    with pytest.raises(ValueError):
        adapter.Model(base_cfg(ckpt_dir, joint_lower=[-1.0] * 6, joint_upper=[1.0] * 6))
    assert len(ROBOT80_JOINT_SLOTS_12) == 12


def test_rejects_non_joint_action_type_and_other_robots(patched, ckpt_dir, monkeypatch):
    with pytest.raises(ValueError):
        adapter.Model(base_cfg(ckpt_dir, action_type="ee"))
    with pytest.raises(ValueError):
        adapter.Model(base_cfg(ckpt_dir, env_cfg_type=None))
    with pytest.raises(ValueError):
        adapter.Model(base_cfg(ckpt_dir, weight_dtype="float16"))
    monkeypatch.setattr(adapter, "get_robot_action_dim_info", lambda env: {"arm_dim": [7], "ee_dim": [1]})
    with pytest.raises(ValueError):
        adapter.Model(base_cfg(ckpt_dir))


def test_missing_checkpoint_dir_fails_loudly(patched, tmp_path):
    with pytest.raises(FileNotFoundError):
        adapter.Model(base_cfg(tmp_path / "nowhere"))


def test_module_hygiene():
    with open(adapter.__file__) as f:
        source = f.read()
    # Tokens are split so this test file itself stays clean under the adapter-check greps.
    forbidden = ("cv2", "imdecode", "COLOR_" + "BGR2RGB", "COLOR_" + "RGB2BGR", "parents" + "[1]", "parents" + "[3]",
                 "_robot_info" + ".json", "env_" + "cfg/")
    for token in forbidden:
        assert token not in source, token
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert not any(tok in stripped for tok in ("dev.", "diffusion.", "sana.", "omegaconf", "pyrallis")), stripped


# -- n_action_steps: partial-chunk execution / replanning -----------------------------------------


def test_n_action_steps_returns_leading_targets_of_the_chunk(patched, ckpt_dir):
    full = adapter.Model(base_cfg(ckpt_dir))
    partial = adapter.Model(base_cfg(ckpt_dir, n_action_steps=8))
    assert partial.n_action_steps == 8 and full.n_action_steps is None
    obs = fake_obs()
    full.update_obs(obs)
    partial.update_obs(obs)
    full_actions = full.get_action()
    partial_actions = partial.get_action()
    assert len(full_actions) == K and len(partial_actions) == 8
    assert partial.action_chunk_size == K
    # The returned targets are exactly the first 8 of the same predicted chunk (same seed, same stub ramp).
    for want, got in zip(full_actions[:8], partial_actions, strict=True):
        for key in ROBODOJO_NATIVE_ACTION_KEYS:
            np.testing.assert_array_equal(got[key], want[key])
    # Every call is one inference: the chunk counter (and thus the diffusion seed) advances per call.
    partial.update_obs(obs)
    partial.get_action()
    assert partial._chunk_index == 2 and len(partial.session.calls) == 2


def test_n_action_steps_batch_path_truncates_every_env(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir, n_action_steps="4"))  # --overrides may deliver strings
    model.update_obs_batch([fake_obs(0), fake_obs(1)])
    batch = model.get_action_batch([0, 1])
    assert [len(chunk) for chunk in batch] == [4, 4]


@pytest.mark.parametrize("value", [None, 0, "0", "null", "all", ""])
def test_n_action_steps_unset_keeps_the_whole_chunk(patched, ckpt_dir, value):
    model = adapter.Model(base_cfg(ckpt_dir, n_action_steps=value))
    assert model.n_action_steps is None
    model.update_obs(fake_obs())
    assert len(model.get_action()) == K


@pytest.mark.parametrize("value", [-1, "abc", 2.5, True])
def test_n_action_steps_rejects_invalid_values(patched, ckpt_dir, value):
    with pytest.raises(ValueError, match="n_action_steps"):
        adapter.Model(base_cfg(ckpt_dir, n_action_steps=value))


def test_n_action_steps_above_chunk_warns_once_and_executes_everything(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir, n_action_steps=K + 10))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.update_obs(fake_obs())
        assert len(model.get_action()) == K
        model.update_obs(fake_obs())
        assert len(model.get_action()) == K
    assert sum("exceeds the predicted chunk" in str(w.message) for w in caught) == 1


# -- anchor_source: measured joints vs the last executed command as state input + delta anchor --------------------


def _joint12(row):
    return np.asarray(row)[list(ROBOT80_JOINT_SLOTS_12)]


def _run_two_chunks(model, obs):
    model.update_obs(obs)
    first = model.get_action()
    model.update_obs(obs)          # the evaluator re-observes (a measured, lagging state); the anchor must not depend on it
    second = model.get_action()
    return first, second


def test_anchor_source_defaults_to_measured_and_keeps_the_historical_inputs(patched, ckpt_dir, capsys):
    model = adapter.Model(base_cfg(ckpt_dir))
    assert model.anchor_source == "measured" and model.anchor_clamp_rad is None
    obs = fake_obs()
    measured, _ = adapter.state80_from_obs(obs["state"])
    _run_two_chunks(model, obs)
    for call in model.session.calls:
        np.testing.assert_array_equal(call["state80_raw"], measured)
    assert "anchor_source=measured" in capsys.readouterr().out


def test_last_command_anchors_the_next_chunk_on_the_last_executed_target(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir, anchor_source="last_command"))
    obs = fake_obs()
    measured, _ = adapter.state80_from_obs(obs["state"])
    first, _ = _run_two_chunks(model, obs)
    # first chunk of an episode: no previous command -> the measured state
    np.testing.assert_array_equal(model.session.calls[0]["state80_raw"], measured)
    anchor = model.session.calls[1]["state80_raw"]
    last = first[-1]                                       # the last target the loop executed before re-observing
    np.testing.assert_allclose(anchor[0:6], last["left_arm_joint_state"], rtol=1e-6)
    np.testing.assert_allclose(anchor[29:35], last["right_arm_joint_state"], rtol=1e-6)
    np.testing.assert_allclose(anchor[16], 1.0 - last["left_ee_joint_state"][0], atol=1e-6)   # closedness of the last command
    np.testing.assert_allclose(anchor[45], 1.0 - last["right_ee_joint_state"][0], atol=1e-6)
    other = [i for i in range(80) if i not in ROBOT80_JOINT_SLOTS_12 and i not in (16, 45)]
    np.testing.assert_array_equal(anchor[other], measured[other])       # every other slot stays as measured
    assert not np.allclose(_joint12(anchor), _joint12(measured))        # the stub ramp moved the joints


def test_last_command_uses_row_n_minus_1_under_n_action_steps_and_forgets_on_reset(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir, anchor_source="last_command", n_action_steps=8))
    obs = fake_obs()
    measured, _ = adapter.state80_from_obs(obs["state"])
    first, _ = _run_two_chunks(model, obs)
    assert len(first) == 8
    anchor = model.session.calls[1]["state80_raw"]
    np.testing.assert_allclose(anchor[0:6], first[7]["left_arm_joint_state"], rtol=1e-6)     # row n-1 of the chunk, not row 23
    model.reset()
    model.update_obs(obs)
    model.get_action()
    np.testing.assert_array_equal(model.session.calls[2]["state80_raw"], measured)


def test_last_command_clamped_limits_the_anchor_shift(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir, anchor_source="last_command_clamped", anchor_clamp_rad="0.05"))
    assert model.anchor_clamp_rad == 0.05
    model.session.ramp = 0.02                    # last row = measured + 0.46 rad on every joint: far beyond the clamp
    obs = fake_obs()
    measured, _ = adapter.state80_from_obs(obs["state"])
    _run_two_chunks(model, obs)
    anchor = model.session.calls[1]["state80_raw"]
    np.testing.assert_allclose(_joint12(anchor), _joint12(measured) + 0.05, rtol=1e-5)


def test_last_command_batch_path_keeps_one_history_per_env(patched, ckpt_dir):
    model = adapter.Model(base_cfg(ckpt_dir, anchor_source="last_command"))
    obs0, obs1 = fake_obs(0), fake_obs(1)
    obs1["state"]["left_arm_joint_state"] = np.array([0.5] * 6, dtype=np.float32)
    model.update_obs_batch([obs0, obs1])
    batch = model.get_action_batch([0, 1])
    model.update_obs_batch([obs0, obs1])
    model.get_action_batch([0, 1])
    calls = model.session.calls
    np.testing.assert_allclose(calls[2]["state80_raw"][0:6], batch[0][-1]["left_arm_joint_state"], rtol=1e-6)
    np.testing.assert_allclose(calls[3]["state80_raw"][0:6], batch[1][-1]["left_arm_joint_state"], rtol=1e-6)


@pytest.mark.parametrize("value", ["foo", 1, True])
def test_anchor_source_rejects_unknown_values(patched, ckpt_dir, value):
    with pytest.raises(ValueError, match="anchor_source"):
        adapter.Model(base_cfg(ckpt_dir, anchor_source=value))


@pytest.mark.parametrize("value", [None, "", "null", "measured"])
def test_anchor_source_unset_means_measured(patched, ckpt_dir, value):
    assert adapter.Model(base_cfg(ckpt_dir, anchor_source=value)).anchor_source == "measured"


@pytest.mark.parametrize("value", [0, -0.1, "abc"])
def test_anchor_clamp_rad_must_be_positive(patched, ckpt_dir, value):
    with pytest.raises(ValueError):
        adapter.Model(base_cfg(ckpt_dir, anchor_source="last_command_clamped", anchor_clamp_rad=value))


def test_anchor_lag_is_logged_per_chunk_in_every_mode(patched, ckpt_dir, capsys):
    model = adapter.Model(base_cfg(ckpt_dir))
    _run_two_chunks(model, fake_obs())
    lines = [line for line in capsys.readouterr().out.splitlines() if "[SANA_WAM] anchor ep" in line]
    assert len(lines) == 1 and "src=measured" in lines[0]
    left_max = float(lines[0].split("left max ")[1].split()[0])
    assert abs(left_max - 0.23) < 1e-3       # stub ramp 0.01 x row 23


# -- robot_base_eef checkpoints: EEF state slots, action_type ee -------------------------------------------------

import yaml  # noqa: E402

from sana_wam_min import eef as eef_mod  # noqa: E402


def _make_eef_checkpoint(ckpt_dir, ratio=(0.0, 1.0, 0.0)):
    """Turn the fixture checkpoint into a robot_base_eef line by rewriting the training yaml's action-mode ratio."""
    cfg_path = ckpt_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["data"]["extra"]["action_mode_sample_ratio"] = list(ratio)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return ckpt_dir


def test_state_profile_auto_detects_the_eef_line_and_fills_the_eef_slots(patched, ckpt_dir):
    joint_model = adapter.Model(base_cfg(ckpt_dir))
    assert (joint_model.state_profile, joint_model.include_eef) == ("joint_only", False) and joint_model.kinematics is None
    model = adapter.Model(base_cfg(_make_eef_checkpoint(ckpt_dir), eef_pose_check=False))
    assert (model.state_profile, model.include_eef) == ("robot_base_eef", True) and model.kinematics is not None
    obs = fake_obs()
    model.update_obs(obs)
    actions = model.get_action()
    call = model.session.calls[0]
    assert call["state_mask80"].sum() == 32 and call["state_mask80"][list(eef_mod.EEF_SLOTS)].all()
    expected_pos, expected_rot = eef_mod.e_pose_slots(model.kinematics.e_in_base(obs["state"]["left_arm_joint_state"]))
    np.testing.assert_allclose(call["state80_raw"][7:10], expected_pos, rtol=1e-6)
    np.testing.assert_allclose(call["state80_raw"][10:16], expected_rot, rtol=1e-6)
    expected_pos_r, _ = eef_mod.e_pose_slots(model.kinematics.e_in_base(obs["state"]["right_arm_joint_state"]))
    np.testing.assert_allclose(call["state80_raw"][36:39], expected_pos_r, rtol=1e-6)
    # joint actions by default, even for an EEF checkpoint (its predicted EEF slots are reconstructed, not emitted)
    assert len(actions) == K and sorted(actions[0]) == sorted(ROBODOJO_NATIVE_ACTION_KEYS)


def test_action_type_ee_emits_world_link6_poses_and_gripper_openings(patched, ckpt_dir):
    model = adapter.Model(base_cfg(_make_eef_checkpoint(ckpt_dir), action_type="ee", eef_pose_check=False))
    model.session.ramp = 0.0          # the stub echoes the anchor: EEF "deltas" == anchor values, joints unchanged
    obs = fake_obs()
    model.update_obs(obs)
    actions = model.get_action()
    assert len(actions) == K and sorted(actions[0]) == sorted(adapter.ROBODOJO_EE_ACTION_KEYS)
    anchor = model.session.calls[0]["state80_raw"]
    row = anchor.copy()
    for _, _, pos, rot, _ in eef_mod.ARM_FIELDS:       # anchor + delta with delta == anchor
        row[pos] = 2 * anchor[pos]
        r = eef_mod.rot6d_to_matrix(anchor[rot])
        row[rot] = eef_mod.matrix_to_rot6d(r @ r)
    for side in ("left", "right"):
        expected = eef_mod.world_link6_pose_from_slots(row, side, model.world_from_base[side])
        np.testing.assert_allclose(actions[0][f"{side}_ee_pose"], expected, atol=1e-5)
        assert actions[0][f"{side}_ee_pose"].dtype == np.float32 and actions[0][f"{side}_ee_pose"].shape == (7,)
    np.testing.assert_allclose(actions[0]["left_ee_joint_state"], [1.0 - anchor[16]], atol=1e-6)
    np.testing.assert_allclose(actions[0]["right_ee_joint_state"], [1.0 - anchor[45]], atol=1e-6)
    # the last command remembers the absolute EEF row
    np.testing.assert_allclose(model._last_command80[0][7:10], row[7:10], atol=1e-6)


def test_action_type_ee_requires_an_eef_checkpoint_and_a_measured_anchor(patched, ckpt_dir):
    with pytest.raises(ValueError, match="robot_base_eef checkpoint"):
        adapter.Model(base_cfg(ckpt_dir, action_type="ee"))
    _make_eef_checkpoint(ckpt_dir)
    with pytest.raises(ValueError, match="anchor_source must be 'measured'"):
        adapter.Model(base_cfg(ckpt_dir, action_type="ee", anchor_source="last_command"))
    with pytest.raises(ValueError, match="action_type"):
        adapter.Model(base_cfg(ckpt_dir, action_type="cartesian"))


def test_last_command_anchor_recomputes_the_eef_slots_from_the_commanded_joints(patched, ckpt_dir):
    model = adapter.Model(base_cfg(_make_eef_checkpoint(ckpt_dir), anchor_source="last_command", eef_pose_check=False))
    obs = fake_obs()
    model.update_obs(obs)
    first = model.get_action()
    model.update_obs(obs)
    model.get_action()
    anchor2 = model.session.calls[1]["state80_raw"]
    np.testing.assert_allclose(anchor2[0:6], first[-1]["left_arm_joint_state"], rtol=1e-6)
    expected_pos, expected_rot = eef_mod.e_pose_slots(model.kinematics.e_in_base(first[-1]["left_arm_joint_state"]))
    np.testing.assert_allclose(anchor2[7:10], expected_pos, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(anchor2[10:16], expected_rot, rtol=1e-5, atol=1e-6)


def test_eef_pose_check_runs_once_per_episode_and_warns_on_inconsistent_observations(patched, ckpt_dir, capsys):
    model = adapter.Model(base_cfg(_make_eef_checkpoint(ckpt_dir)))
    obs = fake_obs()                       # left_ee_pose = ones(7): not the FK of the joints
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.update_obs(obs)
        model.get_action()
        model.update_obs(obs)
        model.get_action()
    assert any("disagrees with FK" in str(w.message) for w in caught)
    assert sum("eef check ep" in line for line in capsys.readouterr().out.splitlines()) == 1
    consistent = eef_mod.world_link6_pose_from_slots(model.session.calls[0]["state80_raw"], "left", model.world_from_base["left"])
    obs["state"]["left_ee_pose"] = np.asarray(consistent, dtype=np.float32)
    model.reset()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.update_obs(obs)
        model.get_action()
    assert not any("disagrees with FK" in str(w.message) for w in caught)
    assert "eef check ep" in capsys.readouterr().out


def test_state_profile_override_and_unsupported_modes(patched, ckpt_dir):
    forced = adapter.Model(base_cfg(ckpt_dir, state_profile="robot_base_eef", eef_pose_check=False))
    assert forced.include_eef and forced.kinematics is not None
    with pytest.raises(ValueError, match="state_profile"):
        adapter.Model(base_cfg(ckpt_dir, state_profile="camera"))
    _make_eef_checkpoint(ckpt_dir, ratio=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="qwen_canonical"):
        adapter.Model(base_cfg(ckpt_dir))


# -- normalization contract log ------------------------------------------------------------------------------------


def _fake_normalization(action_active, state_active=None):
    from sana_wam_min.robot80 import Normalization

    action_mask = np.zeros(80, dtype=bool)
    action_mask[list(action_active)] = True
    state_mask = np.zeros(80, dtype=bool)
    state_mask[list(action_active if state_active is None else state_active)] = True
    return Normalization(
        state_center80=np.zeros(80, np.float32), state_scale80=np.ones(80, np.float32), state_normalization_mask80=state_mask,
        action_center80=np.zeros(80, np.float32), action_scale80=np.ones(80, np.float32), action_normalization_mask80=action_mask,
        joint_target_mode="anchor_delta", action_representation="robot_base_qwen", sha256="abc", action_mode="robot_base_eef",
        num_frames=25, model_fps=25, source_path="/x/normalization.json",
    )


def _log_with(norm, capsys):
    class _Session:
        normalization = norm

    class _Holder:
        session = _Session()

    adapter.Model._log_normalization_contract(_Holder())
    return capsys.readouterr().out


def test_normalization_log_reports_the_arx_x5_joint_only_layout_as_ok(capsys):
    # the ARX-X5 artifact: 6 joints per arm (0-5 / 29-34) + the shared EEF-position slots, grippers identity, slots 6/35 unused
    out = _log_with(_fake_normalization([0, 1, 2, 3, 4, 5, 7, 8, 9, 29, 30, 31, 32, 33, 34, 36, 37, 38]), capsys)
    assert "[SANA_WAM] normalization: path=/x/normalization.json sha256=abc action_mode=robot_base_eef" in out
    assert "joint_target_mode=anchor_delta" in out and "action_slots_normalized=[0, 1, 2, 3, 4, 5, 7, 8, 9, 29" in out
    assert "normalization layout OK: 6 joints per arm" in out and "WARNING" not in out


def test_normalization_log_warns_on_a_seven_joint_or_gripper_normalizing_artifact(capsys):
    out = _log_with(_fake_normalization(list(range(0, 7)) + [16] + list(range(29, 36)) + [45]), capsys)
    assert "WARNING normalization layout does not match" in out
    assert "7th-joint slots [6, 35]" in out and "gripper slots [16, 45] are normalized" in out
    out = _log_with(_fake_normalization([0, 1, 2, 3, 4, 29, 30, 31, 32, 33]), capsys)          # a joint slot without statistics
    assert "joint slots without action statistics: [5, 34]" in out


def test_normalization_log_is_silent_for_session_doubles_without_an_artifact(capsys):
    class _Holder:
        session = object()

    adapter.Model._log_normalization_contract(_Holder())
    assert capsys.readouterr().out == ""


# -- OpenWAM canvas checkpoints: visual_layout ------------------------------------------------------------------------

CANVAS_SNAPSHOT_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "config_openwam_canvas.yaml")


def _make_canvas_checkpoint(ckpt_dir):
    """Turn the fixture checkpoint into a rwm/openwam canvas line (its resolved training yaml)."""
    shutil.copy(CANVAS_SNAPSHOT_CONFIG, ckpt_dir / "config.yaml")
    return ckpt_dir


def test_visual_layout_auto_detects_the_canvas_line(patched, ckpt_dir, capsys):
    three = adapter.Model(base_cfg(ckpt_dir))
    assert three.visual_layout == "three_view_strip" and three.state_as_cross_attention is False
    assert "visual_layout=three_view_strip" in capsys.readouterr().out
    model = adapter.Model(base_cfg(_make_canvas_checkpoint(ckpt_dir)))
    out = capsys.readouterr().out
    assert model.visual_layout == "openwam_canvas" and model.state_as_cross_attention is False
    assert (model.state_profile, model.include_eef) == ("joint_only", False)
    assert "visual_layout=openwam_canvas state_as_cross_attention=False" in out
    assert "visual layout openwam_canvas: the 3 cameras are stretched into one 384x320" in out
    obs = fake_obs()
    model.update_obs(obs)
    actions = model.get_action()
    assert len(actions) == K and sorted(actions[0]) == sorted(ROBODOJO_NATIVE_ACTION_KEYS)
    call = model.session.calls[0]
    assert [f.shape for f in call["frames"]] == [(480, 640, 3)] * 3  # the raw cameras: the session composites them
    assert all(f.dtype == np.uint8 for f in call["frames"]) and call["frames"][2][0, 0].tolist() == [30, 30, 30]


def test_visual_layout_explicit_values_must_match_the_checkpoint(patched, ckpt_dir):
    assert adapter.Model(base_cfg(ckpt_dir, visual_layout="three_view_strip")).visual_layout == "three_view_strip"
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        adapter.Model(base_cfg(ckpt_dir, visual_layout="openwam_canvas"))
    with pytest.raises(ValueError, match="visual_layout must be one of"):
        adapter.Model(base_cfg(ckpt_dir, visual_layout="mosaic"))
    _make_canvas_checkpoint(ckpt_dir)
    assert adapter.Model(base_cfg(ckpt_dir, visual_layout="openwam_canvas")).visual_layout == "openwam_canvas"
    assert adapter.Model(base_cfg(ckpt_dir, visual_layout=None)).visual_layout == "openwam_canvas"
    assert adapter.Model(base_cfg(ckpt_dir, visual_layout="null")).visual_layout == "openwam_canvas"
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        adapter.Model(base_cfg(ckpt_dir, visual_layout="three_view_strip"))


def test_canvas_checkpoint_is_joint_only_and_refuses_ee_actions(patched, ckpt_dir):
    _make_canvas_checkpoint(ckpt_dir)
    with pytest.raises(ValueError, match="robot_base_eef checkpoint"):
        adapter.Model(base_cfg(ckpt_dir, action_type="ee"))


def test_canvas_size_warning_names_the_canvas_slots(patched, ckpt_dir):
    model = adapter.Model(base_cfg(_make_canvas_checkpoint(ckpt_dir)))
    model.update_obs(fake_obs(shape=(240, 320, 3)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert len(model.get_action()) == K
    assert any("384x320 OpenWAM canvas" in str(w.message) for w in caught)


def test_state_as_cross_attention_flag_is_read_from_the_canvas_yaml(patched, ckpt_dir):
    _make_canvas_checkpoint(ckpt_dir)
    cfg_path = ckpt_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["model"]["extra"]["state_as_cross_attention"] = True
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    model = adapter.Model(base_cfg(ckpt_dir))
    assert model.state_as_cross_attention is True and model.visual_layout == "openwam_canvas"
    # the flag is undefined for the three-view line
    cfg = yaml.safe_load(SNAPSHOT_CONFIG and open(SNAPSHOT_CONFIG).read())
    cfg["model"]["extra"]["state_as_cross_attention"] = True
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="only defined for the OpenWAM canvas"):
        adapter.Model(base_cfg(ckpt_dir))
