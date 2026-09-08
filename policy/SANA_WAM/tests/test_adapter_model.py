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

SNAPSHOT_CONFIG = (
    "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/xpolicylab_sana_wam_port_20260908/"
    "ckpt_snapshots/config.yaml"
)
K = 24


class StubSession:
    """Stands in for PolicyInferenceSession: hold-position rows plus a per-row ramp on the active slots."""

    from_paths_kwargs: dict = {}

    def __init__(self, ramp: float = 0.01):
        self.device = torch.device("cpu")
        self.model = None
        self.steps, self.cfg_scale, self.flow_shift = 50, 1.0, 3.5
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
    assert any("joint_lower" in str(w.message) for w in caught)
    assert model.joint_lower is None


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
