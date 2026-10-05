"""CPU tests for the XPolicyLab adapter policy/SANA_MOT/model.py with a stubbed inference session (no weights)."""

from __future__ import annotations

import os
import shutil
import sys
import types
import warnings

import numpy as np
import pytest
import torch

_SANA_MOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(_SANA_MOT_DIR, "..", "..", ".."))
for _p in (_SANA_MOT_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# XPolicyLab.utils.load_file imports h5py at module import; the policy test env does not need HDF5.
try:
    import h5py  # noqa: F401
except ImportError:
    sys.modules["h5py"] = types.ModuleType("h5py")

import XPolicyLab.policy.SANA_MOT.model as adapter  # noqa: E402
from sana_wam_min.robodojo_io import ROBODOJO_NATIVE_ACTION_KEYS, ROBOT80_JOINT_SLOTS_12  # noqa: E402
from sana_wam_min.session import PredictResult  # noqa: E402

SNAPSHOT_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "config_multiview.yaml")
K = 24


class StubSession:
    """Stands in for MoTInferenceSession: hold-position rows plus a per-row ramp on the active slots."""

    from_paths_kwargs: dict = {}

    def __init__(self, ramp: float = 0.01):
        self.device = torch.device("cpu")
        self.model = None
        self.steps, self.cfg_scale, self.flow_shift = 10, 1.0, 3.5
        self.video_cfg_scale, self.action_cfg_scale = 1.0, 1.0
        self.joint_target_mode = "absolute"
        self.video_layout = "multiview"
        self.video_fps, self.video_frame_stride = None, 1
        self.load_report = {"context_layout": "context_embedder"}
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
    monkeypatch.setattr(adapter, "MoTInferenceSession", StubSession)
    StubSession.from_paths_kwargs = {}
    return StubSession


def base_cfg(ckpt_dir, **overrides) -> dict:
    cfg = {
        "policy_name": "SANA_MOT",
        "protocol": "ws",
        "host": "localhost",
        "port": 12345,
        "bench_name": "RoboDojo",
        "task_name": "stack_bowls",
        "ckpt_name": "mot_ckpt",
        "env_cfg_type": "arx_x5",
        "seed": 0,
        "action_type": "joint",
        "checkpoint_dir": str(ckpt_dir),
        "text_encoder_path": "stub-gemma",
        "vae_path": "stub-vae",
        "sampling_steps": 10,
        "cfg_scale": 6.0,
        "action_cfg_scale": 1.0,
        "flow_shift": 3.5,
        "device": "cpu",
        "joint_limit_mode": "clip",
        "joint_lower": [-3.14159] * 12,
        "joint_upper": [3.14159] * 12,
    }
    cfg.update(overrides)
    return cfg


def test_ready_and_action_contract(ckpt_dir, patched, capsys):
    model = adapter.Model(base_cfg(ckpt_dir))
    kwargs = patched.from_paths_kwargs
    assert kwargs["checkpoint_dir"] == str(ckpt_dir) and kwargs["steps"] == 10
    assert kwargs["cfg_scale"] == 6.0 and kwargs["action_cfg_scale"] == 1.0 and kwargs["video_cfg_scale"] is None
    ready = capsys.readouterr().out
    assert "[SANA_MOT] ready" in ready and "video_layout=multiview" in ready and "context_layout=context_embedder" in ready
    model.update_obs(fake_obs())
    actions = model.get_action()
    assert len(actions) == K
    for action in actions:
        assert tuple(action) == ROBODOJO_NATIVE_ACTION_KEYS
        assert action["left_arm_joint_state"].shape == (6,) and action["right_arm_joint_state"].shape == (6,)
        assert action["left_ee_joint_state"].shape == (1,) and action["right_ee_joint_state"].shape == (1,)
        assert all(v.dtype == np.float32 and v.flags["C_CONTIGUOUS"] for v in action.values())
    call = model.session.calls[0]
    assert call["instruction"] == "language instruction"
    assert [int(f[0, 0, 0]) for f in call["frames"]] == [10, 20, 30]           # view order head, left, right
    assert np.allclose(call["state80_raw"][:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert call["state80_raw"][16] == 0.0 and call["state80_raw"][45] == 1.0      # closedness = 1 - opening
    assert np.isclose(actions[0]["left_ee_joint_state"][0], 1.0) and np.isclose(actions[0]["right_ee_joint_state"][0], 0.0)
    assert np.isclose(actions[3]["left_arm_joint_state"][0], 0.1 + 0.03)


def test_batch_and_reset_seeding(ckpt_dir, patched):
    model = adapter.Model(base_cfg(ckpt_dir))
    model.update_obs_batch([fake_obs(0), fake_obs(1, instruction="other")])
    out = model.get_action_batch([1, 0])
    assert len(out) == 2 and len(out[0]) == K
    seeds = [c["seed"] for c in model.session.calls]
    assert seeds[0] != seeds[1]                                              # chunk index advances per inference
    model.reset()
    model.update_obs(fake_obs())
    model.get_action()
    assert model.session.calls[-1]["seed"] != seeds[0]                       # new episode, chunk 0
    assert model.session.calls[-1]["seed"] == adapter.derive_diffusion_seed(20260802, 0, 1, 0)
    with pytest.raises(ValueError):
        model.get_action_batch([7])


def test_n_action_steps_truncates_and_warns(ckpt_dir, patched):
    model = adapter.Model(base_cfg(ckpt_dir, n_action_steps=8))
    model.update_obs(fake_obs())
    assert len(model.get_action()) == 8 and model.action_chunk_size == K
    for raw in ("all", None, 0, "0"):
        assert adapter.Model._resolve_n_action_steps({"n_action_steps": raw}) is None
    assert adapter.Model._resolve_n_action_steps({"n_action_steps": "12"}) == 12
    with pytest.raises(ValueError):
        adapter.Model._resolve_n_action_steps({"n_action_steps": -1})
    big = adapter.Model(base_cfg(ckpt_dir, n_action_steps=100))
    big.update_obs(fake_obs())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert len(big.get_action()) == K
    assert any("exceeds" in str(w.message) for w in caught)


def test_joint_limits_clip(ckpt_dir, patched):
    model = adapter.Model(base_cfg(ckpt_dir, joint_lower=[-0.05] * 12, joint_upper=[0.05] * 12))
    model.update_obs(fake_obs())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actions = model.get_action()
    assert any("clipped" in str(w.message) for w in caught)
    assert np.all(np.abs(actions[-1]["left_arm_joint_state"]) <= 0.05 + 1e-6)
    strict = adapter.Model(base_cfg(ckpt_dir, joint_limit_mode="reject", joint_lower=[-0.05] * 12, joint_upper=[0.05] * 12))
    strict.update_obs(fake_obs())
    with pytest.raises(ValueError):
        strict.get_action()


def test_config_refusals(ckpt_dir, patched, monkeypatch):
    with pytest.raises(ValueError, match="action_type"):
        adapter.Model(base_cfg(ckpt_dir, action_type="ee"))
    with pytest.raises(ValueError, match="bfloat16"):
        adapter.Model(base_cfg(ckpt_dir, weight_dtype="float16"))
    monkeypatch.setattr(adapter, "get_robot_action_dim_info", lambda env: {"arm_dim": [7, 7], "ee_dim": [1, 1]})
    with pytest.raises(ValueError, match="dual ARX-X5"):
        adapter.Model(base_cfg(ckpt_dir))


def _action_mode_ckpt(tmp_path, ratio, **extra):
    import yaml

    ckpt = tmp_path / f"ckpt_{'_'.join(str(r) for r in ratio)}_{len(extra)}"
    (ckpt / "model").mkdir(parents=True)
    with open(SNAPSHOT_CONFIG) as handle:
        cfg = yaml.safe_load(handle)
    cfg["data"]["extra"]["action_mode_sample_ratio"] = list(ratio)
    cfg["model"].setdefault("extra", {}).update(extra)
    with open(ckpt / "config.yaml", "w") as handle:
        yaml.safe_dump(cfg, handle)
    return ckpt


def test_qwen_canonical_and_mixed_yamls_are_refused(tmp_path, patched):
    with pytest.raises(ValueError, match="qwen_canonical"):
        adapter.Model(base_cfg(_action_mode_ckpt(tmp_path, (1.0, 0.0, 0.0))))
    with pytest.raises(ValueError, match="names 2 action modes"):
        adapter.Model(base_cfg(_action_mode_ckpt(tmp_path, (0.0, 0.5, 0.5))))


def test_an_eef_only_mot_checkpoint_is_served_through_ee_poses(tmp_path, patched):
    """rwm/mot 2434e9199 made robot_base_eef EEF-only; a yaml with a post-2026-09-20 marker (model.extra.rope) is that
    contract: the state token and the action mask are the 20 EEF-pose + gripper slots, and only EE actions exist."""

    ckpt = _action_mode_ckpt(tmp_path, (0.0, 1.0, 0.0), rope="independent")
    with pytest.raises(ValueError, match="EEF-only"):
        adapter.Model(base_cfg(ckpt))
    model = adapter.Model(base_cfg(ckpt, action_type="ee", eef_pose_check=False))
    assert (model.action_mode, model.robot_base_eef_layout) == ("robot_base_eef", "eef_only")
    model.update_obs(fake_obs())
    actions = model.get_action()
    mask = model.session.calls[0]["state_mask80"]
    assert mask.sum() == 20 and not mask[list(adapter.ROBOT80_JOINT_SLOTS_12)].any() and mask[[16, 45]].all()
    assert sorted(actions[0]) == sorted(adapter.ROBODOJO_EE_ACTION_KEYS) and actions[0]["left_ee_pose"].shape == (7,)
    # a pre-2026-09-20 robot_base_eef yaml keeps the 32-slot meaning (joint actions allowed)
    old = adapter.Model(base_cfg(_action_mode_ckpt(tmp_path, (0.0, 1.0, 0.0)), eef_pose_check=False))
    assert old.robot_base_eef_layout == "full"
    with pytest.raises(ValueError, match="needs a robot_base_eef checkpoint"):
        adapter.Model(base_cfg(_action_mode_ckpt(tmp_path, (0.0, 0.0, 1.0)), action_type="ee"))


def test_frames_must_be_uint8_rgb(ckpt_dir, patched):
    model = adapter.Model(base_cfg(ckpt_dir))
    model.update_obs(fake_obs(dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        model.get_action()
    small = adapter.Model(base_cfg(ckpt_dir, strict_image_size=True))
    small.update_obs(fake_obs(shape=(240, 320, 3)))
    with pytest.raises(ValueError, match="strict_image_size"):
        small.get_action()
    lenient = adapter.Model(base_cfg(ckpt_dir))
    lenient.update_obs(fake_obs(shape=(240, 320, 3)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert len(lenient.get_action()) == K
    assert any("camera frames are not" in str(w.message) for w in caught)


def test_missing_instruction_falls_back_to_default(ckpt_dir, patched):
    model = adapter.Model(base_cfg(ckpt_dir, default_instruction="do the task"))
    obs = fake_obs()
    del obs["instruction"]
    model.update_obs(obs)
    model.get_action()
    assert model.session.calls[0]["instruction"] == "do the task"
    obs = fake_obs()
    obs["instruction"] = None
    obs["instructions"] = ["first choice", "second"]
    model.update_obs(obs)
    model.get_action()
    assert model.session.calls[1]["instruction"] == "first choice"


NSC_CANVAS_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "config_nsc_f25_openwam.yaml")


def test_video_layout_key_only_asserts_the_checkpoint(ckpt_dir, patched, capsys):
    """auto and a matching value pass; a mismatch fails BEFORE the session (the weights) is built; bad values fail."""

    adapter.Model(base_cfg(ckpt_dir))
    assert "video_layout=multiview (requested auto)" in capsys.readouterr().out
    adapter.Model(base_cfg(ckpt_dir, video_layout="MultiView"))
    assert "video_layout=multiview (requested multiview)" in capsys.readouterr().out
    patched.from_paths_kwargs = {}
    with pytest.raises(ValueError, match="does not match the checkpoint, which resolves to 'multiview'"):
        adapter.Model(base_cfg(ckpt_dir, video_layout="openwam_canvas"))
    assert patched.from_paths_kwargs == {}                                   # refused before any weight is read
    with pytest.raises(ValueError, match="video_layout must be one of"):
        adapter.Model(base_cfg(ckpt_dir, video_layout="three_view_strip"))


def test_pre_switch_canvas_checkpoint_asserts_openwam_canvas(tmp_path, patched, monkeypatch, capsys):
    """The NSC f25 canvas yaml has no video_layout key: auto and openwam_canvas serve it, multiview is refused early,
    and a session that resolved differently from the assertion is refused after the load as well."""

    ckpt = tmp_path / "nsc"
    (ckpt / "model").mkdir(parents=True)
    shutil.copy(NSC_CANVAS_CONFIG, ckpt / "config.yaml")

    class CanvasSession(StubSession):
        def __init__(self, ramp: float = 0.01):
            super().__init__(ramp)
            self.video_layout = "openwam_canvas"

    monkeypatch.setattr(adapter, "MoTInferenceSession", CanvasSession)
    for requested in ("auto", "openwam_canvas"):
        model = adapter.Model(base_cfg(ckpt, video_layout=requested))
        assert model.video_layout == "openwam_canvas"
        assert f"video_layout=openwam_canvas (requested {requested})" in capsys.readouterr().out
    CanvasSession.from_paths_kwargs = {}
    with pytest.raises(ValueError, match="does not match the checkpoint, which resolves to 'openwam_canvas'"):
        adapter.Model(base_cfg(ckpt, video_layout="multiview"))
    assert CanvasSession.from_paths_kwargs == {}
    monkeypatch.setattr(adapter, "MoTInferenceSession", StubSession)          # a session resolving to multiview
    with pytest.raises(ValueError, match="does not match the checkpoint, which resolves to 'multiview'"):
        adapter.Model(base_cfg(ckpt, video_layout="openwam_canvas"))


def test_trajectory_dump_records_every_tick_and_chunk(patched, ckpt_dir, tmp_path):
    dump = tmp_path / "traj"
    model = adapter.Model(base_cfg(ckpt_dir, trajectory_dump_dir=str(dump)))
    assert model.trajectory_dump_dir == dump and dump.is_dir()
    obs = fake_obs()
    model.reset()
    model.update_obs(obs)
    first = model.get_action()
    for _ in range(3):
        model.update_obs(obs)
    model.update_obs(obs)
    model.get_action()
    model.reset()
    z = np.load(dump / "ep0001.npz")
    assert z["slots"].tolist() == list(adapter.TRAJ_SLOTS)
    assert z["obs_state"].shape == (5, 14) and z["obs_step"].tolist() == [0, 1, 2, 3, 4]
    assert z["chunk_actions"].shape == (2, K, 14) and z["chunk_step"].tolist() == [0, 4]
    assert z["chunk_index"].tolist() == [0, 1] and z["chunk_executed"].tolist() == [K, K]
    np.testing.assert_allclose(z["chunk_actions"][0, :, 0:6], np.stack([a["left_arm_joint_state"] for a in first]), rtol=1e-6)
    measured, _ = adapter.state80_from_obs(obs["state"])
    np.testing.assert_array_equal(z["obs_state"][0], measured[list(adapter.TRAJ_SLOTS)])
    np.testing.assert_array_equal(z["chunk_anchor"][0], measured[list(adapter.TRAJ_SLOTS)])
    off = adapter.Model(base_cfg(ckpt_dir))
    assert off.trajectory_dump_dir is None
    off.update_obs(obs)
    off.get_action()
    assert off._traj_obs == [] and off._traj_chunks == []


def test_shipped_deploy_yml_defaults_to_video_only_cfg():
    """User default of 2026-09-25: the action stream is not guided (action_cfg_scale 1.0); cfg_scale guides the video
    stream only. A run that wants the historical single knob passes action_cfg_scale=null (or 6) explicitly."""

    import yaml as _yaml

    deploy = _yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy.yml")))
    assert deploy["action_cfg_scale"] == 1.0 and deploy["video_cfg_scale"] is None
