"""The XPolicyLab adapter's decision cycle against a fake runtime (CPU, no weights): the RoboDojo call order of
``policy/<POLICY>/deploy.py`` (update_obs -> get_action, then update_obs after every executed action but the last, whose
observation arrives right before the next get_action), one canvas per tick, the ticks every decision's prefix encode
reads (``sparse_memory.prefix_rows`` for both memory types) and the world-model reference encode (the previous
decision's ``sequence_rows``), the double reset of every RoboDojo episode, and the refusals."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import _paths

import XPolicyLab.policy.SANA_WAM_SM.model as adapter
from sana_wam_sm import sparse_memory as sm
from sana_wam_sm.contract import bidirectional_view
from sana_wam_min.eef import ArxX5Kinematics, root_transforms
from sana_wam_min.robot80 import normalized_gripper_bounds
from sana_wam_min.session import load_checked_normalization

C = 32
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
FIXED = {"memory_type": "fixed", "memory_frames": 16}
STRIDED = {"memory_type": "strided", "memory_stride": 25}


class FakeSession:
    def __init__(self, log):
        self.log = log
        self.decision = 0
        self.deploy_form = "cached"

    def decide(self, prefix_latents, *, anchor_state, anchor_state_mask, action_mask, generator, gripper_bounds):
        self.log.append(("decide", self.decision, int(prefix_latents.shape[2])))
        action = torch.zeros(1, C, 80)
        action[..., 10] = 1.0   # rot6d column 1 = x axis
        action[..., 14] = 1.0   # column 2 = y axis
        action[..., 39] = 1.0
        action[..., 43] = 1.0
        action[..., 16] = 0.25
        self.decision += 1
        return action.masked_fill(~action_mask, 0), torch.full((1, 4, 1, 1, 1), 1000.0 + self.decision)


class FakeRuntime:
    """Canvases carry the tick of their frame (the frames are filled with tick % 256); an encode of 1 + 8k canvases
    returns 1 + k latents."""

    def __init__(self, log, memory, trained=8):
        self.log = log
        self.device = torch.device("cpu")
        self.contract = SimpleNamespace(actions_per_chunk=C, memory=memory, trained_memory_latents=trained)

    def new_session(self, instruction):
        self.log.append(("new_session", instruction))
        return FakeSession(self.log)

    def canvas(self, frames):
        return torch.full((3, 1, 1), int(frames[0][0, 0, 0]), dtype=torch.uint8)

    def encode(self, canvases):
        ticks = [int(c.flatten()[0]) for c in canvases]
        self.log.append(("encode", ticks))
        assert (len(ticks) - 1) % 8 == 0
        return torch.full((1, 4, 1 + (len(ticks) - 1) // 8, 1, 1), float(sum(ticks)))


def _obs(tick: int, instruction: str = "cover the blocks") -> dict:
    frame = np.full((8, 8, 3), tick % 256, dtype=np.uint8)
    frame.setflags(write=False)  # the ws codec hands read-only arrays
    joints = np.linspace(-0.3, 0.3, 6) + 0.001 * tick
    return {
        "vision": {cam: {"color": frame} for cam in CAMERAS},
        "state": {
            "left_arm_joint_state": joints, "right_arm_joint_state": -joints,
            "left_ee_joint_state": np.array([0.8]), "right_ee_joint_state": np.array([0.6]),
        },
        "instruction": instruction,
    }


def _model(memory, trained=8, log_world_model_error=True):
    with open(os.path.join(_paths.FIXTURES, "recipe_sm_fixed_m16_3fa3e9689.yaml")) as handle:
        view = bidirectional_view(yaml.safe_load(handle))
    normalization = load_checked_normalization(
        "/nonexistent", view, os.path.join(_paths.FIXTURES, "normalization_causal_f33_1fe3b7e7.json")
    )
    log: list = []
    m = adapter.Model.__new__(adapter.Model)
    m.runtime = FakeRuntime(log, memory, trained)
    m.contract = m.runtime.contract
    m.memory = memory
    m.normalization = normalization
    m.gripper_bounds = normalized_gripper_bounds(normalization)
    m.kinematics = ArxX5Kinematics(None)
    m.world_from_base = root_transforms(None)
    m.chunk_actions = C
    m.strict_image_size = False
    m.default_instruction = "follow the instruction"
    m.eef_pose_check = False
    m.max_chunks_per_episode = None
    m.log_world_model_error = log_world_model_error
    m.diffusion_seed_base, m.eval_seed = 1, 0
    m._episode_index = 0
    m._size_warned = True
    m._n_warned = False
    m.reset()
    m.log = log
    return m


def _run_chunk(m, start_tick: int, ticks: int = C) -> None:
    for t in range(1, ticks + 1):
        m.update_obs(_obs(start_tick + t))


def _episode(m, decisions: int, first_tick: int = 0):
    m.update_obs(_obs(first_tick))
    outputs = [m.get_action()]
    for c in range(1, decisions):
        _run_chunk(m, first_tick + C * (c - 1))
        outputs.append(m.get_action())
    return outputs


@pytest.mark.parametrize("memory", [FIXED, STRIDED], ids=["fixed", "strided"])
def test_the_decision_cycle(memory, capsys):
    m = _model(memory)
    actions = _episode(m, 5)
    assert all(len(chunk) == C for chunk in actions)
    assert set(actions[0][0]) == {"left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state"}
    expected = [("new_session", "cover the blocks")]
    for c in range(5):
        t = C * c
        if c:
            expected.append(("encode", sm.sequence_rows(t - C, 0, memory)))     # the previous decision's reference
        rows = sm.prefix_rows(t, 0, memory)
        expected += [("encode", rows), ("decide", c, 1 + sm.memory_latents(t, 0, memory))]
    assert m.log == expected
    assert len(m._canvases) == 1 + 4 * C and [int(c.flatten()[0]) for c in m._canvases] == list(range(1 + 4 * C))
    out = capsys.readouterr().out
    n = sm.memory_latents(C * 4, 0, memory)
    assert f"decision 4 t=128: memory_latents={n} prefix_rows=" in out
    assert "world_model_rel_rmse(decision 3)=" in out and "cmd_vs_reached(decision 3)" in out


def test_without_the_world_model_error_only_the_prefixes_are_encoded(capsys):
    m = _model(FIXED, log_world_model_error=False)
    _episode(m, 3)
    assert [entry[1] for entry in m.log if entry[0] == "encode"] == [sm.prefix_rows(C * c, 0, FIXED) for c in range(3)]
    assert "world_model_rel_rmse" not in capsys.readouterr().out


def test_a_strided_memory_past_the_trained_rung_warns_once():
    m = _model(STRIDED, trained=1)
    with pytest.warns(UserWarning, match="past the 1 the recipe trains"):
        _episode(m, 9)     # t = 224 reads ceil(ceil(224 / 25) / 8) = 2 memory latents
    assert m._n_warned


def test_a_second_episode_starts_over_after_the_double_reset():
    m = _model(FIXED)
    _episode(m, 2)
    m.reset()
    m.reset()
    assert m._canvases == [] and m._session is None and m._pending is None
    m.log.clear()
    _episode(m, 2, first_tick=200)
    assert m._episode_index == 2
    # tick 0 of the new episode is its first frame (filled with 200)
    assert m.log[1] == ("encode", [200] * 17) and m.log[-1] == ("decide", 1, 3)


def test_an_incomplete_chunk_and_a_changed_instruction_are_refused():
    m = _model(FIXED)
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0, ticks=C - 1)
    with pytest.raises(RuntimeError, match="executed for 31 of 32"):
        m.get_action()
    m.update_obs(_obs(C))
    m._latest_obs = _obs(C, instruction="another task")
    with pytest.raises(RuntimeError, match="instruction changed"):
        m.get_action()
    with pytest.raises(RuntimeError, match="observations after a chunk"):
        for t in range(2):
            m.update_obs(_obs(C + 1 + t))


def test_the_seed_the_knobs_and_the_row_log():
    assert adapter.derive_diffusion_seed(1, 0, 1, 0) != adapter.derive_diffusion_seed(1, 0, 1, 1)
    assert adapter.derive_diffusion_seed(1, 0, 1, 0) == adapter.derive_diffusion_seed(1, 0, 1, 0)
    assert adapter._optional_str("null") is None and adapter._optional_str(" Uncached ") == "Uncached"
    assert adapter.compact_rows([0, 1, 2]) == "[0, 1, 2]"
    assert adapter.compact_rows(list(range(65))) == "65x[0, 1, 2, .., 62, 63, 64]"
