"""The XPolicyLab adapter's chunk cycle against a fake runtime (CPU, no weights): the RoboDojo call order of
``policy/<POLICY>/deploy.py`` (update_obs -> get_action, then update_obs after every executed action but the last,
whose observation arrives right before the next get_action), tick counting, the frames a commit encodes, the order
commit -> generate, the double reset of every RoboDojo episode, and the refusals."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import _paths

import XPolicyLab.policy.SANA_WAM_CAUSAL.model as adapter
from sana_wam_causal.contract import bidirectional_view
from sana_wam_min.eef import ArxX5Kinematics, root_transforms
from sana_wam_min.robot80 import normalized_gripper_bounds
from sana_wam_min.session import load_checked_normalization

C, S = 32, 4
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class FakeSession:
    def __init__(self, log, instruction):
        self.log, self.instruction, self.chunk_idx = log, instruction, 0

    def commit_observation(self, latent):
        self.log.append(("commit_observation", float(latent.sum())))

    def read_entry_ids(self):
        return list(range(self.chunk_idx + 1))

    def generate_chunk(self, *, anchor_state, anchor_state_mask, action_mask, generator, gripper_bounds):
        self.log.append(("generate", self.chunk_idx, anchor_state.clone(), action_mask.clone()))
        action = torch.zeros(1, C, 80)
        action[..., 10] = 1.0   # rot6d column 1 = x axis
        action[..., 14] = 1.0   # column 2 = y axis
        action[..., 39] = 1.0
        action[..., 43] = 1.0
        action[..., 16] = 0.25
        return action.masked_fill(~action_mask, 0), torch.zeros(1, 4, 1, 3, 4)

    def commit_chunk(self, latent, executed_action, *, anchor_state, anchor_state_mask, action_mask):
        self.log.append(("commit_chunk", self.chunk_idx, float(latent.sum()), executed_action.clone(), anchor_state.clone()))
        self.chunk_idx += 1


class FakeRuntime:
    def __init__(self, log):
        self.log = log
        self.device = torch.device("cpu")
        self.contract = SimpleNamespace(actions_per_chunk=C, video_frame_stride=S)

    def new_session(self, instruction):
        self.log.append(("new_session", instruction))
        return FakeSession(self.log, instruction)

    def encode_observation(self, frames):
        return torch.full((1, 4, 1, 3, 4), float(frames[0][0, 0, 0]))

    def encode_executed_chunk(self, frames_per_view):
        ticks = [int(frames[0][0, 0, 0]) for frames in frames_per_view]
        self.log.append(("encode_chunk", ticks))
        return torch.full((1, 4, 1, 3, 4), float(sum(ticks)))


def _obs(tick: int, instruction: str = "stack the bowls") -> dict:
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


@pytest.fixture()
def model():
    with open(os.path.join(_paths.FIXTURES, "config_causal_vanilla52k_m48n24_derived.yaml")) as handle:
        view = bidirectional_view(yaml.safe_load(handle))
    normalization = load_checked_normalization(
        "/nonexistent", view, os.path.join(_paths.FIXTURES, "normalization_vanilla52k_f33.json")
    )
    log: list = []
    m = adapter.Model.__new__(adapter.Model)
    m.runtime = FakeRuntime(log)
    m.normalization = normalization
    m.gripper_bounds = normalized_gripper_bounds(normalization)
    m.kinematics = ArxX5Kinematics(None)
    m.world_from_base = root_transforms(None)
    m.chunk_actions, m.frame_stride = C, S
    m.strict_image_size = False
    m.default_instruction = "follow the instruction"
    m.eef_pose_check = False
    m.commit_actions = "executed"
    m.max_chunks_per_episode = None
    m.log_world_model_error = True
    m.diffusion_seed_base, m.eval_seed = 1, 0
    m._episode_index = 0
    m._size_warned = True
    m.reset()
    m.log = log
    return m


def _run_chunk(m, start_tick: int, ticks: int = C) -> None:
    for t in range(1, ticks + 1):
        m.update_obs(_obs(start_tick + t))


def test_the_robodojo_cycle(model):
    m = model
    m.reset()
    m.reset()  # RoboDojo: eval_env reset + deploy.py reset
    m.update_obs(_obs(0))
    assert m._episode_index == 1
    actions = m.get_action()
    assert len(actions) == C and set(actions[0]) == {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}
    assert [entry[0] for entry in m.log] == ["new_session", "commit_observation", "generate"]
    _run_chunk(m, 0)
    m.get_action()
    names = [entry[0] for entry in m.log]
    assert names == ["new_session", "commit_observation", "generate", "encode_chunk", "commit_chunk", "generate"]
    encode = m.log[3]
    assert encode[1] == list(range(0, C + 1, S))  # ticks 0, 4, .., 32 of chunk 0
    commit = m.log[4]
    first_generate = m.log[2]
    assert torch.equal(commit[4], first_generate[2])   # the chunk-start anchor of chunk 0, not the latest state
    second_generate = m.log[5]
    assert not torch.equal(second_generate[2], first_generate[2])  # chunk 1 anchors on the tick-32 observation
    committed = commit[3][0]
    mask = first_generate[3][0]
    assert torch.count_nonzero(committed[~mask]) == 0
    _run_chunk(m, C)
    m.get_action()
    assert m.log[-3][1] == list(range(C, 2 * C + 1, S))  # chunk 1's frames: ticks 32, 36, .., 64


def test_the_episode_counter_advances_once_per_episode(model):
    m = model
    for episode in (1, 2, 3):
        m.reset()
        m.reset()
        m.update_obs(_obs(0))
        m.get_action()
        assert m._episode_index == episode


def test_a_partial_chunk_is_refused(model):
    m = model
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0, ticks=10)
    with pytest.raises(RuntimeError, match="10 of 32 ticks"):
        m.get_action()


def test_more_observations_than_actions_are_refused(model):
    m = model
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0)
    with pytest.raises(RuntimeError, match="observations after a chunk"):
        m.update_obs(_obs(33))


def test_the_instruction_is_fixed_per_episode(model):
    m = model
    m.update_obs(_obs(0))
    m.get_action()
    for t in range(1, C + 1):
        m.update_obs(_obs(t, instruction="pour the water"))
    with pytest.raises(RuntimeError, match="instruction changed"):
        m.get_action()


def test_the_eef_pose_check_runs_on_the_first_chunk_of_every_episode(model, monkeypatch):
    m = model
    m.eef_pose_check = True
    calls = []
    monkeypatch.setattr(m, "_check_observed_eef_pose", lambda obs_state, state80: calls.append(m._episode_index))
    for _episode in (1, 2):
        m.reset()
        m.reset()
        m.update_obs(_obs(0))
        m.get_action()
        _run_chunk(m, 0)
        m.get_action()
    assert calls == [1, 2]


def test_reset_drops_the_memory(model):
    m = model
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0, ticks=7)
    m.reset()   # episode ended mid-chunk: nothing is committed
    m.update_obs(_obs(100))
    m.get_action()
    names = [entry[0] for entry in m.log]
    assert names.count("commit_chunk") == 0 and names.count("new_session") == 2
