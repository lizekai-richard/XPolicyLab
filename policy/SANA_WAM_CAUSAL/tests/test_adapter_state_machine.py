"""The XPolicyLab adapter's chunk cycle against a fake runtime (CPU, no weights), episode protocol: the RoboDojo call
order of ``policy/<POLICY>/deploy.py`` (update_obs -> get_action, then update_obs after every executed action but the
last, whose observation arrives right before the next get_action), tick counting, the episode opening (the per-view
streams started at the first frame, the sink and the frozen first chunk committed, chunk 1 generated from the first
frame), the frames every later commit streams, the order stream -> commit -> observation encode -> generate, the
memory ablation (every chunk reopens the episode at its own start frame), a model without chunk observations, the
double reset of every RoboDojo episode, and the refusals."""

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
from sana_wam_min.robot80 import normalize_action, normalized_gripper_bounds
from sana_wam_min.session import load_checked_normalization

C, S = 32, 4
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _tick(view: torch.Tensor) -> int:
    return int(view.flatten()[0])


class FakeSession:
    WINDOW = 2

    def __init__(self, log, instruction):
        self.log, self.instruction, self.chunk_idx, self.has_sink = log, instruction, 0, False
        self.view_shapes = "unset"

    def read_chunk_ids(self):
        return list(range(max(0, self.chunk_idx - self.WINDOW), self.chunk_idx))

    def softmax_origin(self):
        ids = self.read_chunk_ids()
        return ids[0] if ids else self.chunk_idx

    def reset_memory(self):
        self.log.append(("reset_memory",))
        self.chunk_idx, self.has_sink = 0, False

    def commit_sink(self, latent):
        assert not self.has_sink and self.chunk_idx == 0
        self.log.append(("commit_sink", float(latent.flatten()[0])))
        self.has_sink = True

    def generate_chunk(self, observation_latent, *, anchor_state, anchor_state_mask, action_mask, generator, gripper_bounds):
        assert self.has_sink and self.chunk_idx >= 1
        obs = None if observation_latent is None else float(observation_latent.flatten()[0])
        self.log.append(("generate", self.chunk_idx, anchor_state.clone(), action_mask.clone(), obs))
        action = torch.zeros(1, C, 80)
        action[..., 10] = 1.0   # rot6d column 1 = x axis
        action[..., 14] = 1.0   # column 2 = y axis
        action[..., 39] = 1.0
        action[..., 43] = 1.0
        action[..., 16] = 0.25
        return action.masked_fill(~action_mask, 0), torch.zeros(1, 4, 1, 3, 4)

    def commit_chunk(self, latent, executed_action, *, anchor_state, anchor_state_mask, action_mask):
        assert self.has_sink
        self.log.append(("commit_chunk", self.chunk_idx, float(latent.flatten()[0]), executed_action.clone(), anchor_state.clone()))
        self.chunk_idx += 1


class FakeStreams:
    """Latents carry the ticks they encode: the sink its tick, the frozen chunk 1000 + its tick, a chunk the sum."""

    view_shapes = None

    def __init__(self, log):
        self.log = log

    def start(self, views):
        tick = _tick(views[0])
        self.log.append(("stream_start", tick))
        return torch.full((1, 4, 1, 3, 4), float(tick)), torch.full((1, 4, 1, 3, 4), 1000.0 + tick)

    def encode_chunk(self, frames_per_view):
        ticks = [_tick(frame) for frame in frames_per_view[0]]
        self.log.append(("stream_chunk", ticks))
        return torch.full((1, 4, 1, 3, 4), float(sum(ticks)))


class FakeRuntime:
    """One-view canvas; every view tensor carries the tick of its frame (the frames are filled with tick % 256)."""

    strip = False

    def __init__(self, log):
        self.log = log
        self.device = torch.device("cpu")
        self.contract = SimpleNamespace(actions_per_chunk=C, video_frame_stride=S)

    def new_session(self, instruction):
        self.log.append(("new_session", instruction))
        return FakeSession(self.log, instruction)

    def view_frames(self, frames):
        return [torch.full((3, 1, 1), float(frames[0][0, 0, 0]))]

    def encode_observation(self, views):
        tick = _tick(views[0])
        self.log.append(("encode_obs", tick))
        return torch.full((1, 4, 1, 3, 4), float(tick)), None

    def new_streams(self):
        return FakeStreams(self.log)


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
    with open(os.path.join(_paths.FIXTURES, "config_causal_pantheon33k_episode_n24_videoandaction_0081aa702.yaml")) as handle:
        view = bidirectional_view(yaml.safe_load(handle))
    normalization = load_checked_normalization(
        "/nonexistent", view, os.path.join(_paths.FIXTURES, "normalization_causal_f33_9f8b98ed.json")
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
    m.memory_mode = "full"
    m.memory_content = "video_and_action"
    m.chunk_observation = True
    m.max_chunks_per_episode = None
    m.log_world_model_error = True
    m.attention_dump_dir, m._probe = None, None
    m.diffusion_seed_base, m.eval_seed = 1, 0
    m._episode_index = 0
    m._size_warned = True
    m.reset()
    m.log = log
    return m


def _run_chunk(m, start_tick: int, ticks: int = C) -> None:
    for t in range(1, ticks + 1):
        m.update_obs(_obs(start_tick + t))


OPENING = ["stream_start", "reset_memory", "commit_sink", "commit_chunk", "generate"]


def test_the_robodojo_cycle(model):
    m = model
    m.reset()
    m.reset()  # RoboDojo: eval_env reset + deploy.py reset
    m.update_obs(_obs(0))
    assert m._episode_index == 1
    actions = m.get_action()
    assert len(actions) == C and set(actions[0]) == {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}
    assert [entry[0] for entry in m.log] == ["new_session"] + OPENING
    _, start, _, sink, frozen, first = m.log
    assert start[1] == 0 and sink[1] == 0.0                      # the sink: the first frame alone
    assert frozen[1] == 0 and frozen[2] == 1000.0                # chunk 0: the frozen first frame, committed
    assert torch.equal(frozen[4], first[2])                      # its anchor (and held rows) is the first state
    hold = frozen[3][0]
    mask = first[3][0]
    expected = normalize_action(
        torch.from_numpy(np.repeat(m._anchor(_obs(0))[0][None].astype(np.float32), C, axis=0)), mask, m.normalization
    ).masked_fill(~mask, 0)
    assert torch.allclose(hold, expected, atol=1e-6) and torch.count_nonzero(hold[~mask]) == 0
    assert (first[1], first[4]) == (1, 0.0)                      # chunk 1, generated from the first frame (the sink)
    _run_chunk(m, 0)
    m.get_action()
    names = [entry[0] for entry in m.log]
    assert names == ["new_session"] + OPENING + ["stream_chunk", "commit_chunk", "encode_obs", "generate"]
    stream, commit, encode, second = m.log[-4:]
    assert stream[1] == list(range(S, C + 1, S))                 # chunk 1's frames after its start: ticks 4, 8, .., 32
    assert commit[1] == 1 and commit[2] == float(sum(range(S, C + 1, S)))
    assert torch.equal(commit[4], first[2])                      # the chunk-start anchor of chunk 1, not the latest state
    assert not torch.equal(second[2], first[2])                  # chunk 2 anchors on the tick-32 observation
    assert encode[1] == C and (second[1], second[4]) == (2, float(C))
    committed = commit[3][0]
    assert torch.count_nonzero(committed[~mask]) == 0
    _run_chunk(m, C)
    m.get_action()
    assert [entry[0] for entry in m.log[-4:]] == ["stream_chunk", "commit_chunk", "encode_obs", "generate"]
    assert m.log[-4][1] == list(range(C + S, 2 * C + 1, S))     # ticks 36, .., 64
    assert m.log[-2][1] == 2 * C and m.log[-1][1:2] == (3,) and m.log[-1][4] == float(2 * C)


def test_a_model_without_chunk_observations_generates_from_the_memory(model):
    m = model
    m.chunk_observation = False
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0)
    m.get_action()
    names = [entry[0] for entry in m.log]
    assert names == ["new_session"] + OPENING + ["stream_chunk", "commit_chunk", "generate"]
    assert all(entry[4] is None for entry in m.log if entry[0] == "generate")


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
    assert names == ["new_session"] + OPENING + ["new_session"] + OPENING   # the new episode opens at its first frame
    assert m.log[-5][1] == 100 and m.log[-1][1] == 1 and m.log[-1][4] == 100.0


def test_memory_mode_none_reopens_the_episode_at_every_chunk(model, monkeypatch):
    m = model
    m.memory_mode = "none"
    seeds = []
    real_seed = adapter.derive_diffusion_seed
    monkeypatch.setattr(adapter, "derive_diffusion_seed", lambda base, ev, ep, chunk: seeds.append(chunk) or real_seed(base, ev, ep, chunk))
    m.reset()
    m.reset()
    m.update_obs(_obs(0))
    m.get_action()
    for decision in (1, 2):
        _run_chunk(m, (decision - 1) * C)
        m.get_action()
    names = [entry[0] for entry in m.log]
    # every chunk: the executed chunk streamed on the memory it was generated from (its world-model error), then the
    # episode reopened at the chunk's own start frame; nothing of the past is committed
    assert names == ["new_session"] + OPENING + (["stream_chunk"] + OPENING) * 2
    assert [entry[1] for entry in m.log if entry[0] == "stream_start"] == [0, C, 2 * C]
    assert [entry[1:2] + entry[4:5] for entry in m.log if entry[0] == "generate"] == [(1, 0.0), (1, float(C)), (1, float(2 * C))]
    assert [entry[1] for entry in m.log if entry[0] == "stream_chunk"] == [list(range(S, C + 1, S)), list(range(C + S, 2 * C + 1, S))]
    assert seeds == [0, 1, 2]                                    # the noise of the episode's decision index, as in full
    assert m._chunk_index == 3


def test_memory_mode_none_without_the_world_model_error_streams_no_chunk(model):
    m = model
    m.memory_mode = "none"
    m.log_world_model_error = False
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0)
    m.get_action()
    names = [entry[0] for entry in m.log]
    assert "stream_chunk" not in names and names == ["new_session"] + OPENING * 2


def test_memory_mode_none_still_refuses_a_partial_chunk(model):
    m = model
    m.memory_mode = "none"
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0, ticks=10)
    with pytest.raises(RuntimeError, match="10 of 32 ticks"):
        m.get_action()


def test_commit_reached_commits_the_poses_measured_after_each_action(model):
    from sana_wam_min.robot80 import normalize_state
    m = model
    m.commit_actions = "reached"
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0)
    m.get_action()
    commit = [entry for entry in m.log if entry[0] == "commit_chunk"][1]   # chunk 1 (chunk 0 is the frozen opening)
    mask = next(entry for entry in m.log if entry[0] == "generate")[3][0]
    reached = np.stack([m._measured_row(_obs(t)) for t in range(1, C + 1)])   # the pose after action j = tick j + 1
    expected = normalize_action(torch.from_numpy(reached), mask, m.normalization).masked_fill(~mask, 0)
    assert torch.allclose(commit[3][0], expected, atol=1e-6)
    assert torch.count_nonzero(commit[3][0][~mask]) == 0
    # training's invariant again: the last action row of chunk c-1 IS the state slot of chunk c (same observation, same FK)
    second_generate = [entry for entry in m.log if entry[0] == "generate"][1]
    raw_anchor, anchor_mask, _ = m._anchor(_obs(C))
    assert np.array_equal(reached[-1][anchor_mask], raw_anchor[anchor_mask])
    eef = torch.as_tensor(anchor_mask)
    assert torch.allclose(commit[3][0][-1][eef], second_generate[2][eef], atol=1e-6)
    assert torch.allclose(second_generate[2], normalize_state(raw_anchor, anchor_mask, m.normalization), atol=1e-6)


@pytest.mark.parametrize("memory_mode", ["full", "none"])
def test_command_vs_reached_and_the_world_model_error_are_logged_for_every_executed_chunk(model, capsys, memory_mode):
    m = model
    m.memory_mode = memory_mode
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0)
    m.get_action()
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if "ep 1 decision 1 " in l)
    assert "cmd_vs_reached(decision 0): L pos " in line and " | R pos " in line and " deg grip " in line
    assert "world_model_rel_rmse(decision 0)=" in line and "stream_s=" in line
    if memory_mode == "full":
        assert "chunk 2:" in line and "read_chunks=[0, 1] origin=0" in line and "obs_encode_s=" in line
    else:
        assert "chunk 1:" in line and "read_chunks=[0] origin=0" in line and "open_stream_s=" in line


def test_commit_actions_other_than_executed_need_a_video_and_action_memory():
    adapter.check_commit_actions("executed", "video_only")
    for value in ("executed", "predicted", "reached"):
        adapter.check_commit_actions(value, "video_and_action")
    for value in ("predicted", "reached"):
        with pytest.raises(ValueError, match="needs memory_content video_and_action"):
            adapter.check_commit_actions(value, "video_only")
    with pytest.raises(ValueError, match="commit_actions must be one of"):
        adapter.check_commit_actions("sent", "video_and_action")


def test_command_vs_reached_measures_position_rotation_and_gripper():
    from sana_wam_min.eef import ARM_FIELDS, matrix_to_rot6d
    rows = np.zeros((2, 80), dtype=np.float64)
    mask = np.zeros((2, 80), dtype=np.bool_)
    for _side, _joints, position, rotation, gripper in ARM_FIELDS:
        rows[:, rotation] = matrix_to_rot6d(np.eye(3))
        mask[:, position] = mask[:, rotation] = True
        mask[:, gripper] = True
    reached = rows.copy()
    left = ARM_FIELDS[0]
    reached[1, left[2]] += np.array([0.003, 0.004, 0.0])                     # 5 mm on the second row
    angle = np.radians(10.0)
    rz = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    reached[0, left[3]] = matrix_to_rot6d(rz)
    reached[0, left[4]] = 0.25
    gaps = adapter.command_vs_reached(rows, reached, mask)
    mean, peak, deg, grip = gaps["left"]
    assert np.isclose(peak, 0.005) and np.isclose(mean, 0.0025) and np.isclose(deg, 10.0, atol=1e-4) and np.isclose(grip, 0.25)
    assert gaps["right"] == (0.0, 0.0, 0.0, 0.0)
    assert adapter.format_command_vs_reached(gaps).startswith("L pos 2.5/5.0 mm rot 10.00 deg grip 0.25 | R pos 0.0/0.0 mm")
