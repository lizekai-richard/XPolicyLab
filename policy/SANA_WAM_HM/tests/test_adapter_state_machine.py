"""The XPolicyLab adapter's decision cycle against a fake runtime (CPU, no weights): the RoboDojo call order of
``policy/<POLICY>/deploy.py`` (update_obs -> get_action, then update_obs after every executed action but the last,
whose observation arrives right before the next get_action), one canvas per tick, the ticks the dense stream, the
sparse memory and the world-model reference encode at every decision, the order dense absorb -> observation -> sparse
memory -> generate, the double reset of every RoboDojo episode, and the refusals."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import _paths

import XPolicyLab.policy.SANA_WAM_HM.model as adapter
from sana_wam_hm import memory as hm
from sana_wam_hm.contract import bidirectional_view
from sana_wam_min.eef import ArxX5Kinematics, root_transforms
from sana_wam_min.robot80 import normalized_gripper_bounds
from sana_wam_min.session import load_checked_normalization

C = 32
M = 16
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _latent(value: float, frames: int = 1) -> torch.Tensor:
    return torch.full((1, 4, frames, 1, 1), float(value))


class FakeStream:
    """A causal-VAE stream whose latents carry the ticks it encodes."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def start(self, canvas):
        self.log.append((f"{self.name}.start", int(canvas.flatten()[0])))
        return _latent(int(canvas.flatten()[0]))

    def extend(self, canvases):
        ticks = [int(c.flatten()[0]) for c in canvases]
        self.log.append((f"{self.name}.extend", ticks))
        return _latent(sum(ticks))


class FakeSession:
    def __init__(self, log, streams):
        self.log, self.streams = log, streams
        self.decision, self.dense_count = 0, 0
        self.deploy_form = "streaming"

    @property
    def reads_dense(self):
        return hm.DENSE in self.streams

    @property
    def reads_sparse(self):
        return hm.SPARSE in self.streams

    def add_dense(self, latent):
        self.log.append(("add_dense", self.dense_count, float(latent.flatten()[0])))
        assert self.dense_count == self.decision
        self.dense_count += 1

    def generate(self, observation_latent, sparse_latents, *, anchor_state, anchor_state_mask, action_mask, generator, gripper_bounds):
        self.log.append(("generate", self.decision, float(observation_latent.flatten()[0])))
        if self.reads_dense:
            assert self.dense_count == self.decision + 1
        action = torch.zeros(1, C, 80)
        action[..., 10] = 1.0   # rot6d column 1 = x axis
        action[..., 14] = 1.0   # column 2 = y axis
        action[..., 39] = 1.0
        action[..., 43] = 1.0
        action[..., 16] = 0.25
        self.decision += 1
        return action.masked_fill(~action_mask, 0), _latent(1000.0 + self.decision)


class FakeRuntime:
    """Canvases carry the tick of their frame (the frames are filled with tick % 256)."""

    def __init__(self, log, streams):
        self.log, self.streams = log, streams
        self.device = torch.device("cpu")
        self.contract = SimpleNamespace(actions_per_chunk=C, sparse_frames=M)

    def new_session(self, instruction):
        self.log.append(("new_session", instruction))
        return FakeSession(self.log, self.streams)

    def canvas(self, frames):
        return torch.full((3, 1, 1), int(frames[0][0, 0, 0]), dtype=torch.uint8)

    def encode_observation(self, canvas):
        self.log.append(("encode_obs", int(canvas.flatten()[0])))
        return _latent(int(canvas.flatten()[0]))

    def encode_sparse(self, canvases):
        ticks = [int(c.flatten()[0]) for c in canvases]
        self.log.append(("encode_sparse", ticks))
        stream = FakeStream(self.log, "sparse")
        return _latent(sum(ticks), hm.sparse_latents(M)), stream

    def new_dense_stream(self):
        return FakeStream(self.log, "dense")


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


def _model(streams):
    with open(os.path.join(_paths.FIXTURES, "recipe_hm_dense_sparse_videoonly_e8dde6b56.yaml")) as handle:
        view = bidirectional_view(yaml.safe_load(handle))
    normalization = load_checked_normalization(
        "/nonexistent", view, os.path.join(_paths.FIXTURES, "normalization_causal_f33_1fe3b7e7.json")
    )
    log: list = []
    m = adapter.Model.__new__(adapter.Model)
    m.runtime = FakeRuntime(log, streams)
    m.contract = m.runtime.contract
    m.normalization = normalization
    m.gripper_bounds = normalized_gripper_bounds(normalization)
    m.kinematics = ArxX5Kinematics(None)
    m.world_from_base = root_transforms(None)
    m.chunk_actions = C
    m.strict_image_size = False
    m.default_instruction = "follow the instruction"
    m.eef_pose_check = False
    m.max_chunks_per_episode = None
    m.log_world_model_error = True
    m.diffusion_seed_base, m.eval_seed = 1, 0
    m._episode_index = 0
    m._size_warned = True
    m._k_warned = False
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


def test_the_dense_sparse_cycle(capsys):
    m = _model((hm.DENSE, hm.SPARSE))
    actions = _episode(m, 4)
    assert all(len(chunk) == C for chunk in actions)
    assert set(actions[0][0]) == {"left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state"}
    expected = [("new_session", "cover the blocks"), ("dense.start", 0), ("add_dense", 0, 0.0), ("encode_obs", 0),
                ("encode_sparse", hm.sparse_ticks(0, M)), ("generate", 0, 0.0)]
    for c in range(1, 4):
        ticks = hm.dense_chunk_ticks(c)
        expected += [("dense.extend", ticks), ("add_dense", c, float(sum(ticks))), ("encode_obs", C * c),
                     ("encode_sparse", hm.sparse_ticks(c, M)), ("generate", c, float(C * c))]
    assert m.log == expected
    assert len(m._canvases) == 1 + 3 * C and [int(c.flatten()[0]) for c in m._canvases] == list(range(1 + 3 * C))
    out = capsys.readouterr().out
    assert "decision 0 t=0: dense=L_0..L_0 sparse_ticks=" in out and "decision 3 t=96: dense=L_0..L_3" in out
    assert "world_model_rel_rmse(decision 2)=" in out and "cmd_vs_reached(decision 2)" in out


def test_the_sparse_sparse_cycle_scores_against_the_target_after_the_sparse_rows(capsys):
    m = _model((hm.SPARSE,))
    _episode(m, 3)
    expected = [("new_session", "cover the blocks"), ("encode_obs", 0), ("encode_sparse", hm.sparse_ticks(0, M)),
                ("generate", 0, 0.0)]
    for c in range(1, 3):
        expected += [("sparse.extend", hm.target_rows(C * (c - 1))), ("encode_obs", C * c),
                     ("encode_sparse", hm.sparse_ticks(c, M)), ("generate", c, float(C * c))]
    assert m.log == expected
    assert "world_model_rel_rmse(decision 1)=" in capsys.readouterr().out


def test_a_second_episode_starts_over_after_the_double_reset():
    m = _model((hm.DENSE, hm.SPARSE))
    _episode(m, 2)
    m.reset()
    m.reset()
    assert m._canvases == [] and m._session is None and m._dense_stream is None
    m.log.clear()
    _episode(m, 2, first_tick=200)
    assert m._episode_index == 2
    assert m.log[1] == ("dense.start", 200) and m.log[-1][0] == "generate" and m.log[-1][1] == 1


def test_an_incomplete_chunk_and_a_changed_instruction_are_refused():
    m = _model((hm.DENSE, hm.SPARSE))
    m.update_obs(_obs(0))
    m.get_action()
    _run_chunk(m, 0, ticks=C - 1)
    with pytest.raises(RuntimeError, match="complete chunks only"):
        m.get_action()
    m.update_obs(_obs(C))
    m._latest_obs = _obs(C, instruction="another task")
    with pytest.raises(RuntimeError, match="instruction changed"):
        m.get_action()
    with pytest.raises(RuntimeError, match="observations after a chunk"):
        for t in range(2):
            m.update_obs(_obs(C + 1 + t))


def test_the_seed_and_the_knobs():
    assert adapter.derive_diffusion_seed(1, 0, 1, 0) != adapter.derive_diffusion_seed(1, 0, 1, 1)
    assert adapter.derive_diffusion_seed(1, 0, 1, 0) == adapter.derive_diffusion_seed(1, 0, 1, 0)
    assert adapter._optional_str("null") is None and adapter._optional_str(" full ") == "full"
