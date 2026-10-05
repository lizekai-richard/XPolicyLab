"""D5: normalization and denormalization of the causal adapter, against Sana's deploy robot I/O.

* The anchor state (measured joints -> URDF FK -> base-frame E pose + rot6d, gripper closedness from the opening,
  joints masked for EEF-only) and its normalization equal Sana ``PolicyRobotIO.state80`` / ``normalize_state80``.
* Sampled rows denormalize exactly like ``PolicyRobotIO.actions80`` (absolute EEF, gripper closedness clipped).
* The committed action of an executed chunk = the EE command actually sent, re-normalized with the same artifact:
  orthonormal rot6d and in-range grippers round-trip to the model output (fp32), clipped grippers commit at the
  bound, a non-orthonormal rot6d commits the rotation the evaluator received.

The normalization artifact is the vanilla52k sana_pixel donor's (sha 9f8b98ed, the statistics the causal recipe's
1fe3b7e7 pin carries); the training yaml is the derived causal fixture. Live parts need ``SANA_CAUSAL_REPO``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import yaml

import _paths

from sana_wam_causal.contract import bidirectional_view
from sana_wam_min.actions import model_action_to_absolute
from sana_wam_min.eef import (
    ArxX5Kinematics,
    eef_state_slot_mask,
    fill_eef_state_slots,
    matrix_to_rot6d,
    native_ee_actions_from_action80,
    root_transforms,
)
from sana_wam_min.robodojo_io import ROBOT80_JOINT_SLOTS_12, state80_from_obs
from sana_wam_min.robot80 import normalize_action, normalize_state
from sana_wam_min.session import load_checked_normalization

ARTIFACT = os.path.join(_paths.FIXTURES, "normalization_vanilla52k_f33.json")
CONFIG = os.path.join(_paths.FIXTURES, "config_causal_vanilla52k_m48n24_derived.yaml")
SANA_CAUSAL_REPO = os.path.expanduser(os.environ.get("SANA_CAUSAL_REPO", "~/zekail/Sana"))
FP32 = {"rtol": 2e-6, "atol": 2e-6}

from XPolicyLab.policy.SANA_WAM_CAUSAL.model import executed_rows  # noqa: E402  (the adapter's helper)


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def normalization(cfg, tmp_path_factory):
    return load_checked_normalization(str(tmp_path_factory.mktemp("ckpt")), bidirectional_view(cfg), ARTIFACT)


def _eef_only_mask() -> np.ndarray:
    mask = eef_state_slot_mask(True).copy()
    mask[list(ROBOT80_JOINT_SLOTS_12)] = False
    return mask


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else -q


def _raw_rows(rng: np.random.Generator, k: int = 32) -> np.ndarray:
    rows = np.zeros((k, 80), dtype=np.float32)
    for position, rotation, gripper in ((slice(7, 10), slice(10, 16), 16), (slice(36, 39), slice(39, 45), 45)):
        rows[:, position] = rng.uniform([0.15, -0.25, 0.15], [0.35, 0.25, 0.32], size=(k, 3))
        rows[:, rotation] = np.stack([matrix_to_rot6d(_random_rotation(rng)) for _ in range(k)])
        rows[:, gripper] = rng.uniform(0.0, 1.0, size=k)
    return rows


def test_the_eef_only_slots_and_schemes(normalization):
    mask = _eef_only_mask()
    assert np.flatnonzero(mask).tolist() == list(range(7, 17)) + list(range(36, 46))
    assert normalization.eef_target_mode == "absolute"
    center, scale = np.asarray(normalization.action_center80), np.asarray(normalization.action_scale80)
    assert np.allclose(center[[16, 45]], 0.5) and np.allclose(scale[[16, 45]], 0.5)  # closedness [0, 1] -> [-1, 1]
    norm_mask = np.asarray(normalization.action_normalization_mask80)
    assert norm_mask[7:10].all() and norm_mask[36:39].all() and norm_mask[[16, 45]].all()
    # rot6d: fixed [-1, 1] range = identity (center 0, scale 1), whichever way the artifact encodes it
    rot = np.r_[10:16, 39:45]
    assert np.allclose(np.where(norm_mask[rot], center[rot], 0.0), 0.0)
    assert np.allclose(np.where(norm_mask[rot], scale[rot], 1.0), 1.0)


def test_executed_commands_commit_as_the_model_output_when_nothing_was_changed(normalization):
    rng = np.random.default_rng(0)
    mask = torch.as_tensor(np.broadcast_to(_eef_only_mask(), (32, 80)).copy())
    raw = torch.as_tensor(_raw_rows(rng))
    model_rows = normalize_action(raw, mask, normalization)
    anchor = np.zeros(80, dtype=np.float32)
    back = model_action_to_absolute(model_rows, mask, normalization, anchor, mask[0].numpy())
    torch.testing.assert_close(back, raw, **FP32)
    committed = normalize_action(executed_rows(back, mask), mask, normalization)
    torch.testing.assert_close(committed, model_rows, **FP32)


def test_clipped_grippers_and_non_orthonormal_rotations_commit_what_was_sent(normalization):
    rng = np.random.default_rng(1)
    mask = torch.as_tensor(np.broadcast_to(_eef_only_mask(), (32, 80)).copy())
    model_rows = normalize_action(torch.as_tensor(_raw_rows(rng)), mask, normalization)
    model_rows[:5, 16] = 1.4      # closedness 1.2 -> clipped to 1 (normalized +1)
    model_rows[5:9, 45] = -1.3    # closedness -0.15 -> clipped to 0 (normalized -1)
    model_rows[:, 10:16] += torch.as_tensor(rng.normal(scale=0.05, size=(32, 6)), dtype=torch.float32)  # not orthonormal
    raw = model_action_to_absolute(model_rows, mask, normalization, np.zeros(80, np.float32), mask[0].numpy())
    executed = executed_rows(raw, mask)
    committed = normalize_action(executed, mask, normalization)
    assert torch.allclose(committed[:5, 16], torch.ones(5)) and torch.allclose(committed[5:9, 45], -torch.ones(4))
    for row in executed.numpy():
        first, second = row[10:13], row[13:16]
        assert abs(np.linalg.norm(first) - 1) < 1e-5 and abs(np.linalg.norm(second) - 1) < 1e-5
        assert abs(float(first @ second)) < 1e-5
    world = root_transforms(None)
    sent = native_ee_actions_from_action80(raw, world)
    kept = native_ee_actions_from_action80(executed, world)
    for a, b in zip(sent, kept):
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=2e-6)
    # untouched slots still round-trip
    torch.testing.assert_close(committed[:, 36:45], model_rows[:, 36:45], **FP32)


def _robot_io(cfg):
    if not os.path.isdir(os.path.join(SANA_CAUSAL_REPO, "dev", "rwm")):
        pytest.skip(f"no Sana checkout at {SANA_CAUSAL_REPO}")
    if SANA_CAUSAL_REPO not in sys.path:
        sys.path.insert(0, SANA_CAUSAL_REPO)
    module = pytest.importorskip("dev.rwm.deploy.robot_io")
    return module.PolicyRobotIO.from_config(cfg, ARTIFACT)


def test_state_and_actions_match_sanas_deploy_robot_io(cfg, normalization):
    robot_io = _robot_io(cfg)
    assert robot_io.action_mode == "robot_base_eef" and robot_io.eef_target_mode == "absolute"
    np.testing.assert_array_equal(robot_io.action_mask80.numpy(), _eef_only_mask())
    rng = np.random.default_rng(2)
    kinematics = ArxX5Kinematics(None)
    for _ in range(5):
        joints = {side: rng.uniform(-0.6, 0.6, size=6) for side in ("left", "right")}
        opening = {side: float(rng.uniform(0.0, 1.0)) for side in ("left", "right")}
        obs_state = {
            "left_arm_joint_state": joints["left"], "right_arm_joint_state": joints["right"],
            "left_ee_joint_state": np.array([opening["left"]]), "right_ee_joint_state": np.array([opening["right"]]),
        }
        state80, mask80 = state80_from_obs(obs_state)
        state80, mask80 = fill_eef_state_slots(state80, eef_state_slot_mask(True), kinematics)
        mask80 = mask80.copy()
        mask80[list(ROBOT80_JOINT_SLOTS_12)] = False
        sana_state, sana_mask = robot_io.state80(joints, opening)
        np.testing.assert_array_equal(mask80, sana_mask.numpy())
        np.testing.assert_allclose(np.where(mask80, state80, 0.0), sana_state.numpy(), atol=2e-6)
        ours_norm = normalize_state(state80, mask80, normalization)
        torch.testing.assert_close(ours_norm, robot_io.normalize_state80(sana_state, sana_mask), **FP32)

    mask = torch.as_tensor(np.broadcast_to(_eef_only_mask(), (32, 80)).copy())
    model_rows = torch.as_tensor(rng.normal(scale=0.7, size=(32, 80)), dtype=torch.float32).masked_fill(~mask, 0)
    ours = model_action_to_absolute(model_rows, mask, normalization, np.zeros(80, np.float32), mask[0].numpy())
    theirs = robot_io.actions80(model_rows, torch.zeros(80))
    torch.testing.assert_close(ours, theirs, **FP32)

    from dev.rwm.diffusion.data.robot80_normalization import normalize_robot80_affine

    executed = executed_rows(ours, mask)
    sana_committed = normalize_robot80_affine(
        executed, mask, robot_io.normalization["action_center80"], robot_io.normalization["action_scale80"],
        robot_io.normalization["action_normalization_mask80"],
    )
    torch.testing.assert_close(normalize_action(executed, mask, normalization), sana_committed, **FP32)
