"""Parity of sana_wam_min/eef.py against the Sana training code itself (skipped when a Sana checkout is not importable).

Uses Sana's own ``pack_robot_base_eef_state80`` / ``_current_anchor_action`` on FK poses produced by Sana's URDF FK and
axis transform, and checks that the adapter (a) packs the same state rows from the joints alone and (b) inverts Sana's
anchor-relative EEF targets back to the true rows. Point ``SANA_REPO`` at a checkout (default ~/zekail/Sana)."""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest
import torch

_SANA_WAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SANA_WAM_DIR not in sys.path:
    sys.path.insert(0, _SANA_WAM_DIR)
SANA_REPO = os.environ.get("SANA_REPO", os.path.expanduser("~/zekail/Sana"))
if SANA_REPO not in sys.path and os.path.isdir(SANA_REPO):
    sys.path.insert(0, SANA_REPO)

sana = pytest.importorskip("dev.rwm.diffusion.data.robot80_action", reason="Sana checkout not importable (set SANA_REPO)")
from dev.rwm.diffusion.data.data_format.robotwin2 import _STANDARD_E_ROWS  # noqa: E402
from dev.rwm.diffusion.data.eef.robot_base_eef_kinematics import UrdfModel, batch_fk_all  # noqa: E402
from dev.rwm.diffusion.data.eef.robot_base_eef_transforms import EefAxisTransform, apply_eef_axis_transform_all  # noqa: E402

from sana_wam_min import eef  # noqa: E402
from sana_wam_min.actions import reconstruct_absolute_joints  # noqa: E402

ACTIVE = tuple(list(range(0, 6)) + list(range(7, 17)) + list(range(29, 35)) + list(range(36, 46)))   # dual_arm32
NAMES = tuple(f"joint{i}" for i in range(1, 7))


def _sana_episode(left_q, right_q):
    model = UrdfModel.from_file(os.path.join(SANA_REPO, "dev/rwm/configs/urdf/robotwin2_arx_x5.urdf"))
    axis = EefAxisTransform(revision="parity", source_eef_from_eef=np.asarray(_STANDARD_E_ROWS, dtype=np.float64), meaning="parity test")
    T = len(left_q)
    sides = {
        side: types.SimpleNamespace(
            robot_base_from_eef=apply_eef_axis_transform_all(
                batch_fk_all(model, base_link="base_link", target_link="link6", joint_names=NAMES, joint_position=q), axis
            ),
            gripper=types.SimpleNamespace(closedness=np.linspace(0.0, 1.0, T)),
        )
        for side, q in (("left", left_q), ("right", right_q))
    }
    return types.SimpleNamespace(
        joint_position=np.concatenate([left_q, right_q], axis=1),
        joint_indices_by_side={"left": tuple(range(6)), "right": tuple(range(6, 12))},
        sides=sides,
        grippers=None,
    )


def test_adapter_packs_the_same_state_rows_as_sana():
    assert np.array_equal(eef.LINK6_FROM_E, np.asarray(_STANDARD_E_ROWS, dtype=np.float64))
    rng = np.random.default_rng(1)
    left_q, right_q = rng.uniform(-1.5, 1.5, size=(12, 6)), rng.uniform(-1.5, 1.5, size=(12, 6))
    state_sana, mask_sana = sana.pack_robot_base_eef_state80(_sana_episode(left_q, right_q), active_slots=ACTIVE, include_eef=True)
    kin = eef.ArxX5Kinematics()
    for t in range(len(left_q)):
        row = np.zeros(80, dtype=np.float32)
        row[0:6], row[29:35], row[16], row[45] = left_q[t], right_q[t], state_sana[t, 16], state_sana[t, 45]
        state_adp, mask_adp = eef.fill_eef_state_slots(row, eef.eef_state_slot_mask(True), kin)
        assert np.array_equal(mask_adp, mask_sana[t])
        np.testing.assert_allclose(state_adp, state_sana[t], atol=2e-7)


def test_adapter_inverts_sana_anchor_relative_targets():
    rng = np.random.default_rng(2)
    T = 25
    left_q, right_q = rng.uniform(-1.5, 1.5, size=(T, 6)), rng.uniform(-1.5, 1.5, size=(T, 6))
    state80, mask80 = sana.pack_robot_base_eef_state80(_sana_episode(left_q, right_q), active_slots=ACTIVE, include_eef=True)
    contiguous = np.ones(T, dtype=bool)
    contiguous[0] = False
    action, action_mask = sana._current_anchor_action(
        state80=state80.astype(np.float32), state_mask80=mask80, transition_contiguous=contiguous,
        T_world_camera=np.repeat(np.eye(4)[None], T, axis=0), camera_mask=np.ones(T, dtype=bool),   # robot_base_qwen = identity camera
        active_mask80=np.isin(np.arange(80), ACTIVE), joint_target_mode="anchor_delta", actions_per_chunk=None,
    )
    anchor, anchor_mask = state80[0], mask80[0]
    absolute = reconstruct_absolute_joints(torch.from_numpy(action), torch.from_numpy(action_mask), anchor, anchor_mask)
    absolute = eef.reconstruct_absolute_eef(absolute, action_mask, anchor, anchor_mask).numpy()
    np.testing.assert_allclose(absolute[:, list(ACTIVE)], state80[1:T][:, list(ACTIVE)], atol=5e-7)
