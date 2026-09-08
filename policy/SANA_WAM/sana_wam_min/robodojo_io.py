"""RoboDojo ARX-X5 wire codecs: raw observation dict -> Robot80 state, absolute Robot80 chunk -> native action dicts.

NumPy ports of the joint-only contract in ``dev/rwm/simulation_evaluation/robodojo/common/protocol.py``,
the state assembly of ``client/observation_codec.py``, and the native dict emitted by
``client/action_codec.py``. Grippers cross the boundary as ``closedness = 1 - opening``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch

from .robot80 import LEFT_GRIPPER, LEFT_JOINT, RIGHT_GRIPPER, RIGHT_JOINT, ROBOT80_DIM

ROBODOJO_VIEW_ORDER = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ROBODOJO_ANCHOR_VIEW = "cam_head"
# Spatial slot of each view in the 2x2 multiview canvas (TOP_LEFT, BOTTOM_LEFT, BOTTOM_RIGHT).
VIEW_SLOT_IDS = (0, 2, 3)
VIEW_ROLES = {"cam_head": "anchor", "cam_left_wrist": "wrist_left", "cam_right_wrist": "wrist_right"}

ROBODOJO_ARM_DIM = 6
LEFT_JOINT_SLOTS = LEFT_JOINT
RIGHT_JOINT_SLOTS = RIGHT_JOINT
LEFT_GRIPPER_SLOT = LEFT_GRIPPER
RIGHT_GRIPPER_SLOT = RIGHT_GRIPPER
JOINT_ONLY_ACTIVE_SLOTS = (0, 1, 2, 3, 4, 5, 16, 29, 30, 31, 32, 33, 34, 45)
ROBOT80_JOINT_SLOTS_12 = (0, 1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34)

ROBODOJO_MODEL_FPS_HZ = 25.0
ROBODOJO_ACTION_PERIOD_NS = 40_000_000
ROBODOJO_SUPPORTED_INFERENCE_NUM_FRAMES = (25,)

ROBODOJO_FROZEN_IMAGE_HEIGHT = 480
ROBODOJO_FROZEN_IMAGE_WIDTH = 640
ROBODOJO_FROZEN_FOCAL_PX = 640 * 10.0 / 22.212
ROBODOJO_FROZEN_INTRINSICS = np.array(
    [[ROBODOJO_FROZEN_FOCAL_PX, 0.0, 320.0], [0.0, ROBODOJO_FROZEN_FOCAL_PX, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

ROBODOJO_NATIVE_ACTION_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)

ArrayLike = Union[torch.Tensor, np.ndarray]


def validate_arm_dimensions(left_arm_dim: int, right_arm_dim: int) -> None:
    """Reject any arm dimension other than the ARX-X5's six joints."""

    if int(left_arm_dim) != ROBODOJO_ARM_DIM or int(right_arm_dim) != ROBODOJO_ARM_DIM:
        raise ValueError(f"RoboDojo ARX-X5 arms must have {ROBODOJO_ARM_DIM} joints, got {left_arm_dim}/{right_arm_dim}")


def joint_slot_mask(left_dim: int = ROBODOJO_ARM_DIM, right_dim: int = ROBODOJO_ARM_DIM) -> np.ndarray:
    """Return the Robot80 bool mask of the joint-only contract (joints + grippers; state and action alike)."""

    validate_arm_dimensions(left_dim, right_dim)
    mask = np.zeros(ROBOT80_DIM, dtype=np.bool_)
    mask[LEFT_JOINT_SLOTS.start : LEFT_JOINT_SLOTS.start + left_dim] = True
    mask[LEFT_GRIPPER_SLOT] = True
    mask[RIGHT_JOINT_SLOTS.start : RIGHT_JOINT_SLOTS.start + right_dim] = True
    mask[RIGHT_GRIPPER_SLOT] = True
    mask.setflags(write=False)
    return mask


def target_offsets_ns(steps: int, period_ns: int = ROBODOJO_ACTION_PERIOD_NS) -> tuple[int, ...]:
    """Return the simulator-time offset of each action row: ``(i + 1) * period`` for ``i in range(steps)``."""

    return tuple(int(round((i + 1) * period_ns)) for i in range(int(steps)))


def _joint_vector(value: Any, name: str, dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (dim,):
        raise ValueError(f"{name} must have {dim} entries, got shape {array.shape}")
    if not np.isfinite(array).all() or np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError(f"{name} must be finite and representable in float32")
    return array.astype(np.float32)


def _gripper_opening(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    opening = float(array[0])
    if not np.isfinite(opening):
        raise ValueError(f"{name} must be finite")
    return opening


def closedness_from_opening(opening: float) -> np.float32:
    """Map a RoboDojo normalized gripper opening (1 = open) to Robot80 closedness (1 = closed)."""

    return np.float32(1.0 - min(max(float(opening), 0.0), 1.0))


def opening_from_closedness(closedness: float) -> np.float32:
    """Map Robot80 closedness (1 = closed) to a RoboDojo normalized gripper opening (1 = open)."""

    return np.float32(1.0 - min(max(float(closedness), 0.0), 1.0))


def state80_from_obs(obs_state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``(state80_raw float32[80], mask bool[80])`` from the upstream ``get_obs()["state"]`` dict.

    Joints (radians) fill slots 0..5 / 29..34; grippers 16 / 45 hold ``1 - clip(opening, 0, 1)`` from the
    commanded ``*_ee_joint_state`` opening. Privileged fields (EEF poses, action echo) are never read.
    """

    left_joints = _joint_vector(obs_state["left_arm_joint_state"], "left_arm_joint_state", ROBODOJO_ARM_DIM)
    right_joints = _joint_vector(obs_state["right_arm_joint_state"], "right_arm_joint_state", ROBODOJO_ARM_DIM)
    left_opening = _gripper_opening(obs_state["left_ee_joint_state"], "left_ee_joint_state")
    right_opening = _gripper_opening(obs_state["right_ee_joint_state"], "right_ee_joint_state")

    state80_raw = np.zeros(ROBOT80_DIM, dtype=np.float32)
    state80_raw[LEFT_JOINT_SLOTS.start : LEFT_JOINT_SLOTS.start + ROBODOJO_ARM_DIM] = left_joints
    state80_raw[LEFT_GRIPPER_SLOT] = closedness_from_opening(left_opening)
    state80_raw[RIGHT_JOINT_SLOTS.start : RIGHT_JOINT_SLOTS.start + ROBODOJO_ARM_DIM] = right_joints
    state80_raw[RIGHT_GRIPPER_SLOT] = closedness_from_opening(right_opening)
    return state80_raw, joint_slot_mask(ROBODOJO_ARM_DIM, ROBODOJO_ARM_DIM).copy()


def native_action_from_row(action80_raw_row: np.ndarray) -> dict[str, np.ndarray]:
    """Encode one absolute raw Robot80 row into the RoboDojo ``dual_x5`` action dict (keys in wire order)."""

    row = np.asarray(action80_raw_row, dtype=np.float32).reshape(-1)
    if row.shape != (ROBOT80_DIM,):
        raise ValueError(f"action80 row must have shape ({ROBOT80_DIM},), got {row.shape}")
    native_action: dict[str, np.ndarray] = {}
    arm_contracts = (
        (LEFT_JOINT_SLOTS.start, LEFT_GRIPPER_SLOT, "left"),
        (RIGHT_JOINT_SLOTS.start, RIGHT_GRIPPER_SLOT, "right"),
    )
    for joint_start, gripper_slot, side in arm_contracts:
        native_action[f"{side}_arm_joint_state"] = np.ascontiguousarray(
            row[joint_start : joint_start + ROBODOJO_ARM_DIM], dtype=np.float32
        )
        native_action[f"{side}_ee_joint_state"] = np.array([opening_from_closedness(row[gripper_slot])], dtype=np.float32)
    return native_action


def upstream_actions_from_action80(action80_raw: ArrayLike) -> list[dict[str, np.ndarray]]:
    """Encode an absolute raw ``[K, 80]`` chunk into ``K`` RoboDojo action dicts, one per control tick."""

    chunk = np.asarray(action80_raw.detach().cpu().numpy() if isinstance(action80_raw, torch.Tensor) else action80_raw)
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != ROBOT80_DIM:
        raise ValueError(f"action80 chunk must have shape [K, {ROBOT80_DIM}], got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("action80 chunk must be finite")
    return [native_action_from_row(chunk[k]) for k in range(chunk.shape[0])]


def instruction_from_obs(obs: Mapping[str, Any]) -> str:
    """Return the episode instruction string from ``obs['instruction']`` or the first of ``obs['instructions']``."""

    instruction = obs.get("instruction")
    if instruction is None:
        instructions: Sequence[str] = obs["instructions"]
        instruction = instructions[0]
    return str(instruction)
