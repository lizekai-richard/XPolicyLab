"""Robot-base end-effector support for the RoboDojo ARX-X5 adapter (Sana's ``robot_base_eef`` action mode).

Training convention of that mode (``dev/rwm/diffusion/data/robot80_action.py::pack_robot_base_eef_state80``,
``data_format/robodojo.py``): per arm, the pose of the URDF flange ``link6`` expressed in the arm's own
``base_link`` frame, obtained by forward kinematics of the JOINT DRIVE TARGETS (``pose_source
joint_drive_target_urdf_fk``), with the flange axes right-multiplied by the mechanical-E remap
(``_STANDARD_E_ROWS``: +X approach, +Y down, +Z left; rotation only, no translation) and the rotation encoded as
column rot6d (``dev/rwm/diffusion/utils/geometry.py``). Targets are relative to the window anchor row:
``p_t - p_anchor`` and ``R_t @ R_anchor^T`` -> rot6d (``_current_anchor_action``, identity camera). Grippers stay
absolute and joints keep their anchor_delta treatment.

This module reproduces that chain at deployment: a numpy URDF forward kinematics (port of
``dev/rwm/diffusion/data/eef/robot_base_eef_kinematics.py``) on the vendored ``assets/robotwin2_arx_x5.urdf``,
the E remap, the Robot80 state-slot packing, the absolute-pose reconstruction of a predicted chunk, and the
conversion of an absolute E pose back into the evaluator's ``left_ee_pose`` / ``right_ee_pose`` (``link6`` in the
env-relative world frame, ``[x, y, z, qw, qx, qy, qz]``) through the ARX-X5 root poses of
``env_cfg/robot/dual_x5.yml``.

Verified offline on 2026-09-13: FK(recorded joint state) composed with those root poses reproduces the dataset's
recorded ``state/*_ee_poses`` to 0.14 mm / 0.017 deg on insert_key and stack_bowls episodes (both arms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
from xml.etree import ElementTree

import numpy as np
import torch

from .robot80 import (
    LEFT_EEF_POSITION,
    LEFT_EEF_ROTATION,
    LEFT_GRIPPER,
    LEFT_JOINT,
    RIGHT_EEF_POSITION,
    RIGHT_EEF_ROTATION,
    RIGHT_GRIPPER,
    RIGHT_JOINT,
    ROBOT80_DIM,
)

ArrayLike = Union[torch.Tensor, np.ndarray]

ARM_DIM = 6
PACKAGED_URDF_PATH = Path(__file__).resolve().parent / "assets" / "robotwin2_arx_x5.urdf"
URDF_BASE_LINK = "base_link"
URDF_FLANGE_LINK = "link6"
URDF_ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))

# T_link6_E: coordinates of the mechanical E frame expressed in the URDF flange frame (rotation only).
# Sana: ``_STANDARD_E_ROWS`` / ``EefAxisTransform.source_eef_from_eef``; ``T_base_E = T_base_link6 @ T_link6_E``.
LINK6_FROM_E = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
)
E_FROM_LINK6 = LINK6_FROM_E.T.copy()

# ARX-X5 root poses of the RoboDojo evaluator (env_cfg/robot/dual_x5.yml ``default_root_pos`` / ``default_root_rot``,
# [x, y, z] + [qw, qx, qy, qz] in the env-relative world frame) = the constant world-to-base transform between the
# evaluator's ``*_ee_pose`` observation and the training corpus' base-frame FK.
ARX_X5_ROOT_POSES: dict[str, tuple[float, ...]] = {
    "left": (-0.3, -0.45, 0.765, 0.707, 0.0, 0.0, 0.707),
    "right": (0.3, -0.45, 0.765, 0.707, 0.0, 0.0, 0.707),
}

# Per side: (name, joint slice of the 6 arm joints, EEF position slice, EEF rotation slice, gripper slot).
ARM_FIELDS = (
    ("left", slice(LEFT_JOINT.start, LEFT_JOINT.start + ARM_DIM), LEFT_EEF_POSITION, LEFT_EEF_ROTATION, LEFT_GRIPPER),
    ("right", slice(RIGHT_JOINT.start, RIGHT_JOINT.start + ARM_DIM), RIGHT_EEF_POSITION, RIGHT_EEF_ROTATION, RIGHT_GRIPPER),
)
EEF_SLOTS = tuple(range(LEFT_EEF_POSITION.start, LEFT_EEF_ROTATION.stop)) + tuple(
    range(RIGHT_EEF_POSITION.start, RIGHT_EEF_ROTATION.stop)
)
SIDES = ("left", "right")


# -- rotations ----------------------------------------------------------------------------------------------


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_transform(rotation: Optional[np.ndarray] = None, translation: Optional[np.ndarray] = None) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    if rotation is not None:
        output[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if translation is not None:
        output[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return output


def invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    rotation_t = matrix[:3, :3].T
    return make_transform(rotation_t, -rotation_t @ matrix[:3, 3])


def quat_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Unit-normalize a ``[w, x, y, z]`` quaternion and return its rotation matrix."""

    w, x, y, z = (float(v) for v in np.asarray(quaternion, dtype=np.float64).reshape(4))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid wxyz quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit ``[w, x, y, z]`` quaternion with a non-negative w (Shepperd's method)."""

    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def pose_to_transform(pose7: Sequence[float]) -> np.ndarray:
    """``[x, y, z, qw, qx, qy, qz]`` -> 4x4."""

    values = np.asarray(pose7, dtype=np.float64).reshape(7)
    if not np.isfinite(values).all():
        raise ValueError("pose must be finite")
    return make_transform(quat_wxyz_to_matrix(values[3:]), values[:3])


def transform_to_pose(transform: np.ndarray) -> np.ndarray:
    """4x4 -> ``[x, y, z, qw, qx, qy, qz]`` (float64)."""

    matrix = np.asarray(transform, dtype=np.float64)
    return np.concatenate([matrix[:3, 3], matrix_to_quat_wxyz(matrix[:3, :3])])


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Column rot6d ``[R[:, 0], R[:, 1]]`` (Sana ``geometry.matrix_to_rot6d``)."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotation shape (..., 3, 3), got {matrix.shape}")
    return np.swapaxes(matrix[..., :, :2], -1, -2).reshape(*matrix.shape[:-2], 6)


def rot6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt decode of column rot6d (Sana ``geometry.rot6d_to_matrix``)."""

    encoded = np.asarray(rotation_6d, dtype=np.float64)
    if encoded.shape[-1] != 6:
        raise ValueError(f"expected rot6d shape (..., 6), got {encoded.shape}")
    c0 = encoded[..., :3]
    c1 = encoded[..., 3:]
    c0 = c0 / np.linalg.norm(c0, axis=-1, keepdims=True).clip(min=1e-12)
    c1 = c1 - np.sum(c0 * c1, axis=-1, keepdims=True) * c0
    c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True).clip(min=1e-12)
    c2 = np.cross(c0, c1)
    return np.stack((c0, c1, c2), axis=-1)


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Angle between two rotation matrices in degrees."""

    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = (float(np.trace(relative)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


# -- URDF forward kinematics (numpy port of robot_base_eef_kinematics.py) ------------------------------------------


def _vector(text: Optional[str], length: int, default: Sequence[float]) -> np.ndarray:
    values = np.asarray(default if text is None else [float(v) for v in text.split()], dtype=np.float64)
    if values.shape != (length,):
        raise ValueError(f"expected {length} values, got {values.tolist()}")
    return values


def _origin_transform(element: Optional[ElementTree.Element]) -> np.ndarray:
    if element is None:
        return np.eye(4, dtype=np.float64)
    xyz = _vector(element.get("xyz"), 3, (0.0, 0.0, 0.0))
    roll, pitch, yaw = _vector(element.get("rpy"), 3, (0.0, 0.0, 0.0))
    return make_transform(rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll), xyz)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("a movable URDF joint has a zero axis")
    x, y, z = value / norm
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray

    def motion(self, position: float) -> np.ndarray:
        if self.joint_type == "fixed":
            return np.eye(4, dtype=np.float64)
        if self.joint_type in {"revolute", "continuous"}:
            return make_transform(_axis_angle(self.axis, float(position)))
        if self.joint_type == "prismatic":
            axis = self.axis / np.linalg.norm(self.axis)
            return make_transform(translation=axis * float(position))
        raise ValueError(f"unsupported URDF joint type {self.joint_type!r} for {self.name!r}")


class UrdfModel:
    """Joint tree of one URDF file with chain forward kinematics (rotation/translation only, no collision data)."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).expanduser().resolve()
        root = ElementTree.fromstring(self.path.read_bytes())
        self.joints_by_name: dict[str, UrdfJoint] = {}
        self.parent_joint_by_child: dict[str, UrdfJoint] = {}
        for element in root.findall("joint"):
            name, joint_type = element.get("name"), element.get("type")
            parent_element, child_element = element.find("parent"), element.find("child")
            if not name or not joint_type or parent_element is None or child_element is None:
                raise ValueError("URDF joint is missing name/type/parent/child")
            parent, child = parent_element.get("link"), child_element.get("link")
            if not parent or not child:
                raise ValueError(f"URDF joint {name!r} has an empty link")
            axis_element = element.find("axis")
            joint = UrdfJoint(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=_origin_transform(element.find("origin")),
                axis=_vector(None if axis_element is None else axis_element.get("xyz"), 3, (1.0, 0.0, 0.0)),
            )
            if name in self.joints_by_name:
                raise ValueError(f"duplicate URDF joint name {name!r}")
            if child in self.parent_joint_by_child:
                raise ValueError(f"URDF link {child!r} has multiple parents")
            self.joints_by_name[name] = joint
            self.parent_joint_by_child[child] = joint
        self._chains: dict[tuple[str, str], tuple[UrdfJoint, ...]] = {}

    def chain(self, base_link: str, target_link: str) -> tuple[UrdfJoint, ...]:
        key = (base_link, target_link)
        if key not in self._chains:
            reverse: list[UrdfJoint] = []
            current, visited = target_link, set()
            while current != base_link:
                if current in visited:
                    raise ValueError(f"cycle in URDF chain {base_link!r} -> {target_link!r}")
                visited.add(current)
                joint = self.parent_joint_by_child.get(current)
                if joint is None:
                    raise KeyError(f"no URDF chain {base_link!r} -> {target_link!r}; stopped at {current!r}")
                reverse.append(joint)
                current = joint.parent
            self._chains[key] = tuple(reversed(reverse))
        return self._chains[key]

    def transform_between(self, base_link: str, target_link: str, joint_positions: Mapping[str, float]) -> np.ndarray:
        output = np.eye(4, dtype=np.float64)
        for joint in self.chain(base_link, target_link):
            output = output @ joint.origin @ joint.motion(float(joint_positions.get(joint.name, 0.0)))
        return output


class ArxX5Kinematics:
    """Flange (``link6``) pose in the arm's ``base_link`` frame from the six ARX-X5 joint angles (radians)."""

    def __init__(self, urdf_path: Optional[Union[str, Path]] = None):
        self.model = UrdfModel(PACKAGED_URDF_PATH if urdf_path is None else urdf_path)
        required = {j.name for j in self.model.chain(URDF_BASE_LINK, URDF_FLANGE_LINK) if j.joint_type != "fixed"}
        missing = sorted(required.difference(URDF_ARM_JOINT_NAMES))
        if missing:
            raise ValueError(f"URDF chain {URDF_BASE_LINK}->{URDF_FLANGE_LINK} needs joints {missing}")

    def link6_in_base(self, joints6: Sequence[float]) -> np.ndarray:
        values = np.asarray(joints6, dtype=np.float64).reshape(-1)
        if values.shape != (ARM_DIM,) or not np.isfinite(values).all():
            raise ValueError(f"expected {ARM_DIM} finite joint angles, got {values}")
        positions = {name: float(v) for name, v in zip(URDF_ARM_JOINT_NAMES, values)}
        return self.model.transform_between(URDF_BASE_LINK, URDF_FLANGE_LINK, positions)

    def e_in_base(self, joints6: Sequence[float]) -> np.ndarray:
        """``T_base_E`` = FK flange pose right-multiplied by the E remap (what the training corpus packs)."""

        return self.link6_in_base(joints6) @ LINK6_FROM_E


# -- Robot80 packing ------------------------------------------------------------------------------------------------


def eef_state_slot_mask(include_eef: bool) -> np.ndarray:
    """Bool[80] of the served slots: 6 joints + gripper per arm, plus the EEF position/rot6d slots when ``include_eef``."""

    mask = np.zeros(ROBOT80_DIM, dtype=np.bool_)
    for _, joint_slice, position_slice, rotation_slice, gripper_slot in ARM_FIELDS:
        mask[joint_slice] = True
        mask[gripper_slot] = True
        if include_eef:
            mask[position_slice] = True
            mask[rotation_slice] = True
    return mask


def e_pose_slots(transform_base_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position[3], rot6d[6]) of a ``T_base_E`` pose, as the training state row stores them."""

    matrix = np.asarray(transform_base_e, dtype=np.float64)
    return matrix[:3, 3].astype(np.float32), matrix_to_rot6d(matrix[:3, :3]).astype(np.float32)


def fill_eef_state_slots(state80: np.ndarray, mask80: np.ndarray, kinematics: ArxX5Kinematics) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of ``(state80, mask80)`` with each arm's EEF slots set from the FK of ITS OWN JOINT SLOTS.

    The training state row holds ``FK(joint drive target)`` per arm; feeding the row's joints (measured, or the
    last command under ``anchor_source: last_command``) through the same URDF chain reproduces that pairing.
    """

    state = np.array(state80, dtype=np.float32, copy=True).reshape(ROBOT80_DIM)
    mask = np.array(mask80, dtype=np.bool_, copy=True).reshape(ROBOT80_DIM)
    for side, joint_slice, position_slice, rotation_slice, _ in ARM_FIELDS:
        if not mask[joint_slice].all():
            raise ValueError(f"{side} arm joints are not all valid; cannot derive its EEF pose")
        position, rot6d = e_pose_slots(kinematics.e_in_base(state[joint_slice]))
        state[position_slice] = position
        state[rotation_slice] = rot6d
        mask[position_slice] = True
        mask[rotation_slice] = True
    return state, mask


def reconstruct_absolute_eef(
    action80_raw: ArrayLike, action_mask: ArrayLike, anchor_state80_raw: ArrayLike, anchor_mask80: ArrayLike
) -> torch.Tensor:
    """Turn the anchor-relative EEF slots of a raw ``[K, 80]`` chunk into absolute base-frame E poses.

    Per arm and row (where the position/rotation slots are active): ``p = p_anchor + delta_p`` and
    ``R = rot6d_to_matrix(delta_rot6d) @ R_anchor`` (the inverse of ``R_t @ R_anchor^T``), written back as
    position and column rot6d. Every other slot passes through unchanged (joints are handled by
    ``actions.reconstruct_absolute_joints``).
    """

    rows = torch.as_tensor(action80_raw).detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    mask = np.asarray(torch.as_tensor(action_mask).detach().cpu().numpy(), dtype=np.bool_)
    anchor = np.asarray(torch.as_tensor(anchor_state80_raw).detach().cpu().numpy(), dtype=np.float64).reshape(ROBOT80_DIM)
    anchor_mask = np.asarray(torch.as_tensor(anchor_mask80).detach().cpu().numpy(), dtype=np.bool_).reshape(ROBOT80_DIM)
    if rows.ndim != 2 or rows.shape[1] != ROBOT80_DIM or mask.shape != rows.shape:
        raise ValueError(f"expected action [K, 80] with a matching mask, got {rows.shape} and {mask.shape}")
    for side, _, position_slice, rotation_slice, _ in ARM_FIELDS:
        active = mask[:, position_slice].all(axis=1) & mask[:, rotation_slice].all(axis=1)
        if not active.any():
            continue
        if not (anchor_mask[position_slice].all() and anchor_mask[rotation_slice].all()):
            raise ValueError(f"cannot reconstruct the {side} EEF pose: the anchor state has no valid EEF slots")
        p_anchor = anchor[position_slice]
        r_anchor = rot6d_to_matrix(anchor[rotation_slice])
        for k in np.flatnonzero(active):
            rows[k, position_slice] = (p_anchor + rows[k, position_slice].astype(np.float64)).astype(np.float32)
            rotation = rot6d_to_matrix(rows[k, rotation_slice].astype(np.float64)) @ r_anchor
            rows[k, rotation_slice] = matrix_to_rot6d(rotation).astype(np.float32)
    if not np.isfinite(rows).all():
        raise ValueError("reconstructed absolute EEF rows must be finite")
    return torch.from_numpy(rows)


# -- evaluator interface ---------------------------------------------------------------------------------------------


def root_transforms(root_poses: Optional[Mapping[str, Sequence[float]]] = None) -> dict[str, np.ndarray]:
    """``T_world_base`` per side from ``{side: [x, y, z, qw, qx, qy, qz]}`` (default: the ARX-X5 poses)."""

    poses = dict(ARX_X5_ROOT_POSES if root_poses is None else root_poses)
    missing = [s for s in SIDES if s not in poses]
    if missing:
        raise ValueError(f"robot root poses are missing side(s) {missing}")
    return {side: pose_to_transform(poses[side]) for side in SIDES}


def world_link6_pose_from_slots(row80: np.ndarray, side: str, world_from_base: np.ndarray) -> np.ndarray:
    """Absolute E slots of one arm -> ``link6`` pose in the env-relative world frame, ``[x, y, z, qw, qx, qy, qz]``."""

    _, _, position_slice, rotation_slice, _ = next(f for f in ARM_FIELDS if f[0] == side)
    row = np.asarray(row80, dtype=np.float64).reshape(ROBOT80_DIM)
    base_from_e = make_transform(rot6d_to_matrix(row[rotation_slice]), row[position_slice])
    return transform_to_pose(world_from_base @ base_from_e @ E_FROM_LINK6)


def native_ee_actions_from_action80(action80_abs: ArrayLike, world_from_base: Mapping[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    """Absolute ``[K, 80]`` chunk -> K RoboDojo ``ee`` action dicts: ``{left,right}_ee_pose`` (7) + ``{left,right}_ee_joint_state`` (1)."""

    from .robodojo_io import opening_from_closedness

    chunk = np.asarray(action80_abs.detach().cpu().numpy() if isinstance(action80_abs, torch.Tensor) else action80_abs, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != ROBOT80_DIM or not np.isfinite(chunk).all():
        raise ValueError(f"action80 chunk must be a finite [K, {ROBOT80_DIM}] array, got {chunk.shape}")
    actions = []
    for row in chunk:
        action: dict[str, np.ndarray] = {}
        for side, _, _, _, gripper_slot in ARM_FIELDS:
            action[f"{side}_ee_pose"] = np.ascontiguousarray(world_link6_pose_from_slots(row, side, world_from_base[side]), dtype=np.float32)
            action[f"{side}_ee_joint_state"] = np.array([opening_from_closedness(row[gripper_slot])], dtype=np.float32)
        actions.append(action)
    return actions


def observed_link6_discrepancy(
    obs_state: Mapping[str, Any], state80_fk: np.ndarray, world_from_base: Mapping[str, np.ndarray]
) -> dict[str, tuple[float, float]]:
    """Per side ``(position error m, rotation error deg)`` between the evaluator's ``*_ee_pose`` observation and the
    FK-derived pose in ``state80_fk`` (whose EEF slots hold ``T_base_E`` of the observed joints). A large value means
    the root pose / URDF / axis conventions do not match the simulator, so the EEF state fed to the model is wrong."""

    result = {}
    for side, _, _, _, _ in ARM_FIELDS:
        observed = obs_state.get(f"{side}_ee_pose")
        if observed is None:
            continue
        observed_t = pose_to_transform(observed)
        predicted_t = pose_to_transform(world_link6_pose_from_slots(state80_fk, side, world_from_base[side]))
        result[side] = (
            float(np.linalg.norm(observed_t[:3, 3] - predicted_t[:3, 3])),
            rotation_angle_deg(observed_t[:3, :3], predicted_t[:3, :3]),
        )
    return result


__all__ = [
    "ARM_DIM",
    "ARM_FIELDS",
    "ARX_X5_ROOT_POSES",
    "ArxX5Kinematics",
    "EEF_SLOTS",
    "E_FROM_LINK6",
    "LINK6_FROM_E",
    "PACKAGED_URDF_PATH",
    "UrdfModel",
    "e_pose_slots",
    "eef_state_slot_mask",
    "fill_eef_state_slots",
    "matrix_to_quat_wxyz",
    "matrix_to_rot6d",
    "native_ee_actions_from_action80",
    "observed_link6_discrepancy",
    "pose_to_transform",
    "quat_wxyz_to_matrix",
    "reconstruct_absolute_eef",
    "root_transforms",
    "rot6d_to_matrix",
    "rotation_angle_deg",
    "transform_to_pose",
    "world_link6_pose_from_slots",
]
