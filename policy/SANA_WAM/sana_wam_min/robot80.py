"""Robot80 slot map and the affine state/action normalization contract.

Ports of ``dev/rwm/diffusion/utils/robot80.py`` (slot constants), the joint-target mode
validator from ``dev/rwm/diffusion/data/robot80_action.py``, and the artifact parser plus
``normalize_robot80_affine`` / ``denormalize_robot80_affine`` from
``dev/rwm/diffusion/data/robot80_normalization.py``. Everything runs on CPU in float32, which
is the parity dtype of both the training cache and the evaluation runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import torch

SCHEMA_ID = "rwm_robot80_v4_incoming_motion_observed_gripper_t_human_align"
ROBOT80_ROBOT_SCHEMA_ID = "rwm_robot80_v5_cosmos_incoming_motion_t_robot"
ROTATION_ENCODING = "rot6d_columns"
ROBOT80_DIM = 80

LEFT_JOINT = slice(0, 7)
LEFT_EEF_POSITION = slice(7, 10)
LEFT_EEF_ROTATION = slice(10, 16)
LEFT_GRIPPER = 16
LEFT_DEXTEROUS = slice(17, 29)

RIGHT_JOINT = slice(29, 36)
RIGHT_EEF_POSITION = slice(36, 39)
RIGHT_EEF_ROTATION = slice(39, 45)
RIGHT_GRIPPER = 45
RIGHT_DEXTEROUS = slice(46, 58)

EGO_POSITION = slice(58, 61)
EGO_ROTATION = slice(61, 67)
RESERVED = slice(67, 80)

ROBOT80_JOINT_SLICES = (LEFT_JOINT, RIGHT_JOINT)
ROBOT80_GRIPPER_SLOTS = (LEFT_GRIPPER, RIGHT_GRIPPER)
GRIPPER_SEMANTICS = "closedness_0_open_1_closed"

JOINT_TARGET_ANCHOR_DELTA = "anchor_delta"
JOINT_TARGET_ABSOLUTE = "absolute"
JOINT_TARGET_MODES = (JOINT_TARGET_ANCHOR_DELTA, JOINT_TARGET_ABSOLUTE)

ArrayLike = Union[torch.Tensor, np.ndarray]


def validate_joint_target_mode(joint_target_mode: str) -> str:
    """Return the lowercased joint-target mode; raise ValueError if it is not in JOINT_TARGET_MODES."""

    mode = str(joint_target_mode).lower()
    if mode not in JOINT_TARGET_MODES:
        raise ValueError(f"unsupported joint target mode {mode!r}; expected {JOINT_TARGET_MODES}")
    return mode


@dataclass(frozen=True)
class Normalization:
    """Parsed Robot80 affine normalization artifact: per-slot center/scale for state and action."""

    state_center80: np.ndarray
    state_scale80: np.ndarray
    state_normalization_mask80: np.ndarray
    action_center80: np.ndarray
    action_scale80: np.ndarray
    action_normalization_mask80: np.ndarray
    joint_target_mode: str
    action_representation: Optional[str]
    sha256: Optional[str]
    state_q01_80: Optional[np.ndarray] = None
    state_q99_80: Optional[np.ndarray] = None
    action_q01_80: Optional[np.ndarray] = None
    action_q99_80: Optional[np.ndarray] = None
    action_mode: Optional[str] = None
    normalization_domain_id: Optional[str] = None
    datasets: tuple = ()
    model_fps: Optional[int] = None
    num_frames: Optional[int] = None
    source_path: Optional[str] = None


def _robot80_array(value, name: str, dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (ROBOT80_DIM,):
        raise ValueError(f"{name} must have shape ({ROBOT80_DIM},), got {array.shape}")
    return array


def _parse_block(block: Mapping, kind: str) -> dict:
    """Parse one ``state``/``action`` block into float32 center/scale/q01/q99 and the bool mask."""

    q01 = block.get("q01_80")
    q99 = block.get("q99_80")
    center = block.get("center80")
    scale = block.get("scale80")
    normalization_mask = _robot80_array(block["normalization_mask80"], f"{kind}.normalization_mask80", np.bool_)
    has_explicit_affine = center is not None and scale is not None
    has_explicit_quantiles = q01 is not None and q99 is not None

    if center is None or scale is None:
        if q01 is None or q99 is None:
            raise KeyError(f"{kind} normalization requires center80/scale80 or q01_80/q99_80")
        q01_array = _robot80_array(q01, f"{kind}.q01_80", np.float32)
        q99_array = _robot80_array(q99, f"{kind}.q99_80", np.float32)
        center_array = (q01_array + q99_array) / 2.0
        scale_array = np.maximum((q99_array - q01_array) / 2.0, 1.0e-6)
    else:
        center_array = _robot80_array(center, f"{kind}.center80", np.float32)
        scale_array = _robot80_array(scale, f"{kind}.scale80", np.float32)
        if q01 is not None:
            q01_array = _robot80_array(q01, f"{kind}.q01_80", np.float32)
        else:
            q01_array = center_array - scale_array
        if q99 is not None:
            q99_array = _robot80_array(q99, f"{kind}.q99_80", np.float32)
        else:
            q99_array = center_array + scale_array

    if not np.isfinite(center_array).all():
        raise ValueError(f"{kind}.center80 must be finite")
    if not np.isfinite(scale_array).all() or not np.all(scale_array > 0):
        raise ValueError(f"{kind}.scale80 must be finite and positive")
    if not np.isfinite(q01_array).all() or not np.isfinite(q99_array).all():
        raise ValueError(f"{kind}.q01_80/q99_80 must be finite")
    if np.any(q99_array[normalization_mask] < q01_array[normalization_mask]):
        raise ValueError(f"{kind}.q99_80 must not be smaller than q01_80")
    if has_explicit_affine and has_explicit_quantiles:
        expected_center = (q01_array + q99_array) / 2.0
        expected_scale = np.maximum((q99_array - q01_array) / 2.0, 1.0e-6)
        if not np.allclose(center_array[normalization_mask], expected_center[normalization_mask]) or not np.allclose(
            scale_array[normalization_mask], expected_scale[normalization_mask]
        ):
            raise ValueError(f"{kind}.center80/scale80 disagree with q01_80/q99_80")

    return {
        f"{kind}_q01_80": q01_array,
        f"{kind}_q99_80": q99_array,
        f"{kind}_center80": center_array,
        f"{kind}_scale80": scale_array,
        f"{kind}_normalization_mask80": normalization_mask,
    }


def parse_normalization_artifact(
    artifact: Mapping,
    source_path: Optional[str] = None,
    sha256: Optional[str] = None,
) -> Normalization:
    """Parse a Robot80 affine normalization artifact mapping (``parse_robot80_affine_normalization`` port)."""

    joint_target_mode = validate_joint_target_mode(artifact["joint_target_mode"])
    blocks = {}
    for kind in ("state", "action"):
        blocks.update(_parse_block(artifact[kind], kind))
    model_fps = artifact.get("model_fps")
    num_frames = artifact.get("num_frames")
    return Normalization(
        state_center80=blocks["state_center80"],
        state_scale80=blocks["state_scale80"],
        state_normalization_mask80=blocks["state_normalization_mask80"],
        action_center80=blocks["action_center80"],
        action_scale80=blocks["action_scale80"],
        action_normalization_mask80=blocks["action_normalization_mask80"],
        joint_target_mode=joint_target_mode,
        action_representation=artifact.get("action_representation"),
        sha256=sha256,
        state_q01_80=blocks["state_q01_80"],
        state_q99_80=blocks["state_q99_80"],
        action_q01_80=blocks["action_q01_80"],
        action_q99_80=blocks["action_q99_80"],
        action_mode=artifact.get("action_mode"),
        normalization_domain_id=artifact.get("normalization_domain_id"),
        datasets=tuple(artifact.get("datasets", ())),
        model_fps=None if model_fps is None else int(model_fps),
        num_frames=None if num_frames is None else int(num_frames),
        source_path=None if source_path is None else str(source_path),
    )


def load_normalization(path: Union[str, Path], expected_sha256: Optional[str] = None) -> Normalization:
    """Load the artifact json from disk, record its sha256, and optionally pin it against ``expected_sha256``."""

    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower().removeprefix("sha256:"):
        raise ValueError(f"normalization artifact sha256 mismatch: {path} has {digest}, expected {expected_sha256}")
    return parse_normalization_artifact(json.loads(raw.decode("utf-8")), source_path=str(path), sha256=digest)


def _affine_robot80_tensors(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    center80: ArrayLike,
    scale80: ArrayLike,
    normalization_mask80: ArrayLike,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.shape != valid_mask.shape or values.shape[-1] != ROBOT80_DIM:
        raise ValueError(
            "values and valid_mask must have the same shape ending in 80, got "
            f"{tuple(values.shape)} and {tuple(valid_mask.shape)}"
        )

    center = torch.as_tensor(center80, dtype=values.dtype, device=values.device)
    scale = torch.as_tensor(scale80, dtype=values.dtype, device=values.device)
    normalization_mask = torch.as_tensor(normalization_mask80, dtype=torch.bool, device=values.device)
    for tensor, name in ((center, "center80"), (scale, "scale80"), (normalization_mask, "normalization_mask80")):
        if tensor.shape != (ROBOT80_DIM,):
            raise ValueError(f"{name} must have shape ({ROBOT80_DIM},), got {tuple(tensor.shape)}")
    if not torch.isfinite(center).all():
        raise ValueError("center80 must be finite")
    if not torch.isfinite(scale).all() or not torch.all(scale > 0):
        raise ValueError("scale80 must be finite and positive")
    return center, scale, normalization_mask


def normalize_robot80_affine(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    center80: ArrayLike,
    scale80: ArrayLike,
    normalization_mask80: ArrayLike,
) -> torch.Tensor:
    """Apply an 80D affine map, keep identity slots, and zero invalid slots."""

    center, scale, normalization_mask = _affine_robot80_tensors(values, valid_mask, center80, scale80, normalization_mask80)
    normalized = torch.where(normalization_mask, (values - center) / scale, values)
    return torch.where(valid_mask, normalized, torch.zeros_like(values))


def denormalize_robot80_affine(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    center80: ArrayLike,
    scale80: ArrayLike,
    normalization_mask80: ArrayLike,
) -> torch.Tensor:
    """Invert the 80D affine map, keep identity slots, and zero invalid slots."""

    center, scale, normalization_mask = _affine_robot80_tensors(values, valid_mask, center80, scale80, normalization_mask80)
    denormalized = torch.where(normalization_mask, values * scale + center, values)
    return torch.where(valid_mask, denormalized, torch.zeros_like(values))


def _fp32_pair(values: ArrayLike, valid_mask: ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
    values_t = torch.as_tensor(values).to(torch.float32)
    mask_t = torch.as_tensor(valid_mask).to(torch.bool)
    if not torch.isfinite(values_t).all():
        raise ValueError("robot80 values must be finite")
    return values_t, mask_t.to(values_t.device)


def _finite_result(result: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(result).all():
        raise ValueError("robot80 affine result must be finite")
    return result.contiguous()


def normalize_state(values80: ArrayLike, valid_mask80: ArrayLike, norm: Normalization) -> torch.Tensor:
    """Map a raw ``[80]`` state row into the model domain with the artifact's state statistics (float32)."""

    values, mask = _fp32_pair(values80, valid_mask80)
    return _finite_result(
        normalize_robot80_affine(values, mask, norm.state_center80, norm.state_scale80, norm.state_normalization_mask80)
    )


def denormalize_state(values80: ArrayLike, valid_mask80: ArrayLike, norm: Normalization) -> torch.Tensor:
    """Inverse of ``normalize_state`` (float32)."""

    values, mask = _fp32_pair(values80, valid_mask80)
    return _finite_result(
        denormalize_robot80_affine(values, mask, norm.state_center80, norm.state_scale80, norm.state_normalization_mask80)
    )


def normalize_action(values: ArrayLike, valid_mask: ArrayLike, norm: Normalization) -> torch.Tensor:
    """Map raw ``[K, 80]`` action rows into the model domain with the artifact's action statistics (float32)."""

    values_t, mask = _fp32_pair(values, valid_mask)
    return _finite_result(
        normalize_robot80_affine(values_t, mask, norm.action_center80, norm.action_scale80, norm.action_normalization_mask80)
    )


def denormalize_action(values: ArrayLike, valid_mask: ArrayLike, norm: Normalization) -> torch.Tensor:
    """Map model-domain ``[K, 80]`` action rows back to raw units with the artifact's action statistics (float32)."""

    values_t, mask = _fp32_pair(values, valid_mask)
    return _finite_result(
        denormalize_robot80_affine(values_t, mask, norm.action_center80, norm.action_scale80, norm.action_normalization_mask80)
    )
