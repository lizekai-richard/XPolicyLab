"""Model-domain action chunk to raw absolute Robot80 post-processing.

Torch ports of ``_reconstruct_absolute_joint_targets`` / ``_clip_predicted_gripper_closedness`` and
the three-step body of ``chunk_prediction_from_model_action`` from
``dev/rwm/simulation_evaluation/policy_runtime/in_process.py``, plus the joint-limit gate of
``RoboDojoJointActionCodec.encode`` (``robodojo/client/action_codec.py``). All arithmetic is float32.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

from .robot80 import (
    JOINT_TARGET_ANCHOR_DELTA,
    ROBOT80_DIM,
    ROBOT80_GRIPPER_SLOTS,
    ROBOT80_JOINT_SLICES,
    Normalization,
    denormalize_action,
)

ArrayLike = Union[torch.Tensor, np.ndarray]

JOINT_LIMIT_MODES = ("clip", "reject")


def _fp32(values: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(values).to(torch.float32)


def _bool(mask: ArrayLike, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(mask).to(device=device, dtype=torch.bool)


def reconstruct_absolute_joints(
    action80_raw: ArrayLike,
    action_mask: ArrayLike,
    anchor_state80_raw: ArrayLike,
    anchor_mask80: ArrayLike,
) -> torch.Tensor:
    """Convert one window-anchored joint-delta chunk ``[K, 80]`` to absolute joints (``anchor + delta``).

    Every row is anchored to the same raw observation state. Only the two joint slices are touched,
    and inside them only slots active in ``action_mask``; grippers and every other slot pass through.
    """

    action = _fp32(action80_raw)
    mask = _bool(action_mask, action.device)
    anchor = _fp32(anchor_state80_raw).to(action.device)
    anchor_mask = _bool(anchor_mask80, action.device)
    if action.shape != mask.shape or action.shape[-1] != ROBOT80_DIM or anchor.shape != (ROBOT80_DIM,):
        raise ValueError(
            f"expected action [K, 80] with matching mask and anchor [80], got {tuple(action.shape)}, "
            f"{tuple(mask.shape)}, {tuple(anchor.shape)}"
        )

    reconstructed = action.clone()
    for joint_slice in ROBOT80_JOINT_SLICES:
        active = mask[:, joint_slice]
        state_active = anchor_mask[joint_slice]
        missing_anchor = active & ~state_active[None, :]
        if bool(missing_anchor.any()):
            row, offset = torch.nonzero(missing_anchor)[0].tolist()
            slot = joint_slice.start + int(offset)
            raise ValueError(
                f"cannot reconstruct absolute joint target: active action slot {slot} "
                f"has no active anchor-state value (row {int(row)})"
            )
        delta = action[:, joint_slice]
        reconstructed[:, joint_slice] = torch.where(active, anchor[joint_slice][None, :] + delta, delta)
    if not torch.isfinite(reconstructed).all():
        raise ValueError("reconstructed absolute action80_raw must be finite")
    return reconstructed


def clip_gripper_closedness(action80: ArrayLike, mask: ArrayLike) -> torch.Tensor:
    """Clamp the two gripper slots (16, 45) to closedness ``[0, 1]`` where the mask is valid; nothing else changes."""

    action = _fp32(action80).clone()
    valid = _bool(mask, action.device)
    for slot in ROBOT80_GRIPPER_SLOTS:
        original = action[..., slot]
        action[..., slot] = torch.where(valid[..., slot], original.clamp(0.0, 1.0), original)
    return action


def apply_joint_limits(
    action80: ArrayLike,
    joint_slots: Sequence[int],
    lower: Sequence[float],
    upper: Sequence[float],
    mode: str = "reject",
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Gate joint targets against per-slot ``[lower, upper]`` limits (float64), rejecting or clipping.

    ``mode="reject"`` raises on the first out-of-range slot (evaluation-faithful). ``mode="clip"``
    clamps to float32 bounds nudged one ulp inward whenever the float32 cast moved them outward, and
    returns the sorted Robot80 slots that were clipped.
    """

    if mode not in JOINT_LIMIT_MODES:
        raise ValueError(f"unsupported joint limit mode {mode!r}; expected {JOINT_LIMIT_MODES}")
    action = _fp32(action80).clone()
    slots = np.asarray(tuple(int(s) for s in joint_slots), dtype=np.intp)
    lower64 = np.asarray(lower, dtype=np.float64)
    upper64 = np.asarray(upper, dtype=np.float64)
    if lower64.shape != slots.shape or upper64.shape != slots.shape:
        raise ValueError(f"lower/upper must match joint_slots shape {slots.shape}, got {lower64.shape}, {upper64.shape}")
    if not (np.isfinite(lower64).all() and np.isfinite(upper64).all() and np.all(lower64 <= upper64)):
        raise ValueError("joint limits must be finite with lower <= upper")

    joint_values = action[..., torch.as_tensor(slots, device=action.device)].detach().cpu().numpy().astype(np.float64)
    outside = (joint_values < lower64) | (joint_values > upper64)
    if not np.any(outside):
        return action, ()
    outside_slots = np.flatnonzero(np.any(outside.reshape(-1, slots.shape[0]), axis=0))
    clipped_joint_slots = tuple(sorted(int(slots[i]) for i in outside_slots))
    if mode == "reject":
        index = int(outside_slots[0])
        raise ValueError(
            f"action80_raw joint slot {int(slots[index])} is outside configured limits "
            f"[{lower64[index]}, {upper64[index]}]; clipping is forbidden"
        )

    lower32 = lower64.astype(np.float32)
    upper32 = upper64.astype(np.float32)
    lower32 = np.where(lower32.astype(np.float64) < lower64, np.nextafter(lower32, np.float32(np.inf)), lower32)
    upper32 = np.where(upper32.astype(np.float64) > upper64, np.nextafter(upper32, np.float32(-np.inf)), upper32)
    slot_index = torch.as_tensor(slots, device=action.device)
    lo = torch.as_tensor(lower32, dtype=torch.float32, device=action.device)
    hi = torch.as_tensor(upper32, dtype=torch.float32, device=action.device)
    action[..., slot_index] = torch.minimum(torch.maximum(action[..., slot_index], lo), hi)
    return action, clipped_joint_slots


def model_action_to_absolute(
    action80_model: ArrayLike,
    mask: ArrayLike,
    norm: Normalization,
    anchor_state80_raw: ArrayLike,
    anchor_mask: ArrayLike,
) -> torch.Tensor:
    """Denormalize -> add the raw anchor joints (anchor_delta only) -> clip grippers, in that order.

    Mirrors ``chunk_prediction_from_model_action``; the joint-target mode is read from the artifact,
    never from configuration, and the anchor must be the raw (un-normalized) observation state.
    """

    action80_raw = denormalize_action(action80_model, mask, norm)
    if norm.joint_target_mode == JOINT_TARGET_ANCHOR_DELTA:
        action80_raw = reconstruct_absolute_joints(action80_raw, mask, anchor_state80_raw, anchor_mask)
    return clip_gripper_closedness(action80_raw, mask)
