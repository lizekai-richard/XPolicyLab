"""Strict checkpoint adapter for the inference-only policy specialization."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


def strip_unmodeled_state(
    state_dict: Mapping[str, torch.Tensor],
    *,
    input_size: int,
    hidden_size: int,
    model_max_length: int,
    caption_channels: int,
) -> dict[str, torch.Tensor]:
    """Remove the three checkpoint tensors the mirror does not model.

    ``pos_embed`` and ``y_embedder.y_embedding`` are never read by the
    wan-rope/no-CFG eval forward; ``plucker_embed.weight`` is the camera
    channel's projection, which the mirror does not model (the camera gate
    must be off).  Shape drift on any of them is rejected.
    """

    output = dict(state_dict)
    expected = {
        "pos_embed": (1, input_size * input_size, hidden_size),
        "y_embedder.y_embedding": (model_max_length, caption_channels),
        "plucker_embed.weight": (hidden_size, 6, 1, 1, 1),
    }
    for key, shape in expected.items():
        tensor = output.pop(key, None)
        if tensor is None:
            continue
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"legacy checkpoint buffer {key!r} has shape "
                f"{tuple(tensor.shape)}, expected {shape}"
            )
    return output


def load_policy_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Strict-load every inference-active tensor after validated legacy removal."""

    config = model.policy_config
    active = strip_unmodeled_state(
        state_dict,
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        model_max_length=config.model_max_length,
        caption_channels=config.caption_channels,
    )
    incompatible = model.load_state_dict(active, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict policy load failed: {incompatible}")


__all__ = ["load_policy_state_dict", "strip_unmodeled_state"]
