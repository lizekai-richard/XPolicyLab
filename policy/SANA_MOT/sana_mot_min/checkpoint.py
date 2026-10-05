"""MoT policy construction and strict checkpoint loading from disk.

The trained checkpoint is an accelerate FSDP full state dict (``model/pytorch_model_fsdp.bin``) in the paired layout
(``video_dit.*``, ``action_dit.*``, ``blocks.<i>.{video_block,action_block}.*``, plus ``context_embedder.*`` in the
current text layout). The loading order is: read the payload memory-mapped (:func:`read_mot_state_dict`), detect its
text layout (``mot_model.checkpoint.detect_context_layout``), resolve the architecture from the training yaml with that
layout, build the model at the serving dtype, then :func:`load_mot_weights` -- which verifies the softmax placement of
both experts and the state-conditioning layout against the config before a single tensor is copied, strips the
documented buffers and strict-loads the rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Union

import torch

from sana_wam_min.checkpoint import CHECKPOINT_RELATIVE_PATH, resolve_checkpoint_file

from .mot_model.checkpoint import (
    checkpoint_layer_layout,
    detect_context_layout,
    load_mot_state_dict,
    state_as_context_in_checkpoint,
    strip_unmodeled_mot_state,
)
from .mot_model.model import MoTConfig, MoTPolicyModel, build_mot_policy


def read_mot_state_dict(checkpoint_dir_or_file: str) -> tuple[dict[str, torch.Tensor], str]:
    """Return ``(state_dict, path)``: the payload memory-mapped on CPU, a ``state_dict`` wrapper and ``module.`` prefixes removed."""

    path = resolve_checkpoint_file(checkpoint_dir_or_file)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("state_dict", payload)
    return {k.removeprefix("module."): v for k, v in state.items()}, path


def build_mot_model(config: MoTConfig, dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cpu") -> MoTPolicyModel:
    """Construct the MoT mirror in eval mode at ``dtype`` on ``device`` (converted BEFORE the fp32 weights are copied in)."""

    model = build_mot_policy(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_mot_weights(
    model: MoTPolicyModel,
    checkpoint: Union[str, Mapping[str, torch.Tensor]],
    device: str | torch.device = "cpu",
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Strict-load a checkpoint (a directory / file path, or an already read state dict) into ``model``; return a load report."""

    if isinstance(checkpoint, Mapping):
        state, path = dict(checkpoint), source
    else:
        state, path = read_mot_state_dict(checkpoint)

    config = model.mot_config
    layout = detect_context_layout(state)
    if layout != config.context_layout:
        raise ValueError(f"checkpoint text layout {layout!r} disagrees with the model built for {config.context_layout!r}")
    indices, video_softmax, action_softmax = checkpoint_layer_layout(state)
    if indices != list(range(config.video.depth)):
        raise ValueError(f"checkpoint holds blocks {indices[:4]}...{indices[-4:]} ({len(indices)}), config depth is {config.video.depth}")
    expected = tuple(model.softmax_layer_indices)
    if video_softmax != expected or action_softmax != expected:
        raise ValueError(f"checkpoint softmax layers video={video_softmax} action={action_softmax} differ from the config {expected}")
    if "video_dit.x_embedder.proj.weight" not in state or "action_dit.action_embed.proj.weight" not in state:
        raise ValueError("checkpoint lacks the video / action input embedders; this is not a paired MoT state dict")
    has_state_context = state_as_context_in_checkpoint(state, layout)
    if has_state_context != bool(config.action_state_as_context):
        raise ValueError(
            f"checkpoint state-conditioning layout (state-as-context projector present = {has_state_context}) disagrees "
            f"with model.extra.action_state_as_context = {config.action_state_as_context}"
        )

    active = strip_unmodeled_mot_state(
        state,
        context_layout=layout,
        input_size=config.video.input_size,
        hidden_size=config.video.hidden_size,
        model_max_length=config.video.model_max_length,
        caption_channels=config.video.caption_channels,
    )
    stripped = sorted(set(state) - set(active))
    load_mot_state_dict(model, state)
    model.to(device=device)
    model.eval()
    return {
        "tensors_total": len(state),
        "tensors_loaded": len(active),
        "stripped": stripped,
        "source": path,
        "context_layout": layout,
        "video_layout": config.video_layout,
        "softmax_layer_indices": expected,
        "state_as_context": bool(config.action_state_as_context),
        "dtype": str(model.dtype),
    }


__all__ = [
    "CHECKPOINT_RELATIVE_PATH",
    "build_mot_model",
    "load_mot_weights",
    "read_mot_state_dict",
    "resolve_checkpoint_file",
]
