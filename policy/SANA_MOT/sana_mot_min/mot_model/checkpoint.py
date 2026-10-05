"""Strict checkpoint adapter for the MoT policy mirror.

A trained MoT checkpoint is the accelerate FSDP full state dict of ``SanaRWMMoTAttnResPolicy`` with the trunks already
paired (``video_dit.*`` / ``action_dit.*`` for the expert-level modules, ``blocks.<i>.video_block.*`` /
``blocks.<i>.action_block.*`` per layer). Its text path tells the two layouts apart (:func:`detect_context_layout`):

* ``context_embedder`` (rwm/mot @ b3b9e0e9e .., 1,595 keys at real scale): ``context_embedder.{y_embedder, y_norm,
  action_y_embedder, action_y_norm[, state_proj]}`` -- one caption embedder per expert (``multiview: sana_latent``,
  every canvas run before 2026-09-22, and every mode again from rwm/mot b010eb9a5 on).
* ``shared_caption_embedder`` (rwm/mot @ 4e67e1e1d .. b010eb9a5, the canvas modes ``openwam`` / ``sana_pixel``, 1,589 keys):
  ``context_embedder.{y_embedder, y_norm[, state_proj]}`` only -- ONE caption embedder whose 2560-wide output both
  experts cross-attend to, so every action block's ``cross_attn.kv_linear`` is ``[2048, 2560]``.
* ``legacy_action_mlp`` (rwm/mot @ 606e48dd9, 1,593 keys): ``video_dit.{y_embedder, attention_y_norm}`` +
  ``action_dit.context_mlp[, state_context_embed]``.

The action expert's per-layer attention projections are ``blocks.<i>.action_block.attn.*`` since rwm/mot 3058e5785
(2026-09-18); checkpoints written before carry ``attn_head.*``, which :func:`normalize_action_attention_keys` renames
(a pure key rename, verified bit-identical upstream), so no pre-converted copy of a checkpoint is needed.

Buffers the inference forward never reads are removed after a shape check -- ``video_dit.pos_embed`` (unused under
wan_rope) and every caption-dropout null table ``*.y_embedder.y_embedding`` (``uncond_prob`` is 0 in the policy; CFG
runs on the instruction-free prompt) -- and everything else must load strictly.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

CONTEXT_EMBEDDER = "context_embedder"
SHARED_CAPTION_EMBEDDER = "shared_caption_embedder"
LEGACY_ACTION_MLP = "legacy_action_mlp"
UNMODELED_KEYS = {
    CONTEXT_EMBEDDER: (
        "video_dit.pos_embed",
        "context_embedder.y_embedder.y_embedding",
        "context_embedder.action_y_embedder.y_embedding",
    ),
    SHARED_CAPTION_EMBEDDER: ("video_dit.pos_embed", "context_embedder.y_embedder.y_embedding"),
    LEGACY_ACTION_MLP: ("video_dit.pos_embed", "video_dit.y_embedder.y_embedding"),
}
_OLD_ACTION_ATTENTION = ".action_block.attn_head."
_ACTION_ATTENTION = ".action_block.attn."


def normalize_action_attention_keys(state: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    """Return ``(state, renamed)``: ``blocks.<i>.action_block.attn_head.*`` (before rwm/mot 3058e5785) renamed to the
    current ``blocks.<i>.action_block.attn.*``; a state dict mixing both spellings is refused."""

    old = [key for key in state if _OLD_ACTION_ATTENTION in key]
    if not old:
        return dict(state), 0
    if any(_ACTION_ATTENTION in key for key in state):
        raise ValueError("this MoT state dict mixes action_block.attn_head.* and action_block.attn.* keys")
    return {key.replace(_OLD_ACTION_ATTENTION, _ACTION_ATTENTION): value for key, value in state.items()}, len(old)


def detect_context_layout(state: Mapping[str, torch.Tensor]) -> str:
    """Return the text-path layout a MoT state dict was saved with; refuse the transient in-between layouts."""

    if "action_dit.context_mlp.0.weight" in state:
        return LEGACY_ACTION_MLP
    if "context_embedder.action_y_embedder.y_proj.fc1.weight" in state:
        return CONTEXT_EMBEDDER
    if "context_embedder.y_embedder.y_proj.fc1.weight" in state and not any(
        key.startswith(("context_embedder.text_proj", "context_embedder.action_")) for key in state
    ):
        return SHARED_CAPTION_EMBEDDER
    if any(key.startswith("context_embedder.") for key in state):
        raise ValueError(
            "this MoT checkpoint has a transient context layout of rwm/mot between 606e48dd9 and 8fd95e219 "
            "(context_embedder.text_proj or a shared 2560-wide action context); no evaluation checkpoint uses it, "
            "and it is not served"
        )
    raise ValueError("not a MoT policy state dict: neither action_dit.context_mlp nor context_embedder.action_y_embedder is present")


def strip_unmodeled_mot_state(
    state_dict: Mapping[str, torch.Tensor],
    *,
    context_layout: str,
    input_size: int,
    hidden_size: int,
    model_max_length: int,
    caption_channels: int,
) -> dict[str, torch.Tensor]:
    """Remove the buffers the mirror does not model; shape drift on any of them is rejected."""

    if context_layout not in UNMODELED_KEYS:
        raise ValueError(f"unknown MoT context layout {context_layout!r}")
    output = dict(state_dict)
    expected = {"video_dit.pos_embed": (1, input_size * input_size, hidden_size)}
    for key in UNMODELED_KEYS[context_layout][1:]:
        expected[key] = (model_max_length, caption_channels)
    for key, shape in expected.items():
        tensor = output.pop(key, None)
        if tensor is None:
            continue
        if tuple(tensor.shape) != shape:
            raise ValueError(f"checkpoint buffer {key!r} has shape {tuple(tensor.shape)}, expected {shape}")
    return output


def checkpoint_layer_layout(state: Mapping[str, torch.Tensor]) -> tuple[list[int], tuple[int, ...], tuple[int, ...]]:
    """Return ``(block indices, video softmax layers, action softmax layers)`` recorded in a MoT state dict.

    A softmax layer is one without ``attn.beta_proj`` (video) / ``attn.beta_proj`` or ``attn_head.beta_proj`` (action):
    the GDN heads carry the beta projection, the softmax heads do not.
    """

    indices = sorted({int(k.split(".")[1]) for k in state if k.startswith("blocks.")})
    video_softmax = tuple(i for i in indices if f"blocks.{i}.video_block.attn.beta_proj.weight" not in state)
    action_softmax = tuple(
        i
        for i in indices
        if f"blocks.{i}.action_block.attn.beta_proj.weight" not in state
        and f"blocks.{i}.action_block.attn_head.beta_proj.weight" not in state
    )
    return indices, video_softmax, action_softmax


def state_as_context_in_checkpoint(state: Mapping[str, torch.Tensor], context_layout: str) -> bool:
    """Whether the checkpoint carries the state-as-context projector of its layout."""

    key = (
        "action_dit.state_context_embed.proj.weight"
        if context_layout == LEGACY_ACTION_MLP
        else "context_embedder.state_proj.proj.weight"
    )
    return key in state


def load_mot_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Strict-load every inference-active tensor after the validated buffer removal."""

    config = model.mot_config
    state_dict, _ = normalize_action_attention_keys(state_dict)
    active = strip_unmodeled_mot_state(
        state_dict,
        context_layout=config.context_layout,
        input_size=config.video.input_size,
        hidden_size=config.video.hidden_size,
        model_max_length=config.video.model_max_length,
        caption_channels=config.video.caption_channels,
    )
    incompatible = model.load_state_dict(active, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict MoT policy load failed: {incompatible}")


__all__ = [
    "CONTEXT_EMBEDDER",
    "LEGACY_ACTION_MLP",
    "SHARED_CAPTION_EMBEDDER",
    "normalize_action_attention_keys",
    "UNMODELED_KEYS",
    "checkpoint_layer_layout",
    "detect_context_layout",
    "load_mot_state_dict",
    "state_as_context_in_checkpoint",
    "strip_unmodeled_mot_state",
]
