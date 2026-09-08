"""Policy model construction and strict checkpoint loading.

The trained checkpoint is an accelerate FSDP full state dict
(``model/pytorch_model_fsdp.bin``): a plain ``OrderedDict`` with three
tensors the inference mirror does not model (``pos_embed``,
``y_embedder.y_embedding``, ``plucker_embed.weight``).  Everything else must
load strictly.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .policy_model.checkpoint import load_policy_state_dict, strip_unmodeled_state
from .policy_model.config import PolicyConfig
from .policy_model.model import PolicyModel, build_policy

CHECKPOINT_RELATIVE_PATH = os.path.join("model", "pytorch_model_fsdp.bin")


def resolve_checkpoint_file(checkpoint_dir_or_file: str) -> str:
    """Return the state-dict file for a checkpoint directory or a direct file path."""

    if os.path.isdir(checkpoint_dir_or_file):
        return os.path.join(checkpoint_dir_or_file, CHECKPOINT_RELATIVE_PATH)
    return checkpoint_dir_or_file


def build_policy_model(
    policy_config: PolicyConfig,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> PolicyModel:
    """Construct the policy in eval mode at ``dtype`` on ``device``.

    The dtype conversion happens before any weights are loaded: the SFT
    validation model and the deploy session both convert to bf16 first and
    then copy fp32 checkpoint tensors in (round-to-nearest-even), which is
    what the holdout reference was produced with.
    """

    model = build_policy(policy_config)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_policy_weights(
    model: PolicyModel,
    checkpoint_dir_or_file: str,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Strict-load the checkpoint into ``model`` and return a load report.

    The payload is memory-mapped on CPU, an optional ``state_dict`` wrapper is
    unwrapped and an optional ``module.`` prefix stripped (both no-ops for the
    FSDP full state dict), the three unmodeled tensors are shape-checked and
    removed, and the remainder must match ``model.state_dict()`` exactly.  The
    softmax-layer placement recorded in the checkpoint (blocks without
    ``attn.beta_proj``) must agree with the config before any tensor is copied.
    """

    path = resolve_checkpoint_file(checkpoint_dir_or_file)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("state_dict", payload)
    state = {k.removeprefix("module."): v for k, v in state.items()}

    config = model.policy_config
    block_indices = sorted({int(k.split(".")[1]) for k in state if k.startswith("blocks.")})
    if block_indices != list(range(config.depth)):
        raise ValueError(
            f"checkpoint holds blocks {block_indices[:4]}...{block_indices[-4:]} "
            f"({len(block_indices)}), config depth is {config.depth}"
        )
    softmax_blocks = tuple(
        i for i in block_indices if f"blocks.{i}.attn.beta_proj.weight" not in state
    )
    if softmax_blocks != tuple(config.softmax_layer_indices):
        raise ValueError(
            f"checkpoint softmax blocks {softmax_blocks} differ from config "
            f"{tuple(config.softmax_layer_indices)}"
        )

    active = strip_unmodeled_state(
        state,
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        model_max_length=config.model_max_length,
        caption_channels=config.caption_channels,
    )
    stripped = sorted(set(state) - set(active))
    load_policy_state_dict(model, state)
    model.to(device=device)
    model.eval()
    return {
        "tensors_total": len(state),
        "tensors_loaded": len(active),
        "stripped": stripped,
        "source": path,
        "softmax_layer_indices": softmax_blocks,
        "dtype": str(model.dtype),
    }


__all__ = [
    "CHECKPOINT_RELATIVE_PATH",
    "build_policy_model",
    "load_policy_weights",
    "resolve_checkpoint_file",
]
