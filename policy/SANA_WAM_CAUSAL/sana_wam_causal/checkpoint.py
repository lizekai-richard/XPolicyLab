"""Strict checkpoint loading for the causal mirror.

A causal checkpoint is the accelerate FSDP full state dict of the causal trainer (``model/pytorch_model_fsdp.bin``):
the bidirectional policy's tensors plus every GDN block's recurrence parameters (``gate_proj.{weight,bias}``,
``A_log``, ``dt_bias`` and the never-read ``recall_gate`` buffer). Everything must load strictly, except:

* the three tensors the policy mirror never models (``pos_embed``, ``y_embedder.y_embedding``,
  ``plucker_embed.weight``; shape-checked and dropped by ``strip_unmodeled_state``);
* with ``allow_missing_recurrence`` (an SFT donor served as the causal policy's step-0 model), the recurrence keys,
  which are then re-initialized exactly as the causal trainer does on its first load (Sana
  ``ChunkGDNLinearAttention._load_from_state_dict`` -> ``_init_new_recurrence_params``: zero gate projection,
  ``A_log`` / ``dt_bias`` from ``chunk_causal_policy.a_log_init`` / ``dt_bias_init``).
"""

from __future__ import annotations

from typing import Any

import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.checkpoint import resolve_checkpoint_file  # noqa: E402
from sana_wam_min.policy_model.checkpoint import strip_unmodeled_state  # noqa: E402

from .layers import GDN_KIND  # noqa: E402
from .model import GDN_RECURRENCE_KEYS, CausalPolicyModel  # noqa: E402

# Sana re-initializes a GDN layer's recurrence unless all four are present (``has_gdn_recurrence``)
_RECURRENCE_REQUIRED = ("gate_proj.weight", "gate_proj.bias", "A_log", "dt_bias")


def load_causal_state_dict(
    model: CausalPolicyModel, state: dict[str, torch.Tensor], *, allow_missing_recurrence: bool = False
) -> dict[str, Any]:
    """Load ``state`` (already unwrapped) into ``model``; returns which layers kept / re-initialized their recurrence."""

    config = model.policy_config
    block_indices = sorted({int(k.split(".")[1]) for k in state if k.startswith("blocks.")})
    if block_indices != list(range(config.depth)):
        raise ValueError(
            f"checkpoint holds blocks {block_indices[:4]}...{block_indices[-4:]} ({len(block_indices)}), "
            f"config depth is {config.depth}"
        )
    softmax_blocks = tuple(i for i in block_indices if f"blocks.{i}.attn.beta_proj.weight" not in state)
    if softmax_blocks != tuple(config.softmax_layer_indices):
        raise ValueError(f"checkpoint softmax blocks {softmax_blocks} differ from config {tuple(config.softmax_layer_indices)}")
    if "state_context_embed.proj.weight" in state or "state_embed.proj.weight" not in state:
        raise ValueError("the causal policy conditions on the state token (state_embed); this checkpoint does not")

    active = strip_unmodeled_state(
        state,
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        model_max_length=config.model_max_length,
        caption_channels=config.caption_channels,
    )
    expected = set(model.state_dict().keys())
    unexpected = sorted(set(active) - expected)
    if unexpected:
        raise RuntimeError(f"causal checkpoint carries tensors the mirror does not model: {unexpected[:8]}")
    recurrence = set(model.gdn_recurrence_keys())
    missing = sorted(expected - set(active))
    not_recurrence = [k for k in missing if k not in recurrence]
    if not_recurrence:
        raise RuntimeError(f"causal checkpoint misses policy tensors: {not_recurrence[:8]}")

    fresh_layers, kept_layers = [], []
    for index, block in enumerate(model.blocks):
        if block.attn_type != GDN_KIND:
            continue
        present = all(f"blocks.{index}.attn.{name}" in active for name in _RECURRENCE_REQUIRED)
        (kept_layers if present else fresh_layers).append(index)
    if fresh_layers and not allow_missing_recurrence:
        raise RuntimeError(
            f"the checkpoint has no GDN recurrence parameters for blocks {fresh_layers}: it is a bidirectional SFT "
            "checkpoint, not a causal one. Serve it as the causal policy's step-0 model only on purpose "
            "(deploy key allow_donor_checkpoint: true)."
        )
    if fresh_layers and kept_layers:
        raise RuntimeError(f"GDN recurrence present for blocks {kept_layers} but missing for {fresh_layers}")
    result = model.load_state_dict(active, strict=False)
    still_missing = [k for k in result.missing_keys if not (k in recurrence and int(k.split(".")[1]) in fresh_layers)]
    # a causal checkpoint without the unused recall_gate buffer is accepted (it is zero and never read)
    still_missing = [k for k in still_missing if not k.endswith(".attn.recall_gate")]
    if still_missing or result.unexpected_keys:
        raise RuntimeError(f"strict causal load failed: missing {still_missing[:8]} unexpected {result.unexpected_keys[:8]}")
    for index in fresh_layers:
        model.blocks[index].attn.init_recurrence_parameters()
    return {
        "tensors_total": len(state),
        "tensors_loaded": len(active),
        "stripped": sorted(set(state) - set(active)),
        "softmax_layer_indices": softmax_blocks,
        "gdn_recurrence": "checkpoint" if not fresh_layers else "initialized (SFT donor, causal step 0)",
        "gdn_recurrence_initialized_blocks": fresh_layers,
    }


def load_causal_weights(
    model: CausalPolicyModel,
    checkpoint_dir_or_file: str,
    *,
    allow_missing_recurrence: bool = False,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Memory-map the checkpoint on CPU, unwrap it and load it with :func:`load_causal_state_dict`."""

    path = resolve_checkpoint_file(checkpoint_dir_or_file)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("state_dict", payload)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    report = load_causal_state_dict(model, state, allow_missing_recurrence=allow_missing_recurrence)
    model.to(device=device)
    model.eval()
    report.update({"source": path, "dtype": str(model.dtype)})
    return report


__all__ = ["GDN_RECURRENCE_KEYS", "load_causal_state_dict", "load_causal_weights"]
