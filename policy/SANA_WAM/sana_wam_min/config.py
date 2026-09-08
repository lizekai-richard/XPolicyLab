"""Train-yaml to PolicyConfig resolution for the SANA unified policy.

The training yaml is the only description of the checkpointed architecture
that travels with a run.  This module reproduces the live builder chain
(image_size // vae_stride[-1] -> model_video_init_config -> factory) with
plain dict access so no Sana config class is needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import yaml

from .policy_model.config import PolicyConfig

# Trunk dimensions hard-coded by the registry factory
# ``..._5B_P1_D36`` (depth 32 despite the historical "D36" suffix).
FACTORY_DEPTH = 32
FACTORY_HIDDEN_SIZE = 2560
FACTORY_PATCH_SIZE = (1, 1, 1)
FACTORY_NUM_HEADS = 20

# Validated sampling behaviour of the holdout run.  The yaml's
# ``scheduler.inference_cfg_scale`` (6.0) is a training-time visualization
# knob that the validation and deploy sessions never used for this line; the
# holdout manifests record cfg_scale 1.0 and that is the only value the
# action_mse reference was produced with.
VALIDATED_CFG_SCALE = 1.0


def load_train_config(path: str) -> dict[str, Any]:
    """Parse the training yaml into a plain dict."""

    with open(path) as handle:
        return yaml.safe_load(handle)


def latent_input_size(cfg: dict[str, Any]) -> int:
    """Per-view latent side used by the live builder: image_size // vae_stride[-1]."""

    return int(cfg["model"]["image_size"]) // int(cfg["vae"]["vae_stride"][-1])


def policy_config_from_train_config(cfg: dict[str, Any]) -> PolicyConfig:
    """Resolve the PolicyConfig exactly as the live builder chain does.

    Mirrors ``model_video_init_config`` + the factory: trunk dims from the
    factory, latent/text dims from the yaml, softmax layer indices from
    ``softmax_ratio`` when ``softmax_layer_indices`` is null, rope layout and
    tile from ``model.*``, temporal compression from ``vae.vae_stride[0]``.
    ``y_norm_scale_factor`` and ``timestep_norm_scale_factor`` are not
    forwarded by the video builder, so the constructor defaults (1.0) apply.
    """

    model_cfg = cfg["model"]
    vae_cfg = cfg["vae"]
    text_cfg = cfg["text_encoder"]
    sched_cfg = cfg["scheduler"]
    duck_config = SimpleNamespace(
        model=SimpleNamespace(
            multiview_spatial_rope_layout=model_cfg["multiview_spatial_rope_layout"],
            multiview_spatial_rope_tile_shape=tuple(
                model_cfg["multiview_spatial_rope_tile_shape"]
            ),
        ),
        vae=SimpleNamespace(vae_stride=list(vae_cfg["vae_stride"])),
    )
    return PolicyConfig.from_sana_kwargs(
        depth=FACTORY_DEPTH,
        hidden_size=FACTORY_HIDDEN_SIZE,
        patch_size=FACTORY_PATCH_SIZE,
        num_heads=FACTORY_NUM_HEADS,
        input_size=latent_input_size(cfg),
        in_channels=int(vae_cfg["vae_latent_dim"]),
        caption_channels=int(text_cfg["caption_channels"]),
        model_max_length=int(text_cfg["model_max_length"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        linear_head_dim=int(model_cfg["linear_head_dim"]),
        softmax_head_dim=model_cfg.get("softmax_head_dim"),
        softmax_ratio=float(model_cfg.get("softmax_ratio", 0.25)),
        softmax_layer_indices=model_cfg.get("softmax_layer_indices"),
        attn_res_block_size=int(model_cfg["attn_res_block_size"]),
        qk_norm=bool(model_cfg["qk_norm"]),
        cross_norm=bool(model_cfg["cross_norm"]),
        y_norm=bool(text_cfg["y_norm"]),
        pred_sigma=bool(sched_cfg["pred_sigma"]),
        use_pe=bool(model_cfg["use_pe"]),
        pos_embed_type=model_cfg["pos_embed_type"],
        use_attn_res=bool(model_cfg["use_attn_res"]),
        use_time_conditioning=bool(model_cfg["use_time_conditioning"]),
        use_dual_attn_res_routing=bool(model_cfg["use_dual_attn_res_routing"]),
        cross_attn_image_embeds=bool(model_cfg["cross_attn_image_embeds"]),
        rope_fhw_dim=model_cfg.get("rope_fhw_dim"),
        ffn_type=model_cfg["ffn_type"],
        linear_attn_type=model_cfg["linear_attn_type"],
        softmax_attn_type=model_cfg["softmax_attn_type"],
        use_fp32_attention=bool(model_cfg["fp32_attention"]),
        config=duck_config,
    )


def sampling_defaults_from_train_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Sampling knobs the validated holdout run used.

    steps <- ``train.extra.rwm_validation_steps`` (50); flow_shift <-
    ``scheduler.inference_flow_shift`` falling back to ``scheduler.flow_shift``;
    cfg_scale is pinned to ``VALIDATED_CFG_SCALE`` rather than the yaml's
    ``inference_cfg_scale`` (see the constant's comment).
    """

    extra = cfg["train"].get("extra") or {}
    sched_cfg = cfg["scheduler"]
    flow_shift = sched_cfg.get("inference_flow_shift")
    if flow_shift is None:
        flow_shift = sched_cfg["flow_shift"]
    return {
        "steps": int(extra.get("rwm_validation_steps", 50)),
        "flow_shift": float(flow_shift),
        "cfg_scale": float(VALIDATED_CFG_SCALE),
    }


__all__ = [
    "FACTORY_DEPTH",
    "FACTORY_HIDDEN_SIZE",
    "FACTORY_NUM_HEADS",
    "FACTORY_PATCH_SIZE",
    "VALIDATED_CFG_SCALE",
    "latent_input_size",
    "load_train_config",
    "policy_config_from_train_config",
    "sampling_defaults_from_train_config",
]
