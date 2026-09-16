"""Train-yaml to PolicyConfig resolution for the SANA unified policy.

The training yaml is the only description of the checkpointed architecture
that travels with a run.  This module reproduces the live builder chain
(image_size // vae_stride[-1] -> model_video_init_config -> factory) with
plain dict access so no Sana config class is needed, and reads from the same
yaml the two line-level contracts the deploy session needs: the visual
front-end (three per-view streams packed as a strip, or the OpenWAM composite
canvas) and the action-mode / target-mode words of the prompt.
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

# Visual front-ends this adapter serves, keyed by the registered model factory of the training yaml.
# three_view_strip: every camera resized/cropped to 256x320 and encoded on its own, the three latents packed as
#   one strip (the 320px lines: G = V + 1 prompt groups, semantic_2x2 RoPE tiles).
# openwam_canvas: the three cameras stretched into ONE 384x320 L-shaped RGB canvas encoded once (the rwm/openwam
#   canvas line: native 12x10 latent grid, one prompt shared by the video and action tokens).
VISUAL_LAYOUT_THREE_VIEW_STRIP = "three_view_strip"
VISUAL_LAYOUT_OPENWAM_CANVAS = "openwam_canvas"
VISUAL_LAYOUTS = (VISUAL_LAYOUT_THREE_VIEW_STRIP, VISUAL_LAYOUT_OPENWAM_CANVAS)
THREE_VIEW_POLICY_FACTORY = (
    "SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy_5B_P1_D36"
)
OPENWAM_CANVAS_POLICY_FACTORY = "SanaRWMOpenWAMCanvasPolicy_5B_P1_D36"
OPENWAM_CANVAS_DATASET_TYPE = "RoboDojoOpenWAMCanvasSFTDataset"
OPENWAM_CANVAS_LAYOUT_ID = "openwam_lshape_rgb_v1"
OPENWAM_CANVAS_ENCODE_MODE = "joint_rgb_canvas"
OPENWAM_CANVAS_ASPECT_RATIO_TYPE = "ASPECT_RATIO_OPENWAM_LSHAPE_384_320"

# ``data.extra.action_mode_sample_ratio`` order (Sana ``ACTION_MODES``) and the joint / EEF target modes.
ACTION_MODES = ("qwen_canonical", "robot_base_eef", "joint_only")
TARGET_MODES = ("anchor_delta", "absolute")
# Lines that predate ``data.extra.eef_target_mode`` (the robot_base_eef SFT line) packed anchor-relative EEF targets.
DEFAULT_EEF_TARGET_MODE = "anchor_delta"


def load_train_config(path: str) -> dict[str, Any]:
    """Parse the training yaml into a plain dict."""

    with open(path) as handle:
        return yaml.safe_load(handle)


def latent_input_size(cfg: dict[str, Any]) -> int:
    """Per-view latent side used by the live builder: image_size // vae_stride[-1]."""

    return int(cfg["model"]["image_size"]) // int(cfg["vae"]["vae_stride"][-1])


def _model_value(model_cfg: dict[str, Any], key: str, default: Any) -> Any:
    """``model.<key>`` of the yaml, or the live ``ModelConfig`` default when the key is absent or null."""

    value = model_cfg.get(key)
    return default if value is None else value


def resolve_visual_layout(cfg: dict[str, Any]) -> str:
    """The visual front-end the checkpoint was trained with, from ``model.model`` and ``data.type``.

    The two declarations must agree (a canvas policy on the three-view dataset, or the reverse, is refused),
    and a canvas line must use the one implemented layout / encode mode / aspect bucket.
    """

    model_name = str(cfg["model"]["model"])
    data_cfg = cfg["data"]
    data_type = str(data_cfg.get("type") or "")
    extra = data_cfg.get("extra") or {}
    canvas_options = extra.get("openwam_canvas")
    if model_name == OPENWAM_CANVAS_POLICY_FACTORY:
        if data_type != OPENWAM_CANVAS_DATASET_TYPE:
            raise ValueError(
                f"model.model {model_name} needs data.type {OPENWAM_CANVAS_DATASET_TYPE}, got {data_type!r}"
            )
        options = dict(canvas_options or {})
        layout = str(options.get("layout", OPENWAM_CANVAS_LAYOUT_ID))
        if layout != OPENWAM_CANVAS_LAYOUT_ID:
            raise ValueError(f"unsupported openwam_canvas.layout {layout!r}; only {OPENWAM_CANVAS_LAYOUT_ID!r} is served")
        encode_mode = str(options.get("encode_mode", OPENWAM_CANVAS_ENCODE_MODE))
        if encode_mode != OPENWAM_CANVAS_ENCODE_MODE:
            raise ValueError(
                f"unsupported openwam_canvas.encode_mode {encode_mode!r}; only {OPENWAM_CANVAS_ENCODE_MODE!r} is served"
            )
        aspect = data_cfg.get("aspect_ratio_type")
        if aspect not in (None, OPENWAM_CANVAS_ASPECT_RATIO_TYPE):
            raise ValueError(
                f"the canvas line trains on {OPENWAM_CANVAS_ASPECT_RATIO_TYPE}, the yaml declares {aspect!r}"
            )
        return VISUAL_LAYOUT_OPENWAM_CANVAS
    if model_name == THREE_VIEW_POLICY_FACTORY:
        if data_type == OPENWAM_CANVAS_DATASET_TYPE or canvas_options:
            raise ValueError(
                f"model.model {model_name} is the three-view policy but the yaml declares canvas data "
                f"(data.type {data_type!r}, openwam_canvas {canvas_options!r})"
            )
        return VISUAL_LAYOUT_THREE_VIEW_STRIP
    raise ValueError(
        f"unsupported model.model {model_name!r}; this adapter serves {THREE_VIEW_POLICY_FACTORY} "
        f"and {OPENWAM_CANVAS_POLICY_FACTORY}"
    )


def state_as_cross_attention_from_train_config(cfg: dict[str, Any]) -> bool:
    """``model.extra.state_as_cross_attention`` (default False); only the canvas policy defines it."""

    extra = cfg["model"].get("extra") or {}
    enabled = bool(extra.get("state_as_cross_attention", False))
    if enabled and resolve_visual_layout(cfg) != VISUAL_LAYOUT_OPENWAM_CANVAS:
        raise ValueError("model.extra.state_as_cross_attention is only defined for the OpenWAM canvas policy")
    return enabled


def action_mode_from_train_config(cfg: dict[str, Any]) -> str:
    """The single action mode of the line from ``data.extra.action_mode_sample_ratio``
    ((qwen_canonical, robot_base_eef, joint_only) weights); absent -> joint_only.

    A mixed ratio cannot be served (the deploy prompt names one mode) and qwen_canonical (camera-frame EEF)
    is not supported by this adapter.
    """

    extra = (cfg.get("data") or {}).get("extra") or {}
    ratios = extra.get("action_mode_sample_ratio")
    if ratios is None:
        return "joint_only"
    weights = [float(value) for value in ratios]
    if len(weights) != len(ACTION_MODES) or any(weight < 0 for weight in weights):
        raise ValueError(f"unexpected data.extra.action_mode_sample_ratio {ratios!r} in the training yaml")
    active = [mode for mode, weight in zip(ACTION_MODES, weights) if weight > 0]
    if len(active) != 1:
        raise ValueError(
            f"data.extra.action_mode_sample_ratio {ratios!r} names {len(active)} action modes; the deploy "
            "session serves exactly one"
        )
    if active[0] == "qwen_canonical":
        raise ValueError("qwen_canonical (camera-frame EEF) checkpoints are not supported by this adapter")
    return active[0]


def joint_target_mode_from_train_config(cfg: dict[str, Any]) -> str:
    """``data.extra.joint_target_mode`` (anchor_delta | absolute)."""

    mode = str(cfg["data"]["extra"]["joint_target_mode"])
    if mode not in TARGET_MODES:
        raise ValueError(f"unsupported data.extra.joint_target_mode {mode!r}; expected one of {TARGET_MODES}")
    return mode


def eef_target_mode_from_train_config(cfg: dict[str, Any]) -> str:
    """``data.extra.eef_target_mode`` (anchor_delta | absolute); absent -> ``DEFAULT_EEF_TARGET_MODE``."""

    extra = (cfg.get("data") or {}).get("extra") or {}
    mode = str(extra.get("eef_target_mode") or DEFAULT_EEF_TARGET_MODE)
    if mode not in TARGET_MODES:
        raise ValueError(f"unsupported data.extra.eef_target_mode {mode!r}; expected one of {TARGET_MODES}")
    return mode


def policy_config_from_train_config(cfg: dict[str, Any]) -> PolicyConfig:
    """Resolve the PolicyConfig exactly as the live builder chain does.

    Mirrors ``model_video_init_config`` + the factory: trunk dims from the
    factory, latent/text dims from the yaml, softmax layer indices from
    ``softmax_ratio`` when ``softmax_layer_indices`` is null, rope layout and
    tile from ``model.*`` (the live ``ModelConfig`` defaults when the yaml does
    not set them), temporal compression from ``vae.vae_stride[0]``, and the
    canvas policy's ``shared_prompt`` / ``state_as_cross_attention`` from
    ``model.model`` / ``model.extra``.  ``y_norm_scale_factor`` and
    ``timestep_norm_scale_factor`` are not forwarded by the video builder, so
    the constructor defaults (1.0) apply.
    """

    model_cfg = cfg["model"]
    vae_cfg = cfg["vae"]
    text_cfg = cfg["text_encoder"]
    sched_cfg = cfg["scheduler"]
    layout = resolve_visual_layout(cfg)
    duck_config = SimpleNamespace(
        model=SimpleNamespace(
            multiview_spatial_rope_layout=str(
                _model_value(model_cfg, "multiview_spatial_rope_layout", "local_reset")
            ),
            multiview_spatial_rope_tile_shape=tuple(
                _model_value(model_cfg, "multiview_spatial_rope_tile_shape", (15, 30))
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
        caption_channels=int(_model_value(text_cfg, "caption_channels", 2304)),
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
        use_dual_attn_res_routing=bool(_model_value(model_cfg, "use_dual_attn_res_routing", False)),
        cross_attn_image_embeds=bool(_model_value(model_cfg, "cross_attn_image_embeds", False)),
        rope_fhw_dim=model_cfg.get("rope_fhw_dim"),
        ffn_type=model_cfg["ffn_type"],
        linear_attn_type=model_cfg["linear_attn_type"],
        softmax_attn_type=model_cfg["softmax_attn_type"],
        use_fp32_attention=bool(model_cfg["fp32_attention"]),
        shared_prompt=layout == VISUAL_LAYOUT_OPENWAM_CANVAS,
        state_as_cross_attention=state_as_cross_attention_from_train_config(cfg),
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
    "ACTION_MODES",
    "DEFAULT_EEF_TARGET_MODE",
    "FACTORY_DEPTH",
    "FACTORY_HIDDEN_SIZE",
    "FACTORY_NUM_HEADS",
    "FACTORY_PATCH_SIZE",
    "OPENWAM_CANVAS_ASPECT_RATIO_TYPE",
    "OPENWAM_CANVAS_DATASET_TYPE",
    "OPENWAM_CANVAS_ENCODE_MODE",
    "OPENWAM_CANVAS_LAYOUT_ID",
    "OPENWAM_CANVAS_POLICY_FACTORY",
    "TARGET_MODES",
    "THREE_VIEW_POLICY_FACTORY",
    "VALIDATED_CFG_SCALE",
    "VISUAL_LAYOUTS",
    "VISUAL_LAYOUT_OPENWAM_CANVAS",
    "VISUAL_LAYOUT_THREE_VIEW_STRIP",
    "action_mode_from_train_config",
    "eef_target_mode_from_train_config",
    "joint_target_mode_from_train_config",
    "latent_input_size",
    "load_train_config",
    "policy_config_from_train_config",
    "resolve_visual_layout",
    "sampling_defaults_from_train_config",
    "state_as_cross_attention_from_train_config",
]
