"""Train-yaml to :class:`MoTConfig` resolution for the SANA MoT policy.

The training yaml (``config.yaml`` next to a checkpoint) is the only description of the checkpointed architecture
that travels with a run. This module reproduces the live builder chain for the factory
``SanaRWMMoTAttnResPolicy_5B_P1_D36`` (trunk dims from the factory, latent/text dims from the yaml, softmax layer
indices from ``softmax_ratio``, the MoT knobs from ``model.extra`` under their constructor names) with plain dict
access so no Sana config class is needed. The checkpoint's text layout (``context_layout``) is not in the yaml; the
loader detects it from the state dict and passes it in.

Video layout (the adapter keeps the names ``multiview`` / ``openwam_canvas`` / ``sana_pixel_canvas``): since rwm/mot
a26807821 (2026-09-21) the DATA knob ``data.extra.multiview`` (``sana`` -> ``sana_latent`` on 2026-09-22, plus
``sana_pixel`` and ``openwam``) decides it; before that ``model.extra.video_layout`` (33b220373 .. a26807821); before
that ``data.type``: a ``*Canvas*`` dataset means the OpenWAM canvas, any other dataset the ``multiview`` strip. Before
33b220373 the canvas was the ONLY MoT video layout, so every yaml of that era omits the key -- the legacy text
layout's (606e48dd9) and the ``context_embedder`` runs launched in the canvas-only window 8fd95e219 .. 33b220373 alike
(the NSC f25 canvas runs at step 36,250, source commit 295f48f15). An explicit key that disagrees with ``data.type``
fails before any weight is read.

RoPE (``model.extra.rope``, rwm/mot 71ac93f43, required by the live factory since): ``aligned`` | ``independent``.
A yaml without it trained the table of its era: ``legacy`` before 42aee4fa9 (2026-09-21 17:54 PDT), the action
expert's own clock (= ``independent``) after it. ``data.extra.multiview`` (a26807821, 1.5 h earlier) or a shared caption
embedder in the weights (4e67e1e1d) marks the later era.

Canvas text contract: ONE row (G = 1) read by both experts before a26807821 (the ``composite_view`` descriptor) and
again from 2b4a4dc8d (``l_shape``, then ``two_rows`` from 3ea1f9af9 / ``tiling`` for sana_pixel); G = 2 (the
composite-view row for the video expert, the robot row for the action expert) in between. The shared caption embedder
(4e67e1e1d) always reads ONE row.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sana_wam_min.config import (
    FACTORY_DEPTH,
    FACTORY_HIDDEN_SIZE,
    FACTORY_NUM_HEADS,
    FACTORY_PATCH_SIZE,
    VALIDATED_CFG_SCALE,
    latent_input_size,
    load_train_config,
    sampling_defaults_from_train_config,
    sana_pixel_canvas_hw_from_train_config,
    sft_options_from_train_config,
    strip_spatial_rope_tile_shape_from_train_config,
)
from sana_wam_min.policy_model.config import PolicyConfig

from .mot_model.model import (
    CONTEXT_EMBEDDER,
    LEGACY_ACTION_MLP,
    MOT_FACTORY_NAME,
    ROPE_ALIGNED,
    ROPE_INDEPENDENT,
    ROPE_LEGACY,
    SHARED_CAPTION_EMBEDDER,
    MoTConfig,
)

# gemma-2-2b-it hidden size; the resolved yaml the trainer dumps carries text_encoder.caption_channels explicitly.
DEFAULT_CAPTION_CHANNELS = 2304
MOT_EXTRA_KEYS = (
    "action_hidden_size",
    "action_mlp_ratio",
    "action_cross_attn_heads",
    "action_rope_theta",
    "action_attn_res_block_size",
    "action_state_as_context",
    "video_layout",
    "multiview_spatial_rope_layout",
    "multiview_spatial_rope_tile_shape",
)
CANVAS_DATASET_MARKER = "Canvas"
SANA_PIXEL_DATASET_TYPE = "RoboDojoSanaPixelCanvasSFTDataset"
# data.extra.multiview -> the adapter's video_layout names ("sana" is the 2026-09-21 spelling of sana_latent)
LAYOUT_BY_MULTIVIEW = {
    "sana": "multiview",
    "sana_latent": "multiview",
    "openwam": "openwam_canvas",
    "sana_pixel": "sana_pixel_canvas",
}
ROPE_CHOICES = ("auto", ROPE_LEGACY, ROPE_ALIGNED, ROPE_INDEPENDENT)
# Observation View descriptor keys of the canvas row (texts in sana_wam_min.openwam_canvas / sana_pixel_canvas)
CANVAS_PROMPTS = {
    "openwam_canvas": ("composite_view", "l_shape", "two_rows"),
    "sana_pixel_canvas": ("composite_view", "tiling"),
}


def strided_video_fps(cfg: dict[str, Any]):
    """``data.extra.robot_sft.video_fps`` (``robotwin_sft`` before 2026-09-24; frames sampled after the observation frame)
    of a training yaml; None = dense video."""

    value = sft_options_from_train_config(cfg).get("video_fps")
    return None if value is None else int(value)


def declared_multiview(cfg: dict[str, Any]):
    """``data.extra.multiview`` lower-cased, or None when the yaml predates it (rwm/mot a26807821)."""

    extra = (((cfg.get("data") or {}).get("extra")) or {})
    value = extra.get("multiview")
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode not in LAYOUT_BY_MULTIVIEW:
        raise ValueError(f"unsupported data.extra.multiview {value!r}; expected sana_latent | sana_pixel | openwam (or sana)")
    return mode


def _data_type_layout(data_type: str):
    if not data_type:
        return None
    if data_type == SANA_PIXEL_DATASET_TYPE:
        return "sana_pixel_canvas"
    return "openwam_canvas" if CANVAS_DATASET_MARKER in data_type else "multiview"


def resolve_video_layout(cfg: dict[str, Any], context_layout: str = CONTEXT_EMBEDDER) -> str:
    """The video layout a checkpoint consumes: ``data.extra.multiview``, else ``model.extra.video_layout``, else what
    ``data.type`` trained.

    Without a ``data.type`` either, the legacy text layout means the canvas (606e48dd9 knew nothing else), the shared
    caption embedder a canvas (the openwam one; the yaml would normally name it), and the ``context_embedder`` layout the
    ``multiview`` strip.
    """

    extra = dict((cfg.get("model") or {}).get("extra") or {})
    data_type = str((cfg.get("data") or {}).get("type") or "")
    typed = _data_type_layout(data_type)
    mode = declared_multiview(cfg)
    if mode is not None:
        layout = LAYOUT_BY_MULTIVIEW[mode]
        # the canvas dataset classes are selected by the mode on the family class since a26807821; a yaml naming a canvas
        # class directly must name the same mode
        if typed is not None and typed != "multiview" and typed != layout:
            raise ValueError(f"data.extra.multiview is {mode!r} but data.type is {data_type!r}")
        if layout == "sana_pixel_canvas":
            sana_pixel_canvas_hw_from_train_config(cfg)          # 320x480 | 320x512 (88a22ba0c); anything else refused
        return layout
    layout = extra.get("video_layout")
    if layout is None:
        if typed is None:
            if context_layout in (LEGACY_ACTION_MLP, SHARED_CAPTION_EMBEDDER):
                return "openwam_canvas"
            return "multiview"
        return typed
    layout = str(layout)
    if typed is not None and (typed != "multiview") != (layout == "openwam_canvas"):
        raise ValueError(
            f"model.extra.video_layout is {layout!r} but data.type is {data_type!r}; a canvas dataset trains "
            "video_layout 'openwam_canvas' and every other dataset the 'multiview' strip"
        )
    return layout


def rope_mode_from_train_config(cfg: dict[str, Any]):
    """``model.extra.rope`` (aligned | independent), None when the yaml predates it (rwm/mot 71ac93f43)."""

    extra = (((cfg.get("model") or {}).get("extra")) or {})
    value = extra.get("rope")
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode not in (ROPE_ALIGNED, ROPE_INDEPENDENT):
        raise ValueError(f"model.extra.rope must be aligned or independent, got {value!r}")
    return mode


def _is_auto(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto", "null", "none"))


def resolve_mot_rope(cfg: dict[str, Any], context_layout: str = CONTEXT_EMBEDDER, requested=None) -> tuple[str, str]:
    """``(rope, label)``: ``aligned`` / ``independent`` from ``model.extra.rope``; undeclared = the era's table --
    ``independent`` once ``data.extra.multiview`` or the shared caption embedder mark a post-42aee4fa9 run, ``legacy``
    otherwise (and always for the 606e48dd9 text layout). ``requested`` (deploy ``rope_mode``) forces a regime; a value
    contradicting a declared key is refused."""

    declared = rope_mode_from_train_config(cfg)
    if context_layout == LEGACY_ACTION_MLP:
        if declared is not None or not _is_auto(requested) and str(requested).strip().lower() != ROPE_LEGACY:
            raise ValueError("the legacy MoT text layout (606e48dd9) predates the RoPE modes")
        return ROPE_LEGACY, "legacy (606e48dd9 layout)"
    if _is_auto(requested):
        if declared is not None:
            return declared, f"{declared} (model.extra.rope)"
        if declared_multiview(cfg) is not None or context_layout == SHARED_CAPTION_EMBEDDER:
            return ROPE_INDEPENDENT, "independent (undeclared; post-42aee4fa9 era: action clock, state 0 / actions 1..S)"
        return ROPE_LEGACY, "legacy (undeclared; pre-42aee4fa9 era: physical clock, strided state+actions from 0)"
    choice = str(requested).strip().lower()
    if choice not in ROPE_CHOICES:
        raise ValueError(f"rope_mode must be one of {ROPE_CHOICES}, got {requested!r}")
    if declared is not None and choice != declared:
        raise ValueError(f"rope_mode {choice!r} contradicts the checkpoint's model.extra.rope {declared!r}")
    return choice, f"{choice} (forced)"


def resolve_mot_text_contract(
    cfg: dict[str, Any], context_layout: str = CONTEXT_EMBEDDER, requested_groups=None, requested_prompt=None
) -> tuple[int, Any, str]:
    """``(text groups, canvas prompt key or None, label)`` of a MoT checkpoint.

    The strip reads G = V + 1 rows. A canvas: the legacy text layout and every canvas yaml without
    ``data.extra.multiview`` read ONE row with the ``composite_view`` descriptor (campaign 19045548 and the NSC f25 canvas
    runs); separate caption embedders with ``data.extra.multiview`` are the 2026-09-21..22 G = 2 payload (composite-view
    row + robot row) unless ``model.extra.rope`` is declared; the shared caption embedder reads ONE row -- ``l_shape``
    (openwam) / ``tiling`` (sana_pixel) until 3ea1f9af9, ``two_rows`` / ``tiling`` once ``model.extra.rope`` is declared.
    Since rwm/mot b010eb9a5 (2026-09-23) every mode has one caption embedder per expert again and a canvas ships that
    same ONE row (``two_rows`` / ``tiling``), which each expert projects through its own embedder. ``model.extra.rope``
    (71ac93f43, required by the live factory since) predates b010eb9a5 and the canvas modes had the shared embedder in
    between, so separate embedders + ``multiview`` + a declared rope can only be the b010eb9a5 contract.
    """

    layout = resolve_video_layout(cfg, context_layout)
    if layout == "multiview":
        if not _is_auto(requested_groups) or not _is_auto(requested_prompt):
            raise ValueError("text_groups / canvas_prompt apply to the canvas layouts only; this checkpoint serves the strip")
        return 4, None, "strip: G = V + 1 = 4 (one row per view, then the robot row)"
    openwam = layout == "openwam_canvas"
    if context_layout == SHARED_CAPTION_EMBEDDER:
        declared = rope_mode_from_train_config(cfg) is not None
        prompt = ("two_rows" if declared else "l_shape") if openwam else "tiling"
        groups, why = 1, "shared caption embedder: ONE row" + (" (2026-09-23 contract)" if declared else " (4e67e1e1d .. 71ac93f43)")
    elif context_layout == LEGACY_ACTION_MLP or declared_multiview(cfg) is None:
        groups, prompt, why = 1, "composite_view", "separate embedders, pre-a26807821 canvas: ONE row read by both experts"
    elif rope_mode_from_train_config(cfg) is not None:
        prompt = "two_rows" if openwam else "tiling"
        groups, why = 1, "separate embedders, ONE row projected by each expert's own embedder (b010eb9a5 on)"
    else:
        groups, prompt, why = 2, "composite_view", "separate embedders with data.extra.multiview (a26807821 .. 2b4a4dc8d): G = 2"
    if not _is_auto(requested_groups):
        forced = int(requested_groups)
        if forced not in (1, 2):
            raise ValueError(f"text_groups must be auto, 1 or 2, got {requested_groups!r}")
        if forced != 1 and context_layout in (SHARED_CAPTION_EMBEDDER, LEGACY_ACTION_MLP):
            raise ValueError("this checkpoint's text path reads ONE prompt row (text_groups 1)")
        groups, why = forced, why + f"; text_groups forced to {forced}"
    if not _is_auto(requested_prompt):
        key = str(requested_prompt).strip().lower()
        if key not in CANVAS_PROMPTS[layout]:
            raise ValueError(f"canvas_prompt for {layout} must be one of {CANVAS_PROMPTS[layout]}, got {requested_prompt!r}")
        prompt, why = key, why + f"; canvas_prompt forced to {key}"
    return groups, prompt, why


def require_mot_factory(cfg: dict[str, Any]) -> str:
    """Return ``model.model`` after checking it names the MoT factory this mirror implements."""

    name = str((cfg.get("model") or {}).get("model") or "")
    if name != MOT_FACTORY_NAME:
        raise ValueError(f"the SANA_MOT adapter serves {MOT_FACTORY_NAME!r} checkpoints only; the training yaml names {name!r}")
    return name


def mot_config_from_train_config(
    cfg: dict[str, Any],
    context_layout: str = CONTEXT_EMBEDDER,
    *,
    rope: Any = "auto",
    canvas_text_groups: Any = None,
) -> MoTConfig:
    """Resolve the MoT architecture exactly as the live factory chain does.

    Trunk: ``_5B_P1_D36`` dims (depth 32, hidden 2560, 20 heads, P1), latent side ``image_size // vae_stride[-1]``,
    ``vae_latent_dim`` channels, text dims from ``text_encoder``, softmax placement from ``softmax_ratio`` (or explicit
    ``softmax_layer_indices``), temporal compression from ``vae_stride[0]``. MoT knobs from ``model.extra``; absent
    knobs take the constructor defaults (hidden 1024, mlp 4.0, 8 cross heads, theta 1e4, AttnRes block 8, state as
    self-attention token). The multiview RoPE layout keys of the single-stream line are not part of this model.
    """

    require_mot_factory(cfg)
    layout = resolve_video_layout(cfg, context_layout)
    rope_mode = resolve_mot_rope(cfg, context_layout, None if rope == "auto" else rope)[0]
    groups = resolve_mot_text_contract(cfg, context_layout, canvas_text_groups)[0]
    model_cfg = cfg["model"]
    vae_cfg = cfg["vae"]
    text_cfg = cfg["text_encoder"]
    sched_cfg = cfg["scheduler"]
    duck_config = SimpleNamespace(
        model=SimpleNamespace(multiview_spatial_rope_layout="local_reset", multiview_spatial_rope_tile_shape=(15, 30)),
        vae=SimpleNamespace(vae_stride=list(vae_cfg["vae_stride"])),
    )
    video = PolicyConfig.from_sana_kwargs(
        depth=FACTORY_DEPTH,
        hidden_size=FACTORY_HIDDEN_SIZE,
        patch_size=FACTORY_PATCH_SIZE,
        num_heads=FACTORY_NUM_HEADS,
        input_size=latent_input_size(cfg),
        in_channels=int(vae_cfg["vae_latent_dim"]),
        caption_channels=int(text_cfg.get("caption_channels", DEFAULT_CAPTION_CHANNELS)),
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
        use_dual_attn_res_routing=bool(model_cfg.get("use_dual_attn_res_routing", False)),
        cross_attn_image_embeds=bool(model_cfg.get("cross_attn_image_embeds", False)),
        rope_fhw_dim=model_cfg.get("rope_fhw_dim"),
        ffn_type=model_cfg["ffn_type"],
        linear_attn_type=model_cfg["linear_attn_type"],
        softmax_attn_type=model_cfg["softmax_attn_type"],
        use_fp32_attention=bool(model_cfg["fp32_attention"]),
        config=duck_config,
    )
    extra = dict(model_cfg.get("extra") or {})
    knobs = {key: extra[key] for key in MOT_EXTRA_KEYS if key in extra}
    return MoTConfig(
        video=video,
        action_hidden_size=int(knobs.get("action_hidden_size", 1024)),
        action_mlp_ratio=float(knobs.get("action_mlp_ratio", 4.0)),
        action_cross_attn_heads=int(knobs.get("action_cross_attn_heads", 8)),
        action_rope_theta=float(knobs.get("action_rope_theta", 10000.0)),
        action_attn_res_block_size=int(knobs.get("action_attn_res_block_size", 8)),
        action_state_as_context=bool(knobs.get("action_state_as_context", False)),
        video_layout=resolve_video_layout(cfg, context_layout),
        multiview_spatial_rope_layout=str(knobs.get("multiview_spatial_rope_layout", "semantic_2x2")),
        # the sana_latent strip's tile: (15, 30) before rwm/mot 66ded97a6 (= zekai-merge 6da565230), (8, 16) since, no yaml key;
        # a model.extra key wins, else the SANA_WAM rule (robot_sft configs postdate the change)
        multiview_spatial_rope_tile_shape=tuple(
            int(v) for v in knobs.get("multiview_spatial_rope_tile_shape", strip_spatial_rope_tile_shape_from_train_config(cfg))
        ),
        context_layout=str(context_layout),
        canvas_text_groups=groups if layout != "multiview" else 1,
        rope=rope_mode,
        separate_action_schedule=any(
            ((cfg.get("scheduler") or {}).get(key)) is not None for key in ("action_flow_shift", "inference_action_flow_shift")
        ),
    ).validate()


__all__ = [
    "CANVAS_DATASET_MARKER",
    "CANVAS_PROMPTS",
    "DEFAULT_CAPTION_CHANNELS",
    "LAYOUT_BY_MULTIVIEW",
    "MOT_EXTRA_KEYS",
    "ROPE_CHOICES",
    "SANA_PIXEL_DATASET_TYPE",
    "declared_multiview",
    "resolve_mot_rope",
    "resolve_mot_text_contract",
    "resolve_video_layout",
    "rope_mode_from_train_config",
    "strided_video_fps",
    "VALIDATED_CFG_SCALE",
    "load_train_config",
    "mot_config_from_train_config",
    "require_mot_factory",
    "sampling_defaults_from_train_config",
]
