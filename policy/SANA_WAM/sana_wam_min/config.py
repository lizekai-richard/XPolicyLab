"""Train-yaml to PolicyConfig resolution for the SANA unified policy.

The training yaml is the only description of the checkpointed architecture
that travels with a run.  This module reproduces the live builder chain
(image_size // vae_stride[-1] -> model_video_init_config -> factory) with
plain dict access so no Sana config class is needed, and reads from the same
yaml the line-level contracts the deploy session needs: the visual front-end
(``data.extra.multiview``: three per-view streams packed as a strip, the
OpenWAM head-over-wrists canvas, or the sana_pixel 2x2 pixel canvas), the RoPE
mode (``model.extra.rope``), the prompt contract of the canvas modes and the
action-mode / target-mode words of the prompt.

The yaml keys changed over time; the resolvers below map every era a trained
checkpoint can come from (rwm/zekai-merge, rwm/openwam, rwm/strided_video):

* ``data.extra.multiview`` (2026-09-21 ``b31137206``; values ``sana | openwam``,
  renamed ``sana_latent | sana_pixel | openwam`` on 2026-09-22 ``b71474ad1``),
  its short-lived predecessor ``data.extra.video_layout`` (``f8eaad39a``), and
  before both the model factory + ``data.type`` (the rwm/openwam canvas class).
* ``model.extra.rope`` (2026-09-23 ``b13415841``); undeclared = the tables of
  the run's own era (dense ``aligned``; strided independent with the first
  action at 0 before ``8a61ae18a``, at 1 after it).
* The canvas modes' text contract: G = 2 (view row + robot row) with the
  "composite view" descriptor from ``b31137206`` to ``2a10d0d75``, then ONE
  shared row (G = 1) whose descriptor changed on ``92b9e64d1``.
* The SFT option block ``data.extra.robot_sft`` (2026-09-24 ``875b3f659``),
  ``data.extra.robotwin_sft`` in every config frozen before it (same options).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import yaml

from . import pixels
from .frame_stride import parse_video_fps, video_frame_stride_from_video_fps
from .policy_model.config import SANA_PIXEL_PAD_MODES, PolicyConfig
from .sana_pixel_canvas import sana_pixel_canvas_hw

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
# sana_pixel_canvas: the three cameras tiled in PIXEL space into one 320x480 2x2 canvas (160x240 tiles on the semantic
#   quadrants, the unclaimed top-right quadrant black), encoded once (``data.extra.multiview: sana_pixel``, 2026-09-22).
VISUAL_LAYOUT_THREE_VIEW_STRIP = "three_view_strip"
VISUAL_LAYOUT_OPENWAM_CANVAS = "openwam_canvas"
VISUAL_LAYOUT_SANA_PIXEL_CANVAS = "sana_pixel_canvas"
VISUAL_LAYOUTS = (VISUAL_LAYOUT_THREE_VIEW_STRIP, VISUAL_LAYOUT_OPENWAM_CANVAS, VISUAL_LAYOUT_SANA_PIXEL_CANVAS)
CANVAS_LAYOUTS = (VISUAL_LAYOUT_OPENWAM_CANVAS, VISUAL_LAYOUT_SANA_PIXEL_CANVAS)
THREE_VIEW_POLICY_FACTORY = (
    "SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy_5B_P1_D36"
)
OPENWAM_CANVAS_POLICY_FACTORY = "SanaRWMOpenWAMCanvasPolicy_5B_P1_D36"
OPENWAM_CANVAS_DATASET_TYPE = "RoboDojoOpenWAMCanvasSFTDataset"
SANA_PIXEL_CANVAS_DATASET_TYPE = "RoboDojoSanaPixelCanvasSFTDataset"
OPENWAM_CANVAS_LAYOUT_ID = "openwam_lshape_rgb_v1"
OPENWAM_CANVAS_ENCODE_MODE = "joint_rgb_canvas"
OPENWAM_CANVAS_ASPECT_RATIO_TYPE = "ASPECT_RATIO_OPENWAM_LSHAPE_384_320"
SANA_PIXEL_ASPECT_RATIO_TYPE = "ASPECT_RATIO_SANA_PIXEL_2X2_320_480"

# data.extra.multiview (dev/rwm/diffusion/data/multiview.py). ``sana`` is the 2026-09-21 spelling of sana_latent (no
# alias in the live code, whose yaml parser now rejects it; a checkpoint of that day still carries it).
MULTIVIEW_SANA_LATENT = "sana_latent"
MULTIVIEW_SANA_PIXEL = "sana_pixel"
MULTIVIEW_OPENWAM = "openwam"
MULTIVIEW_MODES = (MULTIVIEW_SANA_LATENT, MULTIVIEW_SANA_PIXEL, MULTIVIEW_OPENWAM)
LEGACY_MULTIVIEW_ALIASES = {"sana": MULTIVIEW_SANA_LATENT}
LAYOUT_BY_MULTIVIEW = {
    MULTIVIEW_SANA_LATENT: VISUAL_LAYOUT_THREE_VIEW_STRIP,
    MULTIVIEW_SANA_PIXEL: VISUAL_LAYOUT_SANA_PIXEL_CANVAS,
    MULTIVIEW_OPENWAM: VISUAL_LAYOUT_OPENWAM_CANVAS,
}

# model.extra.rope (layers/wan_mrope.py ROPE_MODES, 2026-09-23).
ROPE_ALIGNED = "aligned"
ROPE_INDEPENDENT = "independent"
ROPE_MODES = (ROPE_ALIGNED, ROPE_INDEPENDENT)

# Canvas prompt descriptors ("Observation View" sentence of the one canvas row), by era; the text lives in
# openwam_canvas.py / sana_pixel_canvas.py.
#   openwam: composite_view (rwm/openwam .. zekai-merge 2a10d0d75, also the G = 2 view row), l_shape (2a10d0d75 ..
#   92b9e64d1), two_rows (92b9e64d1 on); sana_pixel: composite_view (the G = 2 smokes of b71474ad1 .. e4bb170ad),
#   tiling (e4bb170ad on).
CANVAS_PROMPT_COMPOSITE_VIEW = "composite_view"
CANVAS_PROMPT_L_SHAPE = "l_shape"
CANVAS_PROMPT_TWO_ROWS = "two_rows"
CANVAS_PROMPT_TILING = "tiling"
CANVAS_PROMPTS = {
    VISUAL_LAYOUT_OPENWAM_CANVAS: (CANVAS_PROMPT_COMPOSITE_VIEW, CANVAS_PROMPT_L_SHAPE, CANVAS_PROMPT_TWO_ROWS),
    VISUAL_LAYOUT_SANA_PIXEL_CANVAS: (CANVAS_PROMPT_COMPOSITE_VIEW, CANVAS_PROMPT_TILING),
}

# robot_base_eef meaning (docs/eef_only_action_mode.md): ``full`` = the 32 dual_arm32 slots (joints + EEF pose +
# grippers, every run before 2026-09-20), ``eef_only`` = EEF pose + grippers (20 slots), joints neither fed nor supervised.
ROBOT_BASE_EEF_FULL = "full"
ROBOT_BASE_EEF_ONLY = "eef_only"
ROBOT_BASE_EEF_LAYOUTS = (ROBOT_BASE_EEF_FULL, ROBOT_BASE_EEF_ONLY)

# The SFT option block of data.extra (Sana dev/rwm/diffusion/data/sft_options.py): renamed robotwin_sft -> robot_sft on
# 2026-09-24 (875b3f659, same options inside); the configs frozen beside older checkpoints keep the retired name.
SFT_OPTIONS_KEY = "robot_sft"
RETIRED_SFT_OPTIONS_KEY = "robotwin_sft"

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


def declared_multiview(cfg: dict[str, Any]) -> Optional[str]:
    """``data.extra.multiview`` (or its 2026-09-21 predecessor ``data.extra.video_layout``), normalized; None if absent."""

    extra = ((cfg.get("data") or {}).get("extra")) or {}
    for key in ("multiview", "video_layout"):
        raw = extra.get(key)
        if raw is None:
            continue
        mode = str(raw).strip().lower()
        mode = LEGACY_MULTIVIEW_ALIASES.get(mode, mode)
        if mode not in MULTIVIEW_MODES:
            raise ValueError(f"unsupported data.extra.{key} {raw!r}; expected one of {MULTIVIEW_MODES} (or 'sana')")
        return mode
    return None


def _check_openwam_canvas_options(data_cfg: dict[str, Any]) -> None:
    """The one implemented canvas layout / encode mode / aspect bucket."""

    options = dict(((data_cfg.get("extra") or {}).get("openwam_canvas")) or {})
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
            f"the openwam canvas trains on {OPENWAM_CANVAS_ASPECT_RATIO_TYPE}, the yaml declares {aspect!r}"
        )


def is_openwam_canvas_policy_class(cfg: dict[str, Any]) -> bool:
    """True for the rwm/openwam canvas policy class (``SanaRWMOpenWAMCanvasPolicy_5B_P1_D36``, deleted on zekai-merge
    2026-09-21): its own shared-prompt (G = 1) / state_as_cross_attention contract, dense video only.

    Named directly in ``model.model``, or -- zekai-merge f8eaad39a .. b31137206 (2026-09-21 12:46 .. 14:30 PDT) -- chosen
    by the standard factory name when ``data.extra.video_layout: openwam`` (``policy_class_for_video_layout``); the
    ``data.extra.multiview`` key that replaced it serves the canvas on the one policy class instead."""

    if str(cfg["model"]["model"]) == OPENWAM_CANVAS_POLICY_FACTORY:
        return True
    extra = ((cfg.get("data") or {}).get("extra")) or {}
    layout = extra.get("video_layout")
    return extra.get("multiview") is None and layout is not None and str(layout).strip().lower() == MULTIVIEW_OPENWAM


def resolve_visual_layout(cfg: dict[str, Any]) -> str:
    """The visual front-end the checkpoint was trained with.

    Since 2026-09-21 ONE policy class serves every mode and ``data.extra.multiview`` picks the front-end
    (``sana_latent`` strip, ``openwam`` canvas, ``sana_pixel`` canvas; absent = sana_latent). The rwm/openwam canvas
    class (``SanaRWMOpenWAMCanvasPolicy_5B_P1_D36``) is always the openwam canvas. A canvas dataset ``data.type`` and a
    declared mode must agree, and each canvas mode must use its one implemented aspect bucket / layout.
    """

    model_name = str(cfg["model"]["model"])
    data_cfg = cfg["data"]
    data_type = str(data_cfg.get("type") or "")
    mode = declared_multiview(cfg)
    extra = data_cfg.get("extra") or {}
    if model_name == OPENWAM_CANVAS_POLICY_FACTORY:
        if mode not in (None, MULTIVIEW_OPENWAM):
            raise ValueError(f"model.model {model_name} is the OpenWAM canvas policy but the yaml declares multiview {mode!r}")
        if data_type != OPENWAM_CANVAS_DATASET_TYPE and mode != MULTIVIEW_OPENWAM:
            raise ValueError(
                f"model.model {model_name} needs data.type {OPENWAM_CANVAS_DATASET_TYPE} (or data.extra.video_layout / "
                f"multiview openwam), got {data_type!r}"
            )
        _check_openwam_canvas_options(data_cfg)
        return VISUAL_LAYOUT_OPENWAM_CANVAS
    if model_name != THREE_VIEW_POLICY_FACTORY:
        raise ValueError(
            f"unsupported model.model {model_name!r}; this adapter serves {THREE_VIEW_POLICY_FACTORY} "
            f"and {OPENWAM_CANVAS_POLICY_FACTORY}"
        )
    typed = {OPENWAM_CANVAS_DATASET_TYPE: MULTIVIEW_OPENWAM, SANA_PIXEL_CANVAS_DATASET_TYPE: MULTIVIEW_SANA_PIXEL}.get(data_type)
    if mode is None:
        if typed is None and extra.get("openwam_canvas"):
            raise ValueError(
                f"model.model {model_name} with data.type {data_type!r} declares openwam_canvas options but no "
                "data.extra.multiview; the pre-2026-09-21 canvas line used the OpenWAM canvas policy class"
            )
        mode = typed or MULTIVIEW_SANA_LATENT
    elif typed is not None and typed != mode:
        raise ValueError(f"data.type {data_type!r} serves multiview {typed!r} but the yaml declares {mode!r}")
    if mode == MULTIVIEW_OPENWAM:
        _check_openwam_canvas_options(data_cfg)
    elif mode == MULTIVIEW_SANA_PIXEL:
        sana_pixel_canvas_hw(data_cfg.get("aspect_ratio_type"))      # 320x480 | 320x512 (88a22ba0c); anything else refused
    elif extra.get("openwam_canvas"):
        raise ValueError(f"multiview {mode!r} yaml carries openwam_canvas options {extra.get('openwam_canvas')!r}")
    return LAYOUT_BY_MULTIVIEW[mode]


def sana_pixel_canvas_hw_from_train_config(cfg: dict[str, Any]) -> tuple[int, int]:
    """The sana_pixel canvas ``(height, width)`` of a training yaml: ``data.aspect_ratio_type`` (320x480 or 320x512)."""

    return sana_pixel_canvas_hw((cfg.get("data") or {}).get("aspect_ratio_type"))


def rope_mode_from_train_config(cfg: dict[str, Any]) -> Optional[str]:
    """``model.extra.rope`` (aligned | independent), None when the yaml predates the key (2026-09-23 ``b13415841``)."""

    extra = ((cfg.get("model") or {}).get("extra")) or {}
    raw = extra.get("rope")
    if raw is None:
        return None
    mode = str(raw).strip().lower()
    if mode not in ROPE_MODES:
        raise ValueError(f"model.extra.rope must be one of {ROPE_MODES}, got {raw!r}")
    return mode


def legacy_strided_action_origin(cfg: dict[str, Any]) -> int:
    """First local position of the independent action RoPE a strided checkpoint WITHOUT ``model.extra.rope`` trained
    with: 0 before zekai-merge ``8a61ae18a`` (2026-09-21 17:30 PDT: state and the first action both at 0; every
    rwm/strided_video and NSC f33fps8 run), 1 after it. The yaml has no key of that day; ``data.extra.multiview``
    (``b31137206``, three hours earlier) is the nearest marker, so a yaml carrying it is taken as post-``8a61ae18a``
    (campaign 19085339, the strided openwam canvas, pinned at ``a5c7909ae``). Override with ``rope_mode``."""

    extra = ((cfg.get("data") or {}).get("extra")) or {}
    return 1 if any(extra.get(key) is not None for key in ("multiview", "video_layout")) else 0


def resolve_rope_contract(cfg: dict[str, Any], requested: Optional[str] = None) -> tuple[Optional[str], int, str]:
    """``(PolicyConfig.rope, legacy_strided_action_origin, label)`` for a checkpoint.

    ``requested`` (deploy ``rope_mode``): ``auto`` / None = the yaml's ``model.extra.rope`` when declared, else the
    tables of the run's era (dense = aligned, strided = independent from ``legacy_strided_action_origin``);
    ``aligned`` / ``independent`` force a declared mode; ``independent_from0`` forces the pre-``8a61ae18a`` strided
    table (state and first action at 0). An explicit value that contradicts a declared key is refused.
    """

    declared = rope_mode_from_train_config(cfg)
    choice = "auto" if _is_auto(requested) else str(requested).strip().lower()
    choices = ("auto",) + ROPE_MODES + ("independent_from0",)
    if choice not in choices:
        raise ValueError(f"rope_mode must be one of {choices}, got {requested!r}")
    if choice == "auto":
        if declared is not None:
            return declared, 1, f"{declared} (model.extra.rope)"
        origin = legacy_strided_action_origin(cfg)
        return None, origin, f"undeclared: dense aligned, strided independent from {origin}"
    if declared is not None and choice != declared:
        raise ValueError(f"rope_mode {choice!r} contradicts the checkpoint's model.extra.rope {declared!r}")
    if choice == "independent_from0":
        return None, 0, "forced: dense aligned, strided independent from 0"
    return choice, 1, f"{choice} (forced)"


def _is_auto(value: Any) -> bool:
    """None / ``auto`` / empty / ``null`` / ``none`` (the spellings a deploy override can deliver) mean "resolve"."""

    return value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto", "null", "none"))


def resolve_canvas_text_contract(
    cfg: dict[str, Any],
    requested_groups: Any = None,
    requested_prompt: Optional[str] = None,
) -> tuple[int, Optional[str], str]:
    """``(text groups G, canvas prompt key, label)`` of the checkpoint's language conditioning.

    The strip (``sana_latent``) always carries G = V + 1 = 4 rows and no canvas prompt. The rwm/openwam canvas class
    carries ONE shared row with the ``composite_view`` descriptor. The canvas modes of the one policy class shipped,
    by era: G = 2 (the composite-view row for the video, the robot row for the action tail) from ``b31137206`` to
    ``2a10d0d75`` (openwam) / ``b71474ad1`` to ``e4bb170ad`` (sana_pixel), then ONE row (G = 1) naming the layout.
    ``auto``: a yaml with ``model.extra.rope`` (2026-09-23) is the current contract (G = 1, ``two_rows`` / ``tiling``);
    an openwam yaml without it is taken as the G = 2 ``composite_view`` payload of campaign 19085339 (``a5c7909ae``);
    a sana_pixel yaml without it as G = 1 ``tiling`` (no sana_pixel run trained on the G = 2 smokes' payload).
    ``text_groups`` / ``canvas_prompt`` override either part.
    """

    layout = resolve_visual_layout(cfg)
    if layout == VISUAL_LAYOUT_THREE_VIEW_STRIP:
        if not _is_auto(requested_groups) or not _is_auto(requested_prompt):
            raise ValueError("text_groups / canvas_prompt apply to the canvas modes only; this checkpoint serves the strip")
        return 4, None, "strip: G = V + 1 = 4 (one row per view, then the robot row)"
    if is_openwam_canvas_policy_class(cfg):
        groups, prompt, why = 1, CANVAS_PROMPT_COMPOSITE_VIEW, "rwm/openwam canvas class"
    elif rope_mode_from_train_config(cfg) is not None:
        prompt = CANVAS_PROMPT_TWO_ROWS if layout == VISUAL_LAYOUT_OPENWAM_CANVAS else CANVAS_PROMPT_TILING
        groups, why = 1, "model.extra.rope declared (2026-09-23 contract)"
    elif layout == VISUAL_LAYOUT_OPENWAM_CANVAS:
        groups, prompt, why = 2, CANVAS_PROMPT_COMPOSITE_VIEW, "openwam without model.extra.rope (b31137206..2a10d0d75 payload)"
    else:
        groups, prompt, why = 1, CANVAS_PROMPT_TILING, "sana_pixel without model.extra.rope (e4bb170ad..b13415841 payload)"
    if not _is_auto(requested_groups):
        forced = int(requested_groups)
        if forced not in (1, 2):
            raise ValueError(f"text_groups must be auto, 1 or 2, got {requested_groups!r}")
        if is_openwam_canvas_policy_class(cfg) and forced != 1:
            raise ValueError("the rwm/openwam canvas policy class takes ONE shared prompt (G = 1)")
        groups, why = forced, why + f"; text_groups forced to {forced}"
    if not _is_auto(requested_prompt):
        key = str(requested_prompt).strip().lower()
        if key not in CANVAS_PROMPTS[layout]:
            raise ValueError(f"canvas_prompt for {layout} must be one of {CANVAS_PROMPTS[layout]}, got {requested_prompt!r}")
        prompt, why = key, why + f"; canvas_prompt forced to {key}"
    return groups, prompt, why


def state_as_cross_attention_from_train_config(cfg: dict[str, Any]) -> bool:
    """``model.extra.state_as_cross_attention`` (default False); only the rwm/openwam canvas policy class defines it
    (the knob was dropped with the class on 2026-09-21)."""

    extra = cfg["model"].get("extra") or {}
    enabled = bool(extra.get("state_as_cross_attention", False))
    if enabled and not is_openwam_canvas_policy_class(cfg):
        raise ValueError("model.extra.state_as_cross_attention is only defined for the rwm/openwam canvas policy class")
    return enabled


def robot_base_eef_layout_from_train_config(cfg: dict[str, Any], normalization: Any = None) -> str:
    """``full`` (32 slots: joints + EEF + grippers) or ``eef_only`` (EEF + grippers) for a robot_base_eef checkpoint.

    ``robot_base_eef`` became EEF-only on 2026-09-20 (zekai-merge ``dc902ddce`` / rwm/mot ``2434e9199``) with nothing in
    the yaml naming it. Three later markers identify a run trained after that change: a normalization artifact of the
    2026-09-20 forced scheme (``gripper_normalization`` / ``rotation_normalization`` stamped), ``data.extra.multiview``
    (2026-09-21) or ``model.extra.rope`` (2026-09-23). Without any of them the run is the 32-slot meaning (the
    2026-09-10 R12 line, the NSC eefabs runs). Override with the deploy key ``robot_base_eef_layout``."""

    extra = ((cfg.get("data") or {}).get("extra")) or {}
    markers = any(extra.get(key) is not None for key in ("multiview", "video_layout"))
    markers = markers or rope_mode_from_train_config(cfg) is not None
    if normalization is not None:
        markers = markers or getattr(normalization, "gripper_normalization", None) is not None
        markers = markers or getattr(normalization, "rotation_normalization", None) is not None
    return ROBOT_BASE_EEF_ONLY if markers else ROBOT_BASE_EEF_FULL


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


# The sana_latent strip's semantic 2x2 spatial RoPE tile (dev/rwm/diffusion/multiview_utils.py
# MULTIVIEW_SPATIAL_ROPE_TILE_SHAPE): (15, 30) until rwm/zekai-merge 6da565230 (2026-09-24), (8, 16) since -- 256x320 views
# = 8x10 latents, centred at x + 3 in each quadrant ("rwm/yuyang's SFT tile"). It is a fixed code constant with NO yaml key
# since then (pyrallis rejects model.multiview_spatial_rope_*), so a frozen config cannot say which one it trained with. The
# SFT options block tells: data.extra.robot_sft exists only since 875b3f659, a descendant of 6da565230, so a robot_sft config
# trained on (8, 16); every earlier strip checkpoint (R18, R19, the 157k donor, the NSC runs) ran on (15, 30).
LEGACY_STRIP_SPATIAL_ROPE_TILE_SHAPE = (15, 30)
STRIP_SPATIAL_ROPE_TILE_SHAPE = (8, 16)


def strip_spatial_rope_tile_shape_from_train_config(cfg: dict[str, Any]) -> tuple[int, int]:
    """The semantic 2x2 tile a strip checkpoint trained with: a tile the yaml spells out (old trainer dumps) wins, else
    (8, 16) for a ``data.extra.robot_sft`` config (tree at or after 6da565230), else the legacy (15, 30)."""

    declared = (cfg.get("model") or {}).get("multiview_spatial_rope_tile_shape")
    if declared is not None:
        return tuple(int(value) for value in declared)
    extra = ((cfg.get("data") or {}).get("extra")) or {}
    return STRIP_SPATIAL_ROPE_TILE_SHAPE if SFT_OPTIONS_KEY in extra else LEGACY_STRIP_SPATIAL_ROPE_TILE_SHAPE


def sft_options_from_train_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The SFT option block of a training yaml under either name, as Sana's ``sft_options_block`` reads it:
    ``data.extra.robot_sft`` (2026-09-24 ``875b3f659``), else the retired ``data.extra.robotwin_sft`` of the configs
    frozen before the rename; ``{}`` when the yaml declares neither. Declaring both is refused (ambiguous)."""

    extra = ((cfg.get("data") or {}).get("extra")) or {}
    current, retired = extra.get(SFT_OPTIONS_KEY), extra.get(RETIRED_SFT_OPTIONS_KEY)
    if current is not None and retired is not None:
        raise ValueError(
            f"data.extra declares both {SFT_OPTIONS_KEY} and the retired {RETIRED_SFT_OPTIONS_KEY}; keep only {SFT_OPTIONS_KEY}"
        )
    block = current if current is not None else retired
    if block is not None and not isinstance(block, dict):
        raise ValueError(f"data.extra.{SFT_OPTIONS_KEY if current is not None else RETIRED_SFT_OPTIONS_KEY} must be a mapping")
    return block or {}


def view_resize_from_train_config(cfg: dict[str, Any], requested: Optional[str] = None) -> tuple[str, str]:
    """How the checkpoint's SFT views reached their bucket: ``("crop" | "stretch", source)``.

    rwm/zekai-merge 1061b16f0 / rwm/mot 034e55dca (2026-09-27) stretched every SFT view whole (``data.extra.robot_sft.
    view_resize: stretch``); 77cf81fbf / f06615b2f then removed the switch -- stretch with NO yaml key -- so a yaml
    without the key is either a stretch checkpoint (77cf81fbf on) or a crop one (every tree before 1061b16f0), which only
    the training revision tells. Order: the yaml's own key (a deploy value that disagrees is refused); else the deploy
    ``view_resize`` (``crop`` for a checkpoint trained before 1061b16f0 / 034e55dca); else ``stretch``, the only SFT
    resize upstream (default since 2026-09-29, user)."""

    declared = sft_options_from_train_config(cfg).get("view_resize")
    wanted = None if requested is None or str(requested).strip().lower() in ("", "auto", "none", "null") else requested
    if declared is not None:
        mode = pixels.validate_view_resize(declared)
        if wanted is not None and pixels.validate_view_resize(wanted) != mode:
            raise ValueError(f"deploy view_resize={wanted!r} contradicts the training yaml's data.extra.robot_sft.view_resize {declared!r}")
        return mode, "yaml"
    if wanted is not None:
        return pixels.validate_view_resize(wanted), "deploy"
    return "stretch", "default"




def sana_pixel_pad_from_train_config(cfg: dict[str, Any]) -> str:
    """``model.extra.sana_pixel_pad`` (rwm/zekai-merge 319d3f666 / rwm/mot 2a0c69d1f): ``unmasked`` (default: the black
    quadrant's latent cells stay in the sequence) or ``masked`` (they are dropped from the policy's token sequence; the
    SANA_WAM mirror serves it on the one-view 320x512 sana_pixel canvas, SANA_MOT refuses it). ``masked`` without
    ``data.extra.multiview: sana_pixel`` is refused, as the live layout module does."""

    value = ((cfg.get("model") or {}).get("extra") or {}).get("sana_pixel_pad")
    mode = "unmasked" if value is None else str(value).strip().lower()
    if mode not in SANA_PIXEL_PAD_MODES:
        raise ValueError(f"model.extra.sana_pixel_pad must be one of {SANA_PIXEL_PAD_MODES}, got {value!r}")
    multiview = ((cfg.get("data") or {}).get("extra") or {}).get("multiview")
    if mode == "masked" and str(multiview).strip().lower() != "sana_pixel":
        raise ValueError(f"model.extra.sana_pixel_pad: masked needs data.extra.multiview: sana_pixel, got {multiview!r}")
    return mode


def video_fps_from_train_config(cfg: dict[str, Any]) -> Optional[int]:
    """``data.extra.robot_sft.video_fps`` (``robotwin_sft`` before 2026-09-24; rwm/strided_video): the video frames
    sampled after the observation frame of every training window; absent -> None, the dense video of the historical lines."""

    return parse_video_fps(sft_options_from_train_config(cfg).get("video_fps"))


def video_frame_stride_from_train_config(cfg: dict[str, Any]) -> int:
    """The video frame stride the checkpoint trained with: 1 without ``video_fps``, else ``(tier_num_frames - 1) / video_fps``
    exactly as the dataset derives it (``RoboTwin2SFTDataset``: the stride is defined on the locked window length, which
    ``data.num_frames`` must equal)."""

    video_fps = video_fps_from_train_config(cfg)
    if video_fps is None:
        return 1
    options = sft_options_from_train_config(cfg)
    rows = options.get("tier_num_frames")
    if rows is None:
        raise ValueError("data.extra.robot_sft.video_fps needs tier_num_frames: the frame stride is (rows - 1) / video_fps")
    num_frames = cfg["data"].get("num_frames")
    if num_frames is not None and int(num_frames) != int(rows):
        raise ValueError(
            f"data.num_frames {num_frames} != data.extra.robot_sft.tier_num_frames {rows}; the video frame stride "
            "is defined on the locked window length"
        )
    return video_frame_stride_from_video_fps(int(rows), video_fps)


def policy_config_from_train_config(
    cfg: dict[str, Any], rope: Optional[str] = "auto", legacy_origin: Optional[int] = None
) -> PolicyConfig:
    """Resolve the PolicyConfig exactly as the live builder chain does.

    Mirrors ``model_video_init_config`` + the factory: trunk dims from the
    factory, latent/text dims from the yaml, softmax layer indices from
    ``softmax_ratio`` when ``softmax_layer_indices`` is null, rope layout and
    tile from ``model.*`` (the live ``ModelConfig`` defaults when the yaml does
    not set them), temporal compression from ``vae.vae_stride[0]``, and the
    canvas policy's ``shared_prompt`` / ``state_as_cross_attention`` from
    ``model.model`` / ``model.extra``.  ``y_norm_scale_factor`` and
    ``timestep_norm_scale_factor`` are not forwarded by the video builder, so
    the constructor defaults (1.0) apply. ``rope`` / ``legacy_origin``: the
    resolved RoPE contract (:func:`resolve_rope_contract`); ``"auto"`` resolves
    it from the yaml here.
    """

    if rope == "auto":
        rope, legacy_origin, _ = resolve_rope_contract(cfg, None)
    elif legacy_origin is None:
        legacy_origin = legacy_strided_action_origin(cfg)
    model_cfg = cfg["model"]
    vae_cfg = cfg["vae"]
    text_cfg = cfg["text_encoder"]
    sched_cfg = cfg["scheduler"]
    resolve_visual_layout(cfg)  # refuses an inconsistent or unsupported front-end before anything is built
    duck_config = SimpleNamespace(
        model=SimpleNamespace(
            # absent since 2026-09-21 (4798b12e7): V > 1 is always the semantic 2x2 tiling
            multiview_spatial_rope_layout=str(
                _model_value(model_cfg, "multiview_spatial_rope_layout", "semantic_2x2")
            ),
            # the sana_latent strip's tile: (15, 30) before rwm/zekai-merge 6da565230, (8, 16) since (no yaml key)
            multiview_spatial_rope_tile_shape=strip_spatial_rope_tile_shape_from_train_config(cfg),
        ),
        vae=SimpleNamespace(vae_stride=list(vae_cfg["vae_stride"])),
        scheduler=SimpleNamespace(
            action_flow_shift=sched_cfg.get("action_flow_shift"),
            inference_action_flow_shift=sched_cfg.get("inference_action_flow_shift"),
        ),
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
        # the rwm/openwam canvas class's G = 1 contract; the one policy class decides its spans from the text's G
        shared_prompt=is_openwam_canvas_policy_class(cfg),
        state_as_cross_attention=state_as_cross_attention_from_train_config(cfg),
        rope=rope,
        legacy_strided_action_origin=int(legacy_origin),
        sana_pixel_pad=sana_pixel_pad_from_train_config(cfg),
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
    "SANA_PIXEL_PAD_MODES",
    "CANVAS_LAYOUTS",
    "CANVAS_PROMPTS",
    "CANVAS_PROMPT_COMPOSITE_VIEW",
    "CANVAS_PROMPT_L_SHAPE",
    "CANVAS_PROMPT_TILING",
    "CANVAS_PROMPT_TWO_ROWS",
    "DEFAULT_EEF_TARGET_MODE",
    "FACTORY_DEPTH",
    "FACTORY_HIDDEN_SIZE",
    "FACTORY_NUM_HEADS",
    "FACTORY_PATCH_SIZE",
    "MULTIVIEW_MODES",
    "MULTIVIEW_OPENWAM",
    "MULTIVIEW_SANA_LATENT",
    "MULTIVIEW_SANA_PIXEL",
    "OPENWAM_CANVAS_ASPECT_RATIO_TYPE",
    "OPENWAM_CANVAS_DATASET_TYPE",
    "OPENWAM_CANVAS_ENCODE_MODE",
    "OPENWAM_CANVAS_LAYOUT_ID",
    "OPENWAM_CANVAS_POLICY_FACTORY",
    "ROBOT_BASE_EEF_FULL",
    "ROBOT_BASE_EEF_LAYOUTS",
    "ROBOT_BASE_EEF_ONLY",
    "ROPE_ALIGNED",
    "ROPE_INDEPENDENT",
    "ROPE_MODES",
    "RETIRED_SFT_OPTIONS_KEY",
    "SANA_PIXEL_ASPECT_RATIO_TYPE",
    "SFT_OPTIONS_KEY",
    "SANA_PIXEL_CANVAS_DATASET_TYPE",
    "TARGET_MODES",
    "THREE_VIEW_POLICY_FACTORY",
    "VALIDATED_CFG_SCALE",
    "VISUAL_LAYOUTS",
    "VISUAL_LAYOUT_OPENWAM_CANVAS",
    "VISUAL_LAYOUT_SANA_PIXEL_CANVAS",
    "VISUAL_LAYOUT_THREE_VIEW_STRIP",
    "action_mode_from_train_config",
    "declared_multiview",
    "eef_target_mode_from_train_config",
    "is_openwam_canvas_policy_class",
    "joint_target_mode_from_train_config",
    "latent_input_size",
    "legacy_strided_action_origin",
    "load_train_config",
    "policy_config_from_train_config",
    "resolve_canvas_text_contract",
    "resolve_rope_contract",
    "resolve_visual_layout",
    "robot_base_eef_layout_from_train_config",
    "rope_mode_from_train_config",
    "sampling_defaults_from_train_config",
    "sana_pixel_canvas_hw_from_train_config",
    "sft_options_from_train_config",
    "sana_pixel_pad_from_train_config",
    "view_resize_from_train_config",
    "state_as_cross_attention_from_train_config",
    "video_fps_from_train_config",
    "video_frame_stride_from_train_config",
]
