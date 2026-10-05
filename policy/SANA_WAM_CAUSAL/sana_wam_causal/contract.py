"""The deploy contract of a chunk-causal policy checkpoint, read from its training ``config.yaml``.

A causal recipe is its parent SFT recipe plus the causal blocks (Sana ``posttrain_causal_handoff/04_configs.md``
section 5): ``model.model`` names the causal class, ``model.extra.chunk_causal_policy`` sets the chunk size, the
softmax sliding window and the observation layout, ``data.type`` is the causal chunk-window dataset and the window
ladder replaces the SFT tier. Everything else (multiview front-end, RoPE mode, action / target modes, normalization
pin, noise schedule) keeps its SFT meaning, so the shared ``sana_wam_min`` resolvers read it from a *bidirectional
view* of the yaml: the same dict with the SFT policy class and the one-chunk tier put back.

This adapter serves the contract the RoboDojo causal line trains (the m48n24 recipes): one-view canvas
(``sana_pixel``), ``rope: aligned``, ``obs_in_first_chunk: true``, no AttnRes time conditioning, unmasked canvas pad.
Anything else is refused at start-up with the reason, before the weights are read.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.config import (  # noqa: E402
    ROPE_ALIGNED,
    THREE_VIEW_POLICY_FACTORY,
    VISUAL_LAYOUT_SANA_PIXEL_CANVAS,
    declared_multiview,
    resolve_rope_contract,
    resolve_visual_layout,
    sana_pixel_pad_from_train_config,
    sft_options_from_train_config,
    video_frame_stride_from_train_config,
)
from sana_wam_min.robodojo_io import ROBODOJO_MODEL_FPS_HZ  # noqa: E402

CAUSAL_POLICY_FACTORY = "SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal_5B_P1_D36"
CAUSAL_DATASET_TYPES = ("CausalRoboDojoSFTDataset",)
# VAE temporal stride: one latent frame per 8 source frames (``model.action_temporal_compression``)
ACTION_TEMPORAL_COMPRESSION = 8
# Sana ``ChunkGDNLinearAttention`` defaults for recurrence parameters a checkpoint does not carry (an SFT donor)
DEFAULT_DT_BIAS_INIT = -5.0
DEFAULT_A_LOG_INIT = 0.0


@dataclass(frozen=True)
class CausalContract:
    """What the causal deploy needs from the training yaml (all resolved, all validated)."""

    actions_per_chunk: int           # C: action rows per chunk (32 for the 33-row tier)
    video_frame_stride: int          # s: source rows between consecutive video frames (4 at video_fps 8 on 33 rows)
    latent_frames_per_chunk: int     # f_c = C / (8 s): target latent frames of one chunk
    sliding_window_chunks: Optional[int]  # N: softmax context entries a chunk reads (None = every committed entry)
    obs_in_first_chunk: bool
    model_fps: float
    tier_num_frames: int             # C + 1: the one-chunk window of the parent SFT recipe
    multiview: str
    visual_layout: str
    rope: str
    dt_bias_init: float
    a_log_init: float
    reset_legacy_beta: bool

    def describe(self) -> str:
        window = "all" if self.sliding_window_chunks is None else str(self.sliding_window_chunks)
        return (
            f"actions_per_chunk={self.actions_per_chunk} video_frame_stride={self.video_frame_stride} "
            f"latent_frames_per_chunk={self.latent_frames_per_chunk} sliding_window_chunks={window} "
            f"obs_in_first_chunk={self.obs_in_first_chunk} multiview={self.multiview} rope={self.rope}"
        )


def _extra(container: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if container is None:
        return {}
    extra = container.get("extra") if isinstance(container, Mapping) else None
    return extra or {}


def chunk_causal_options(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """``model.extra.chunk_causal_policy`` (required for a causal checkpoint)."""

    options = _extra(cfg.get("model")).get("chunk_causal_policy")
    if not isinstance(options, Mapping):
        raise ValueError("a causal checkpoint's config.yaml must carry model.extra.chunk_causal_policy")
    return options


def resolve_actions_per_chunk(spec: Any, fps: float, vae_stride: int = ACTION_TEMPORAL_COMPRESSION) -> int:
    """C for one fps tier: one int, or a map from the integer fps written as a string to C (Sana
    ``causal_rwm_train.resolve_actions_per_chunk``); C must be a positive multiple of the VAE stride."""

    if isinstance(spec, Mapping):
        key = str(int(round(float(fps))))
        if key not in spec:
            raise KeyError(f"actions_per_chunk has no entry for fps={key}; configured tiers: {sorted(spec)}")
        chunk = int(spec[key])
    else:
        chunk = int(spec)
    if chunk <= 0 or chunk % vae_stride != 0:
        raise ValueError(f"actions_per_chunk={chunk} must be a positive multiple of the VAE temporal stride {vae_stride}")
    return chunk


def is_causal_policy_config(cfg: Mapping[str, Any]) -> bool:
    return str((cfg.get("model") or {}).get("model")) == CAUSAL_POLICY_FACTORY


def bidirectional_view(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The yaml with the parent SFT recipe's policy class and one-chunk tier put back.

    Only the two keys the causal derivation changes for the policy class and the frame tier are reverted; the shared
    resolvers of ``sana_wam_min`` then read every SFT-meaning key (multiview, rope, action modes, normalization pin,
    schedule, text contract) exactly as for the parent SFT checkpoint.
    """

    view = copy.deepcopy(dict(cfg))
    view["model"] = dict(view["model"])
    view["model"]["model"] = THREE_VIEW_POLICY_FACTORY
    data = dict(view["data"])
    tier = int(sft_options_from_train_config(view).get("tier_num_frames") or 0)
    if tier <= 1:
        raise ValueError("the causal recipe's data.extra.robot_sft.tier_num_frames (C + 1) is missing")
    data["num_frames"] = tier
    data["multi_fps"] = {str(int(ROBODOJO_MODEL_FPS_HZ)): [tier]}
    view["data"] = data
    return view


def resolve_causal_contract(cfg: Mapping[str, Any]) -> CausalContract:
    """Validate a causal checkpoint's yaml against what this adapter implements and return its contract."""

    model_cfg = cfg.get("model") or {}
    name = str(model_cfg.get("model"))
    if name != CAUSAL_POLICY_FACTORY:
        raise ValueError(f"model.model {name!r} is not the chunk-causal policy class {CAUSAL_POLICY_FACTORY}")
    data_type = str((cfg.get("data") or {}).get("type") or "")
    if data_type not in CAUSAL_DATASET_TYPES:
        raise ValueError(f"data.type {data_type!r} is not a causal chunk-window dataset {CAUSAL_DATASET_TYPES}")
    options = chunk_causal_options(cfg)
    view = bidirectional_view(cfg)

    fps = float(ROBODOJO_MODEL_FPS_HZ)
    chunk = resolve_actions_per_chunk(options.get("actions_per_chunk", 32), fps)
    data_chunk = _extra(cfg.get("data")).get("actions_per_chunk")
    if data_chunk is not None and resolve_actions_per_chunk(data_chunk, fps) != chunk:
        raise ValueError(
            f"data.extra.actions_per_chunk {data_chunk!r} and model.extra.chunk_causal_policy.actions_per_chunk "
            f"{options.get('actions_per_chunk')!r} disagree at {fps:g} fps"
        )
    tier = int(view["data"]["num_frames"])
    if tier != chunk + 1:
        raise ValueError(f"robot_sft.tier_num_frames {tier} must be actions_per_chunk + 1 = {chunk + 1}")
    stride = int(video_frame_stride_from_train_config(view))
    if chunk % (ACTION_TEMPORAL_COMPRESSION * stride):
        raise ValueError(
            f"actions_per_chunk={chunk} is not a multiple of {ACTION_TEMPORAL_COMPRESSION * stride} rows per latent frame "
            f"(VAE stride {ACTION_TEMPORAL_COMPRESSION} x video frame stride {stride})"
        )
    f_chunk = chunk // (ACTION_TEMPORAL_COMPRESSION * stride)

    obs_first = bool(options.get("obs_in_first_chunk", False))
    if not obs_first:
        raise ValueError(
            "model.extra.chunk_causal_policy.obs_in_first_chunk must be true: the causal trainer requires it since "
            "2026-09-26, and the observation-segment layout of noisy chunk 0 is not implemented here"
        )
    window = options.get("sliding_window_chunks")
    window = None if window is None else int(window)
    if window is not None and window < 1:
        raise ValueError(f"sliding_window_chunks must be >= 1 or null, got {window}")

    multiview = declared_multiview(view)
    layout = resolve_visual_layout(view)
    if layout != VISUAL_LAYOUT_SANA_PIXEL_CANVAS:
        raise NotImplementedError(
            f"the causal adapter serves the one-view sana_pixel canvas; this checkpoint's front-end is {layout} "
            f"(multiview {multiview!r})"
        )
    pad = sana_pixel_pad_from_train_config(view)
    if pad != "unmasked":
        raise NotImplementedError(f"model.extra.sana_pixel_pad {pad!r} is not implemented by the causal adapter")
    rope, _origin, _label = resolve_rope_contract(view, None)
    if rope != ROPE_ALIGNED:
        raise NotImplementedError(f"the causal adapter implements rope: aligned only; this checkpoint resolves to {rope!r}")
    if bool(model_cfg.get("use_time_conditioning", False)):
        raise NotImplementedError(
            "model.use_time_conditioning: true feeds the cached past tokens into the AttnRes time pool, which this "
            "adapter does not model"
        )
    if bool(model_cfg.get("use_dual_attn_res_routing", False)):
        raise NotImplementedError("dual AttnRes routing is refused by the causal policy itself")
    return CausalContract(
        actions_per_chunk=chunk,
        video_frame_stride=stride,
        latent_frames_per_chunk=f_chunk,
        sliding_window_chunks=window,
        obs_in_first_chunk=obs_first,
        model_fps=fps,
        tier_num_frames=tier,
        multiview=str(multiview),
        visual_layout=layout,
        rope=rope,
        dt_bias_init=float(options.get("dt_bias_init", DEFAULT_DT_BIAS_INIT)),
        a_log_init=float(options.get("a_log_init", DEFAULT_A_LOG_INIT)),
        reset_legacy_beta=bool(options.get("reset_legacy_beta", False)),
    )


__all__ = [
    "ACTION_TEMPORAL_COMPRESSION",
    "CAUSAL_DATASET_TYPES",
    "CAUSAL_POLICY_FACTORY",
    "CausalContract",
    "bidirectional_view",
    "chunk_causal_options",
    "is_causal_policy_config",
    "resolve_actions_per_chunk",
    "resolve_causal_contract",
]
