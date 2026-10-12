"""The deploy contract of a hierarchical-memory checkpoint, read from its training ``config.yaml``.

A hierarchical-memory recipe (Sana rwm/sparse-memory, ``configs/causal/robot_pretrain/
causal_robodojo_pantheon14k_eefabs_f33fps8_sana_pixel_aligned_hm_{dense_sparse,sparse_sparse}_videoonly.yaml``) is
its parent SFT recipe plus the causal blocks: ``model.model`` names the causal class, ``model.extra.chunk_causal_policy``
sets the chunk size, ``memory_content: video_only`` and the ``hierarchical_memory`` block (which memory each block kind
reads), ``data.type`` is the causal dataset and the frame-anchor rung ladder replaces the SFT tier. Everything else keeps
its SFT meaning, so the shared ``sana_wam_min`` resolvers read it from a *bidirectional view* of the yaml (the SFT policy
class and the one-chunk tier put back), exactly as for the causal line.

What this adapter implements, refused otherwise at start-up with the reason (before any weight is read):

* ``hierarchical_memory`` present (a causal checkpoint without it is the per-chunk observation protocol of
  SANA_WAM_CAUSAL), ``memory_content: video_only`` (the memory latents carry no robot rows), no
  ``sliding_window_chunks`` (a softmax block reads its whole memory), no ``obs_in_first_chunk`` (a retired layout);
* one chunk = 32 rows = one latent frame (``actions_per_chunk`` 32 at 25 fps, VAE stride 8, video frame stride 4), the
  one-view ``sana_pixel`` 320 x 512 canvas (10 x 16 = 160 tokens per latent frame), ``rope: aligned``, unmasked canvas
  pad, no AttnRes time conditioning (the memory latents' hidden states must not depend on the noisy chunk's t), the
  causal LTX encoder (``vae.use_causal_encode`` true or absent).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

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
from sana_wam_min.sana_pixel_canvas import sana_pixel_canvas_hw  # noqa: E402

from . import memory as hm  # noqa: E402

CAUSAL_POLICY_FACTORY = "SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal_5B_P1_D36"
CAUSAL_DATASET_TYPES = ("CausalRoboDojoSFTDataset",)
ACTION_TEMPORAL_COMPRESSION = 8
DEFAULT_DT_BIAS_INIT = -5.0
DEFAULT_A_LOG_INIT = 0.0
MEMORY_VIDEO_ONLY = "video_only"
# the 320 x 512 sana_pixel canvas: 10 x 16 latent tokens per frame at the VAE's spatial stride 32
CANVAS_LATENT_HW = (10, 16)

DEPLOY_STREAMING = "streaming"
DEPLOY_FULL = "full"
DEPLOY_FORMS = (DEPLOY_STREAMING, DEPLOY_FULL)


@dataclass(frozen=True)
class HMContract:
    """What the hierarchical-memory deploy needs from the training yaml (all resolved, all validated)."""

    actions_per_chunk: int           # C = 32 rows per chunk
    video_frame_stride: int          # 4: rows between consecutive video frames
    gdn_memory: str                  # dense | sparse: the memory the GDN blocks read
    softmax_memory: str              # dense | sparse: the memory the softmax blocks read
    sparse_frames: int               # m: rows sampled uniformly over the history (16 in every recipe)
    model_fps: float
    tier_num_frames: int             # C + 1: the one-chunk window of the parent SFT recipe (the prompt's window)
    multiview: str
    visual_layout: str
    rope: str
    dt_bias_init: float
    a_log_init: float
    reset_legacy_beta: bool

    @property
    def memory(self) -> dict:
        """The parsed ``hierarchical_memory`` block (``memory.parse_memory`` form)."""

        return {hm.GDN: self.gdn_memory, hm.SOFTMAX: self.softmax_memory, "sparse_frames": self.sparse_frames}

    @property
    def streams(self) -> tuple:
        return hm.memory_streams(self.memory)

    def default_deploy_form(self) -> str:
        """Streaming (every memory latent committed once, the noisy chunk alone per denoising step) serves every memory
        choice; ``full`` (the training forward per step) is the reference."""

        return DEPLOY_STREAMING

    def resolve_deploy_form(self, value: Any) -> str:
        """``deploy_form`` knob: null -> streaming."""

        form = self.default_deploy_form() if value in (None, "") else str(value).strip().lower()
        if form not in DEPLOY_FORMS:
            raise ValueError(f"deploy_form must be one of {DEPLOY_FORMS} or null, got {value!r}")
        return form

    def describe(self) -> str:
        return (
            f"protocol=hierarchical_memory gdn={self.gdn_memory} softmax={self.softmax_memory} "
            f"sparse_frames={self.sparse_frames} actions_per_chunk={self.actions_per_chunk} "
            f"video_frame_stride={self.video_frame_stride} memory_content={MEMORY_VIDEO_ONLY} multiview={self.multiview} "
            f"rope={self.rope}"
        )


def _extra(container: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if container is None:
        return {}
    extra = container.get("extra") if isinstance(container, Mapping) else None
    return extra or {}


def chunk_causal_options(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    options = _extra(cfg.get("model")).get("chunk_causal_policy")
    if not isinstance(options, Mapping):
        raise ValueError("a hierarchical-memory checkpoint's config.yaml must carry model.extra.chunk_causal_policy")
    return options


def resolve_actions_per_chunk(spec: Any, fps: float, vae_stride: int = ACTION_TEMPORAL_COMPRESSION) -> int:
    """C for one fps tier: one int, or a map from the integer fps written as a string to C (Sana
    ``causal_rwm_train.resolve_actions_per_chunk``)."""

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


def is_hierarchical_memory_config(cfg: Mapping[str, Any]) -> bool:
    try:
        return hm.MEMORY_KEY in chunk_causal_options(cfg)
    except ValueError:
        return False


def bidirectional_view(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The yaml with the parent SFT recipe's policy class and one-chunk tier put back (the frame-anchor rung ladder
    and the training length 1 + 48 * 32 of a dense recipe replaced by the C + 1 window every sample's prompt is built
    for)."""

    view = copy.deepcopy(dict(cfg))
    view["model"] = dict(view["model"])
    view["model"]["model"] = THREE_VIEW_POLICY_FACTORY
    data = dict(view["data"])
    tier = int(sft_options_from_train_config(view).get("tier_num_frames") or 0)
    if tier <= 1:
        raise ValueError("the recipe's data.extra.robot_sft.tier_num_frames (C + 1) is missing")
    data["num_frames"] = tier
    data["multi_fps"] = {str(int(ROBODOJO_MODEL_FPS_HZ)): [tier]}
    view["data"] = data
    return view


def resolve_hm_contract(cfg: Mapping[str, Any]) -> HMContract:
    """Validate a hierarchical-memory checkpoint's yaml against what this adapter implements and return its contract."""

    model_cfg = cfg.get("model") or {}
    name = str(model_cfg.get("model"))
    if name != CAUSAL_POLICY_FACTORY:
        raise ValueError(f"model.model {name!r} is not the chunk-causal policy class {CAUSAL_POLICY_FACTORY}")
    data_type = str((cfg.get("data") or {}).get("type") or "")
    if data_type not in CAUSAL_DATASET_TYPES:
        raise ValueError(f"data.type {data_type!r} is not the causal dataset {CAUSAL_DATASET_TYPES}")
    options = chunk_causal_options(cfg)
    if hm.MEMORY_KEY not in options:
        raise ValueError(
            f"model.extra.chunk_causal_policy carries no {hm.MEMORY_KEY} block: this is a causal checkpoint of the per-chunk "
            "observation protocol (serve it with SANA_WAM_CAUSAL), not a hierarchical-memory one"
        )
    memory = hm.parse_memory(options[hm.MEMORY_KEY])
    memory_content = options.get("memory_content")
    if memory_content != MEMORY_VIDEO_ONLY:
        raise ValueError(
            f"{hm.MEMORY_KEY} serves memory_content {MEMORY_VIDEO_ONLY} only (Sana refuses any other value, an absent key "
            f"included); got {memory_content!r}"
        )
    if options.get("sliding_window_chunks") is not None:
        raise ValueError(f"{hm.MEMORY_KEY} has no softmax window; the yaml declares sliding_window_chunks")
    if "obs_in_first_chunk" in options:
        raise ValueError("model.extra.chunk_causal_policy.obs_in_first_chunk belongs to a retired layout")
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
    if chunk != hm.CHUNK_ROWS or ACTION_TEMPORAL_COMPRESSION * stride != hm.CHUNK_ROWS:
        raise ValueError(
            f"{hm.MEMORY_KEY} lays one {hm.CHUNK_ROWS}-row chunk per latent frame: actions_per_chunk {hm.CHUNK_ROWS} and VAE "
            f"stride {ACTION_TEMPORAL_COMPRESSION} x video frame stride {hm.CHUNK_ROWS // ACTION_TEMPORAL_COMPRESSION}; got "
            f"C={chunk} and stride {stride}"
        )

    multiview = declared_multiview(view)
    layout = resolve_visual_layout(view)
    if layout != VISUAL_LAYOUT_SANA_PIXEL_CANVAS:
        raise NotImplementedError(
            f"{hm.MEMORY_KEY} serves the one-view sana_pixel canvas; this checkpoint's front-end is {layout} ({multiview!r})"
        )
    canvas = tuple(int(v) for v in sana_pixel_canvas_hw((view.get("data") or {}).get("aspect_ratio_type")))
    latent_hw = (canvas[0] // 32, canvas[1] // 32)
    if latent_hw != CANVAS_LATENT_HW:
        raise NotImplementedError(
            f"{hm.MEMORY_KEY} lays {hm.LATENT_TOKENS} tokens per latent frame (the 320 x 512 canvas, latent "
            f"{CANVAS_LATENT_HW}); this checkpoint's canvas is {canvas[0]} x {canvas[1]} (latent {latent_hw})"
        )
    pad = sana_pixel_pad_from_train_config(view)
    if pad != "unmasked":
        raise NotImplementedError(f"model.extra.sana_pixel_pad {pad!r} is not implemented by this adapter")
    rope, _origin, _label = resolve_rope_contract(view, None)
    if rope != ROPE_ALIGNED:
        raise NotImplementedError(f"{hm.MEMORY_KEY} places every latent on the aligned clock; got rope {rope!r}")
    if bool(model_cfg.get("use_time_conditioning", False)):
        raise NotImplementedError(
            "model.use_time_conditioning: true pools the noisy chunk's t into every token's AttnRes depth query, so the "
            "memory latents' hidden states would depend on it (Sana's hierarchical-memory plan requires it off)"
        )
    if bool(model_cfg.get("use_dual_attn_res_routing", False)):
        raise NotImplementedError("dual AttnRes routing is refused by the causal policy itself")
    vae_cfg = cfg.get("vae") or {}
    if vae_cfg.get("use_causal_encode") is False:
        raise NotImplementedError(f"{hm.MEMORY_KEY} trains on the causal LTX encoder; the yaml sets vae.use_causal_encode false")
    return HMContract(
        actions_per_chunk=chunk,
        video_frame_stride=stride,
        gdn_memory=memory[hm.GDN],
        softmax_memory=memory[hm.SOFTMAX],
        sparse_frames=memory["sparse_frames"],
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
    "CANVAS_LATENT_HW",
    "CAUSAL_DATASET_TYPES",
    "CAUSAL_POLICY_FACTORY",
    "DEPLOY_FORMS",
    "DEPLOY_FULL",
    "DEPLOY_STREAMING",
    "HMContract",
    "MEMORY_VIDEO_ONLY",
    "bidirectional_view",
    "chunk_causal_options",
    "is_hierarchical_memory_config",
    "resolve_actions_per_chunk",
    "resolve_hm_contract",
]
