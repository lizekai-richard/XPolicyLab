"""The deploy contract of a sparse-memory checkpoint, read from its training ``config.yaml``.

A sparse-memory recipe (Sana rwm/sparse-memory, ``configs/causal/robot_pretrain/
causal_robodojo_vanilla_pantheon14k_eefabs_f33fps8_sana_pixel_aligned_sparse_memory_{global,strided_m16s16,...}[_cached].yaml``,
the sana_latent twin ``causal_robodojo_vanilla_mixv3_memory_uniform10k_eefabs_f33fps8_sana_latent_aligned_sparse_memory_strided_m8s25``)
is its parent SFT recipe plus the causal blocks: ``model.model`` names the causal class, ``model.extra.chunk_causal_policy``
sets the chunk size and the memory (``memory_frames`` m rows by ``memory_type: global``, uniform over the history, or
``memory_type: strided`` with ``memory_stride``), ``data.type`` is the causal dataset and the memory-latent rung ladder
replaces the SFT tier.
Everything else keeps its SFT meaning, so the shared ``sana_wam_min`` resolvers read it from a *bidirectional view* of
the yaml (the SFT policy class and the one-chunk tier put back), exactly as for the causal line. The data-side keys
(``robot_sft.tail_windows``, ``sparse_memory_latent_store``) shape the training samples only.

What this adapter implements, refused otherwise at start-up with the reason (before any weight is read):

* ``memory_type`` global or strided with ``memory_frames``, parsed as Sana's ``sparse_memory.memory_from_policy``
  (cc18ad78f) parses it, plus ``fixed``, the global memory's name before Sana eef7beeab (the fixed-16 run's configs up to
  step 40k), read as global; a causal checkpoint without a memory is SANA_WAM_CAUSAL's, and Sana's refusals are this
  adapter's (another type, ``memory_frames`` not a positive multiple of 8 -- so also the unbounded strided memory before
  eef7beeab --, a strided memory without a positive ``memory_stride``, a global one with one);
* ``sequence_attention`` causal or absent (``full``, the whole sequence as one chunk, has no mirror here);
* none of the hierarchical memory's keys (``hierarchical_memory`` and its pre-rename ``memory_uniform`` /
  ``memory_split``): those checkpoints trained routed memories the sparse-memory rewrite (Sana 3f6f8daa5) removed, and
  SANA_WAM_HM up to XPolicyLab d57a617 serves them;
* one chunk = 32 rows = one latent frame (``actions_per_chunk`` 32 at 25 fps, VAE stride 8, video frame stride 4; the
  net refuses anything else);
* the one-view ``sana_pixel`` 320 x 512 canvas (10 x 16 = 160 tokens per latent frame, unmasked pad, one prompt group)
  or the three-view ``sana_latent`` strip (Sana a881daf7a: each camera its own causal encode, 256 x 320 at
  ``ASPECT_RATIO_VIDEO_320_ROBOT``, 8 x 10 = 80 tokens per view, 240 per latent frame, the semantic 2x2 RoPE tiles, the
  G = V + 1 token-group prompts); a chunk's tokens are one latent frame's of every view;
* ``rope: aligned``, no AttnRes time conditioning (the prefix's hidden states must not depend on the target's t, the
  deploy's prefix forward runs without it), the causal LTX encoder (``vae.use_causal_encode`` true or absent).

The keys the sparse-memory forms never read (``memory_content``, ``sliding_window_chunks``, ``chunk_observation``,
``obs_in_first_chunk``) are ignored as Sana ignores them under a ``memory_type``.
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
    VISUAL_LAYOUT_THREE_VIEW_STRIP,
    declared_multiview,
    resolve_rope_contract,
    resolve_visual_layout,
    sana_pixel_pad_from_train_config,
    sft_options_from_train_config,
    video_frame_stride_from_train_config,
)
from sana_wam_min.robodojo_io import ROBODOJO_MODEL_FPS_HZ  # noqa: E402
from sana_wam_min.sana_pixel_canvas import sana_pixel_canvas_hw  # noqa: E402

from . import sparse_memory as sm  # noqa: E402

CAUSAL_POLICY_FACTORY = "SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal_5B_P1_D36"
CAUSAL_DATASET_TYPES = ("CausalRoboDojoSFTDataset",)
ACTION_TEMPORAL_COMPRESSION = 8
DEFAULT_DT_BIAS_INIT = -5.0
DEFAULT_A_LOG_INIT = 0.0
# the hierarchical memory's keys (Sana rwm/sparse-memory before 3f6f8daa5; memory_uniform / memory_split before 3e7435754)
HIERARCHICAL_MEMORY_KEYS = ("hierarchical_memory", "memory_uniform", "memory_split")
# the 320 x 512 sana_pixel canvas: 10 x 16 latent tokens per frame at the VAE's spatial stride 32
CANVAS_LATENT_HW = (10, 16)
SERVED_VISUAL_LAYOUTS = (VISUAL_LAYOUT_SANA_PIXEL_CANVAS, VISUAL_LAYOUT_THREE_VIEW_STRIP)

DEPLOY_CACHED = "cached"
DEPLOY_UNCACHED = "uncached"
DEPLOY_FORMS = (DEPLOY_CACHED, DEPLOY_UNCACHED)


@dataclass(frozen=True)
class SMContract:
    """What the sparse-memory deploy needs from the training yaml (all resolved, all validated)."""

    actions_per_chunk: int           # C = 32 rows per chunk
    video_frame_stride: int          # 4: rows between consecutive video frames
    memory_type: str                 # global | strided
    memory_frames: int               # m: memory rows between the sink and the target (n = m / 8 latents)
    memory_stride: int | None        # s (strided): rows between two memory rows
    model_fps: float
    tier_num_frames: int             # C + 1: the one-chunk window of the parent SFT recipe (the prompt's window)
    trained_memory_latents: int      # n = m / 8, whose rung the recipe's ladder (data.multi_fps) lists
    multiview: str
    visual_layout: str
    rope: str
    dt_bias_init: float
    a_log_init: float
    reset_legacy_beta: bool

    @property
    def memory(self) -> dict:
        """The memory in Sana's ``memory_from_policy`` form."""

        if self.memory_type == sm.GLOBAL:
            return {sm.MEMORY_TYPE: sm.GLOBAL, sm.MEMORY_FRAMES: int(self.memory_frames)}
        return {sm.MEMORY_TYPE: sm.STRIDED, sm.MEMORY_FRAMES: int(self.memory_frames), sm.MEMORY_STRIDE: int(self.memory_stride)}

    def resolve_deploy_form(self, value: Any) -> str:
        """``deploy_form`` knob: null -> cached (Sana ``SparseMemoryPolicySession.decide``)."""

        form = DEPLOY_CACHED if value in (None, "") else str(value).strip().lower()
        if form not in DEPLOY_FORMS:
            raise ValueError(f"deploy_form must be one of {DEPLOY_FORMS} or null, got {value!r}")
        return form

    def describe(self) -> str:
        size = f"memory_frames={self.memory_frames}" + (
            f" memory_stride={self.memory_stride}" if self.memory_type == sm.STRIDED else ""
        )
        return (
            f"protocol=sparse_memory memory_type={self.memory_type} {size} "
            f"trained_memory_latents={self.trained_memory_latents} actions_per_chunk={self.actions_per_chunk} "
            f"video_frame_stride={self.video_frame_stride} multiview={self.multiview} rope={self.rope}"
        )


def _extra(container: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if container is None:
        return {}
    extra = container.get("extra") if isinstance(container, Mapping) else None
    return extra or {}


def chunk_causal_options(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    options = _extra(cfg.get("model")).get("chunk_causal_policy")
    if not isinstance(options, Mapping):
        raise ValueError("a sparse-memory checkpoint's config.yaml must carry model.extra.chunk_causal_policy")
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


def is_sparse_memory_config(cfg: Mapping[str, Any]) -> bool:
    try:
        options = chunk_causal_options(cfg)
    except ValueError:
        return False
    return any(key in options for key in sm.MEMORY_KEYS)


def resolve_memory(options: Mapping[str, Any]) -> dict:
    """The checkpoint's sparse memory (``sparse_memory.memory_from_policy``), refusing the designs it would misread."""

    hierarchical = [key for key in HIERARCHICAL_MEMORY_KEYS if key in options]
    if hierarchical:
        raise ValueError(
            f"model.extra.chunk_causal_policy declares {hierarchical}: a hierarchical-memory checkpoint (Sana "
            "rwm/sparse-memory before 3f6f8daa5, routed dense / sparse memories the sparse-memory rewrite removed); it is "
            "served by SANA_WAM_HM up to XPolicyLab d57a617, not by SANA_WAM_SM"
        )
    memory = sm.memory_from_policy(options)
    if memory is None:
        raise ValueError(
            "model.extra.chunk_causal_policy declares no memory_type: a causal checkpoint without a sparse memory (serve "
            "it with SANA_WAM_CAUSAL)"
        )
    if sm.full_sequence_attention(options):
        raise NotImplementedError(
            "chunk_causal_policy.sequence_attention: full runs the whole [S | M | T | robot] sequence as one chunk (Sana "
            "b76502831), which this mirror does not implement; it would be served as the causal sequence"
        )
    return memory


def bidirectional_view(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The yaml with the parent SFT recipe's policy class and one-chunk tier put back (the memory-latent rung ladder
    replaced by the C + 1 window every sample's prompt is built for)."""

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


def trained_memory_latents(cfg: Mapping[str, Any], memory: dict, fps: float) -> int:
    """The memory latent count n = m / 8 every sample trains, whose rung 1 + (n + 1) * 32 the recipe's ladder
    (``data.multi_fps``) must list (Sana's dataset refuses a sample whose rung is not listed)."""

    ladder = (cfg.get("data") or {}).get("multi_fps") or {}
    rungs = ladder.get(str(int(round(float(fps))))) if isinstance(ladder, Mapping) else None
    if not rungs:
        raise ValueError(f"data.multi_fps lists no rung ladder for {fps:g} fps")
    rungs = sorted(int(rung) for rung in rungs)
    if any((rung - 1) % sm.CHUNK_ROWS or rung < 1 + sm.CHUNK_ROWS for rung in rungs):
        raise ValueError(f"data.multi_fps {rungs}: a sparse-memory rung is 1 + (n + 1) * {sm.CHUNK_ROWS}")
    n = sm.memory_latents(memory)
    if sm.sampler_rung(n) not in rungs:
        raise ValueError(
            f"memory_frames {memory[sm.MEMORY_FRAMES]} puts every sample on rung {sm.sampler_rung(n)}, which "
            f"data.multi_fps {rungs} does not list"
        )
    return n


def resolve_sm_contract(cfg: Mapping[str, Any]) -> SMContract:
    """Validate a sparse-memory checkpoint's yaml against what this adapter implements and return its contract."""

    model_cfg = cfg.get("model") or {}
    name = str(model_cfg.get("model"))
    if name != CAUSAL_POLICY_FACTORY:
        raise ValueError(f"model.model {name!r} is not the chunk-causal policy class {CAUSAL_POLICY_FACTORY}")
    data_type = str((cfg.get("data") or {}).get("type") or "")
    if data_type not in CAUSAL_DATASET_TYPES:
        raise ValueError(f"data.type {data_type!r} is not the causal dataset {CAUSAL_DATASET_TYPES}")
    options = chunk_causal_options(cfg)
    memory = resolve_memory(options)
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
    if chunk != sm.CHUNK_ROWS or ACTION_TEMPORAL_COMPRESSION * stride != sm.CHUNK_ROWS:
        raise ValueError(
            f"the sparse-memory sequence ends in one target latent of {sm.CHUNK_ROWS} rows: actions_per_chunk "
            f"{sm.CHUNK_ROWS} and VAE stride {ACTION_TEMPORAL_COMPRESSION} x video frame stride "
            f"{sm.CHUNK_ROWS // ACTION_TEMPORAL_COMPRESSION}; got C={chunk} and stride {stride}"
        )
    trained = trained_memory_latents(cfg, memory, fps)

    multiview = declared_multiview(view)
    layout = resolve_visual_layout(view)
    if layout not in SERVED_VISUAL_LAYOUTS:
        raise NotImplementedError(
            f"the sparse memory serves the one-view sana_pixel canvas and the three-view sana_latent strip; this "
            f"checkpoint's front-end is {layout} ({multiview!r})"
        )
    if layout == VISUAL_LAYOUT_SANA_PIXEL_CANVAS:
        canvas = tuple(int(v) for v in sana_pixel_canvas_hw((view.get("data") or {}).get("aspect_ratio_type")))
        latent_hw = (canvas[0] // 32, canvas[1] // 32)
        if latent_hw != CANVAS_LATENT_HW:
            raise NotImplementedError(
                f"the sparse memory lays {sm.TOKENS_PER_FRAME} tokens per canvas latent frame (the 320 x 512 canvas, "
                f"latent {CANVAS_LATENT_HW}); this checkpoint's canvas is {canvas[0]} x {canvas[1]} (latent {latent_hw})"
            )
        pad = sana_pixel_pad_from_train_config(view)
        if pad != "unmasked":
            raise NotImplementedError(f"model.extra.sana_pixel_pad {pad!r} is not implemented by this adapter")
    rope, _origin, _label = resolve_rope_contract(view, None)
    if rope != ROPE_ALIGNED:
        raise NotImplementedError(f"the sparse-memory mirror implements rope {ROPE_ALIGNED} only; got rope {rope!r}")
    if bool(model_cfg.get("use_time_conditioning", False)):
        raise NotImplementedError(
            "model.use_time_conditioning: true pools the target's t into every token's AttnRes depth query, so the "
            "prefix's hidden states would depend on it and the deploy's prefix forward could not run once per decision"
        )
    if bool(model_cfg.get("use_dual_attn_res_routing", False)):
        raise NotImplementedError("dual AttnRes routing is refused by the causal policy itself")
    vae_cfg = cfg.get("vae") or {}
    if vae_cfg.get("use_causal_encode") is False:
        raise NotImplementedError("the sparse memory trains on the causal LTX encoder; the yaml sets vae.use_causal_encode false")
    return SMContract(
        actions_per_chunk=chunk,
        video_frame_stride=stride,
        memory_type=memory[sm.MEMORY_TYPE],
        memory_frames=memory.get(sm.MEMORY_FRAMES),
        memory_stride=memory.get(sm.MEMORY_STRIDE),
        model_fps=fps,
        tier_num_frames=tier,
        trained_memory_latents=trained,
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
    "DEPLOY_CACHED",
    "DEPLOY_FORMS",
    "DEPLOY_UNCACHED",
    "HIERARCHICAL_MEMORY_KEYS",
    "SERVED_VISUAL_LAYOUTS",
    "SMContract",
    "bidirectional_view",
    "chunk_causal_options",
    "is_sparse_memory_config",
    "resolve_actions_per_chunk",
    "resolve_memory",
    "resolve_sm_contract",
    "trained_memory_latents",
]
