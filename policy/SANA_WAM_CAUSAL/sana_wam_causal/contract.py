"""The deploy contract of a chunk-causal policy checkpoint, read from its training ``config.yaml``.

A causal recipe is its parent SFT recipe plus the causal blocks (Sana ``posttrain_causal_handoff/04_configs.md``
section 5): ``model.model`` names the causal class, ``model.extra.chunk_causal_policy`` sets the chunk size, the
softmax sliding window and what a memory chunk carries, ``data.type`` is the causal chunk-window dataset and the window
ladder replaces the SFT tier. Everything else
(multiview front-end, RoPE mode, action / target modes, normalization pin, noise schedule) keeps its SFT meaning, so the
shared ``sana_wam_min`` resolvers read it from a *bidirectional view* of the yaml: the same dict with the SFT policy
class and the one-chunk tier put back.

This adapter serves the episode protocol (Sana ``d37817788`` / ``fcccb4c88`` / ``3cbe3d85e``, deploy ``4bd1b674c``;
inference handoff 03, first status note) in its single form since Sana ``4bd6eca83`` (rwm/zekai-merge, checked against
``7de7711c2``): every training window is a whole episode from its first frame, the memory opens with the sink (the
episode's first frame, on every canvas), a noisy chunk is its target latents alone and reads its observation from the
memory -- the last frame of the most recent memory latent: the sink for chunk 0, chunk c - 1's last target latent for
chunk c -- and a memory chunk holds its target latents (plus its anchor state and actions under ``memory_content:
video_and_action``, the value of a config without the key). ``model.extra.chunk_causal_policy.chunk_observation`` is
accepted absent or false (Sana ``refuse_chunk_observation``); a checkpoint trained with chunk observations (``true``)
is refused: its protocol needs Sana 404c9bf0e or earlier, served by this adapter up to XPolicyLab ``d57a617``. Like
Sana, the config alone cannot tell a run from before 2026-10-14 that trained chunk observations without declaring the
key (absent meant true then; e.g. the episode VO run). The Hub upload of a causal checkpoint records the run's protocol
next to its ``config.yaml`` (``RUN_ID.json``, copied into ``upload_manifest.json``): ``check_run_record`` refuses a
record of chunk observations and a record that contradicts the config, before the weights are read. Front-ends: the
three-view ``sana_latent`` strip and the one-view ``sana_pixel`` canvas, ``rope: aligned``, no AttnRes time
conditioning, unmasked canvas pad.

``data.extra.aligned_phase_window`` (Sana ``04c8acbee``; ``padded`` when absent) decides how a deployed episode opens. The
phase window whose chunk grid lands on the first frame is ``padded`` (h = C: its chunk 0 the first frame frozen over
one chunk, memory only; the deploy commits it and emits from chunk 1) or ``first_frame`` (h = 0: chunk 0 is the first
frame's own chunk with no memory before it but the sink and is supervised; the deploy emits from chunk 0, which reads
the sink as its observation).

``model.extra.chunk_causal_policy.causal_observation`` (Sana ``764ee2113``, deploy ``adc4867ea``; false when absent) gives
every noisy chunk its own observation latent o_c, its start frame encoded alone (chunk 0's is the sink's latent), clean,
one latent frame after the most recent memory latent: it reads the sink, the softmax window, the GDN state and itself,
never its chunk's targets or robot rows, and is never committed; the chunk's targets (one frame after it) and robot
rows (C / f_c steps later) read it besides the window. ``causal_state`` (Sana ``afc9874b6``, deploy ``8b599e49d``; false
when absent, true only with ``causal_observation``) joins the chunk's state row to o_c's group: the denoise window then
carries the targets and the C action rows alone.

``model.extra.chunk_causal_policy.physical_rope`` (Sana ``6e91eaf2a``; false when absent, ``rope: aligned`` only) puts
every token at its physical instant: a target latent (committed or generated) at the middle of the frames it covers, c *
f_c + i + 0.5625 (``latent_centre_offset``, (8 + 1) / 16 for the VAE's 8), an observation latent at its chunk's start c *
f_c, the state row and action j at their steps c * C and c * C + j (no observation shift), the sink at 0; a softmax read
numbered from its window start (the sink rotated there) then sees a one-chunk window at 0 / 0.5625 / 1 / 1.5625. Sana's
own cached session refuses it (posttrain_causal_handoff 11, R4); this adapter serves it.

``model.memory_gdn_output_normalizer`` (Sana ``c414e9b32``; ``true`` when absent) decides every GDN layer's readout: true
divides the numerator ``q (gS + K^T diag(beta) V)`` by the denominator ``ReLU(q) . (gz + K_den^T beta) + eps``, false
returns the numerator alone (z is still carried, so a commit writes the same state either way). The per-head RMSNorm
after the readout cancels any positive denominator, so the two differ where it is near zero or negative.

The sink-less per-chunk observation protocol (Sana ``61aad2d58`` .. ``3cbe3d85e``'s deploy, the pantheon14k m48n24
checkpoints) is retired here as in Sana; its configs carry ``data.extra.window_repeats``, which no episode recipe does
(the episode launcher refuses it), and are refused with the reason, at start-up, before the weights are read. So are a
config that still names ``obs_in_first_chunk`` (the chunk-0-only observation of the vanilla52k line) and anything else
outside the contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

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

CAUSAL_POLICY_FACTORY = "SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal_5B_P1_D36"
CAUSAL_DATASET_TYPES = ("CausalRoboDojoSFTDataset",)
# VAE temporal stride: one latent frame per 8 source frames (``model.action_temporal_compression``)
ACTION_TEMPORAL_COMPRESSION = 8
# Sana ``ChunkGDNLinearAttention`` defaults for recurrence parameters a checkpoint does not carry (an SFT donor)
DEFAULT_DT_BIAS_INIT = -5.0
DEFAULT_A_LOG_INIT = 0.0
# ``model.extra.chunk_causal_policy.memory_content`` (Sana ``causal_rwm_train.MEMORY_CONTENTS``): what a memory chunk
# carries besides its target latents; an absent key reads as video_and_action (Sana ``resolve_memory_content``)
MEMORY_VIDEO_AND_ACTION = "video_and_action"
MEMORY_VIDEO_ONLY = "video_only"
MEMORY_CONTENTS = (MEMORY_VIDEO_AND_ACTION, MEMORY_VIDEO_ONLY)
# the front-ends this adapter serves: the three-view strip (multiview sana_latent) and the one-view sana_pixel canvas
SERVED_VISUAL_LAYOUTS = (VISUAL_LAYOUT_THREE_VIEW_STRIP, VISUAL_LAYOUT_SANA_PIXEL_CANVAS)
PROTOCOL_EPISODE = "episode"
# data.extra.aligned_phase_window (Sana causal_chunk_window_data.aligned_phase_window): absent = padded
ALIGNED_PHASE_PADDED = "padded"
ALIGNED_PHASE_FIRST_FRAME = "first_frame"
ALIGNED_PHASE_WINDOWS = (ALIGNED_PHASE_PADDED, ALIGNED_PHASE_FIRST_FRAME)
# the run record a Hub upload of the causal line writes next to config.yaml (RUN_ID.json; upload_manifest.json carries
# the same keys): the run's protocol switches, which a config from before Sana 4bd6eca83 need not declare
RUN_RECORD_NAMES = ("RUN_ID.json", "upload_manifest.json")


@dataclass(frozen=True)
class CausalContract:
    """What the causal deploy needs from the training yaml (all resolved, all validated)."""

    actions_per_chunk: int           # C: action rows per chunk (32 for the 33-row tier)
    video_frame_stride: int          # s: source rows between consecutive video frames (4 at video_fps 8 on 33 rows)
    latent_frames_per_chunk: int     # f_c = C / (8 s): target latent frames of one chunk
    sliding_window_chunks: Optional[int]  # W: committed chunks a chunk's softmax layers read (None = all of them)
    memory_content: str              # video_and_action | video_only: what a committed chunk keeps besides its targets
    model_fps: float
    tier_num_frames: int             # C + 1: the one-chunk window of the parent SFT recipe
    multiview: str
    visual_layout: str
    rope: str
    dt_bias_init: float
    a_log_init: float
    reset_legacy_beta: bool
    aligned_phase_window: str = ALIGNED_PHASE_PADDED  # padded: chunk 0 frozen, memory only | first_frame: chunk 0 emitted
    gdn_output_normalizer: bool = True  # false: every GDN layer reads its numerator alone (memory_gdn_output_normalizer)
    causal_observation: bool = False    # every noisy chunk opens with its own observation latent, read causally
    causal_state: bool = False          # the chunk's state row joins its observation latent (causal_observation only)
    physical_rope: bool = False         # every token at its physical instant (target latents at their frames' middle)

    @property
    def first_chunk(self) -> int:
        """The chunk an episode emits first: 1 after the frozen chunk 0 of a padded window, 0 on a first-frame window."""

        return 0 if self.aligned_phase_window == ALIGNED_PHASE_FIRST_FRAME else 1

    def describe(self) -> str:
        window = "all" if self.sliding_window_chunks is None else str(self.sliding_window_chunks)
        return (
            f"protocol={PROTOCOL_EPISODE} actions_per_chunk={self.actions_per_chunk} "
            f"video_frame_stride={self.video_frame_stride} latent_frames_per_chunk={self.latent_frames_per_chunk} "
            f"sliding_window_chunks={window} memory_content={self.memory_content} "
            f"chunk_observation=False causal_observation={self.causal_observation} causal_state={self.causal_state} "
            f"multiview={self.multiview} visual_layout={self.visual_layout} "
            f"rope={self.rope} physical_rope={self.physical_rope} aligned_phase_window={self.aligned_phase_window} "
            f"first_chunk={self.first_chunk} "
            f"gdn_output_normalizer={self.gdn_output_normalizer}"
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


def resolve_memory_content(value: Any) -> str:
    """``memory_content`` as Sana reads it: None (a config older than the key) -> video_and_action."""

    if value is None:
        return MEMORY_VIDEO_AND_ACTION
    if value not in MEMORY_CONTENTS:
        raise ValueError(
            f"model.extra.chunk_causal_policy.memory_content={value!r}: allowed values are {list(MEMORY_CONTENTS)}"
        )
    return str(value)


def resolve_aligned_phase_window(value: Any) -> str:
    """``data.extra.aligned_phase_window`` as Sana reads it (``causal_chunk_window_data.aligned_phase_window``): None (a
    config older than the key) -> padded; padded | first_frame otherwise."""

    value = ALIGNED_PHASE_PADDED if value is None else value
    if value not in ALIGNED_PHASE_WINDOWS:
        raise ValueError(f"data.extra.aligned_phase_window must be one of {list(ALIGNED_PHASE_WINDOWS)}; got {value!r}")
    return str(value)


def resolve_gdn_output_normalizer(value: Any) -> bool:
    """``model.memory_gdn_output_normalizer`` as Sana reads it (``sana_qwennext_policy_causal``, c414e9b32): None (a
    config older than the key) -> true; a yaml boolean otherwise (Sana's ``bool()`` would read the string "false" as
    true, so anything but a boolean is refused)."""

    if value is None:
        return True
    if not isinstance(value, bool):
        raise ValueError(f"model.memory_gdn_output_normalizer must be true or false, got {value!r}")
    return value


def resolve_causal_observation(value: Any) -> bool:
    """``causal_observation`` as Sana reads it (``causal_rwm_train.resolve_causal_observation``, 764ee2113): None ->
    false, a yaml boolean otherwise."""

    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"model.extra.chunk_causal_policy.causal_observation={value!r}: must be true or false")
    return value


def resolve_causal_state(value: Any, observation: bool) -> bool:
    """``causal_state`` as Sana reads it (``causal_rwm_train.resolve_causal_state``, afc9874b6): None -> false, a yaml
    boolean otherwise, true only with ``causal_observation`` (the state row joins the observation latent)."""

    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"model.extra.chunk_causal_policy.causal_state={value!r}: must be true or false")
    if value and not observation:
        raise ValueError(
            "model.extra.chunk_causal_policy.causal_state joins every noisy chunk's state row to its observation latent; "
            "set causal_observation: true"
        )
    return value


def resolve_physical_rope(value: Any) -> bool:
    """``physical_rope`` as Sana reads it (``causal_rwm_train.resolve_physical_rope``, 6e91eaf2a): None -> false, a yaml
    boolean otherwise (it needs ``rope: aligned``, the only RoPE mode this adapter serves)."""

    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"model.extra.chunk_causal_policy.physical_rope={value!r}: must be true or false")
    return value


def refuse_chunk_observation(value: Any) -> None:
    """``chunk_observation`` as Sana reads it since 4bd6eca83 (``causal_rwm_train.refuse_chunk_observation``): chunk
    observations are retired, every noisy chunk reads its observation from the memory; absent or ``false`` is accepted
    and changes nothing, anything else is refused."""

    if value is not None and value is not False:
        raise ValueError(
            f"model.extra.chunk_causal_policy.chunk_observation={value!r}: chunk observations are retired (Sana "
            "4bd6eca83), every noisy chunk reads its observation from the memory. A checkpoint trained with chunk "
            "observations needs Sana 404c9bf0e or earlier; it is served by SANA_WAM_CAUSAL up to XPolicyLab d57a617"
        )


def load_run_records(directory: str | Path) -> list[tuple[Path, Mapping[str, Any]]]:
    """The run records (``RUN_RECORD_NAMES``) in ``directory``, the one holding the checkpoint's ``config.yaml``; an
    unreadable record is refused rather than skipped."""

    records = []
    for name in RUN_RECORD_NAMES:
        path = Path(directory) / name
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read the run record {path}: {error}") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"the run record {path} is not a JSON object")
        records.append((path, record))
    return records


def check_run_record(record: Mapping[str, Any], contract: CausalContract, source: str = "the run record") -> None:
    """Refuse a checkpoint whose run record contradicts the contract its config resolves to: a run that trained chunk
    observations (``chunk_observation`` anything but false: its config may lack the key, which the single form reads
    as false) and a recorded ``aligned_phase_window`` / ``memory_content`` / ``memory_gdn_output_normalizer`` /
    ``causal_observation`` / ``causal_state`` / ``physical_rope`` / ``sliding_window_chunks`` other than the config's, at
    the record's top level or in its ``deploy`` block (the record of the physical-RoPE line). Keys the record does not
    carry are not checked."""

    observation = record.get("chunk_observation")
    if observation is not None and observation is not False:
        raise ValueError(
            f"{source} records chunk_observation={observation!r} (source_commit {record.get('source_commit')!r}): the "
            "run trained chunk observations, which a config from before Sana 4bd6eca83 need not declare and this adapter "
            "would serve as the single form without them; it is served by SANA_WAM_CAUSAL up to XPolicyLab d57a617"
        )
    deploy = record.get("deploy") if isinstance(record.get("deploy"), Mapping) else {}
    for key, resolved in (
        ("aligned_phase_window", contract.aligned_phase_window),
        ("memory_content", contract.memory_content),
        ("memory_gdn_output_normalizer", contract.gdn_output_normalizer),
        ("causal_observation", contract.causal_observation),
        ("causal_state", contract.causal_state),
        ("physical_rope", contract.physical_rope),
        ("sliding_window_chunks", contract.sliding_window_chunks),
    ):
        for where, block in (("", record), ("deploy.", deploy)):
            recorded = block.get(key)
            if recorded is not None and recorded != resolved:
                raise ValueError(
                    f"{source} records {where}{key}={recorded!r} but config.yaml resolves to {resolved!r}: the record "
                    "and the config describe different runs"
                )


def describe_run_records(records: list[tuple[Path, Mapping[str, Any]]], config_path: str | Path) -> str:
    """One line for the ready report: the records read, the protocol they declare and whether the config is the one
    they recorded (``config_sha256``)."""

    if not records:
        return "none"
    config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    names = "+".join(path.name for path, _ in records)
    record = records[0][1]
    shas = {str(r["config_sha256"]) for _, r in records if r.get("config_sha256")}
    match = "unrecorded" if not shas else ("match" if shas == {config_sha} else "differs")
    commit = str(record.get("source_commit") or "")[:9] or None
    return (
        f"{names}(chunk_observation={record.get('chunk_observation')} "
        f"aligned_phase_window={record.get('aligned_phase_window')} memory_content={record.get('memory_content')} "
        f"step={record.get('step')} source_commit={commit} config_sha256={match})"
    )


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
    # the memory designs of Sana rwm/sparse-memory, which the causal net ignores (it would mis-serve them): the sparse
    # memory (memory_type strided | fixed, since 3f6f8daa5) and the hierarchical memory it replaced (hierarchical_memory;
    # memory_uniform / memory_split before 3e7435754 renamed them)
    sparse_keys = [key for key in ("memory_type", "memory_stride", "memory_frames") if key in options]
    if sparse_keys:
        raise ValueError(
            f"model.extra.chunk_causal_policy.{sparse_keys[0]}: this is a sparse-memory checkpoint (Sana "
            "rwm/sparse-memory: one strided / fixed RGB memory every block reads), which this adapter would mis-serve as "
            "the episode protocol; serve it with SANA_WAM_SM"
        )
    memory_keys = [key for key in ("hierarchical_memory", "memory_uniform", "memory_split") if key in options]
    if memory_keys:
        raise ValueError(
            f"model.extra.chunk_causal_policy.{memory_keys[0]}: this is a hierarchical-memory checkpoint (Sana "
            "rwm/sparse-memory before 3f6f8daa5, a dense / sparse memory read per block kind), which this adapter would "
            "mis-serve as the episode protocol; it is served by SANA_WAM_HM up to XPolicyLab d57a617 (SANA_WAM_SM serves "
            "the sparse memory that replaced it)"
        )
    if "window_repeats" in _extra(cfg.get("data")):
        raise ValueError(
            "data.extra.window_repeats: this checkpoint trained the sink-less per-chunk observation protocol (the "
            "pantheon14k m48n24 recipes; Sana before d37817788), which this adapter no longer serves (the episode "
            "protocol: the first-observation sink, the frozen first chunk, one causal VAE stream per view). Serve it "
            "from the archived adapter (XPolicyLab snapshot SANA_WAM_CAUSAL_per_chunk_obs_pre_episode_20261013); an "
            "episode recipe never sets window_repeats (its launcher refuses it)"
        )
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

    if "obs_in_first_chunk" in options:
        raise ValueError(
            "model.extra.chunk_causal_policy.obs_in_first_chunk="
            f"{options['obs_in_first_chunk']!r}: this checkpoint trained a retired observation layout (the chunk-0-only "
            "observation of the vanilla52k line, Sana before 61aad2d58); this adapter serves the per-chunk observation "
            "protocol only (the earlier protocol is archived in XPolicyLab commit 69cfb8e)"
        )
    memory_content = resolve_memory_content(options.get("memory_content"))
    refuse_chunk_observation(options.get("chunk_observation"))
    causal_observation = resolve_causal_observation(options.get("causal_observation"))
    causal_state = resolve_causal_state(options.get("causal_state"), causal_observation)
    physical_rope = resolve_physical_rope(options.get("physical_rope"))
    aligned_phase_window = resolve_aligned_phase_window(_extra(cfg.get("data")).get("aligned_phase_window"))
    window = options.get("sliding_window_chunks")
    window = None if window is None else int(window)
    if window is not None and window < 1:
        raise ValueError(f"sliding_window_chunks must be >= 1 or null, got {window}")

    multiview = declared_multiview(view)
    layout = resolve_visual_layout(view)
    if layout not in SERVED_VISUAL_LAYOUTS:
        raise NotImplementedError(
            f"the causal adapter serves the three-view sana_latent strip and the one-view sana_pixel canvas; this "
            f"checkpoint's front-end is {layout} (multiview {multiview!r})"
        )
    pad = sana_pixel_pad_from_train_config(view) if layout == VISUAL_LAYOUT_SANA_PIXEL_CANVAS else "unmasked"
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
        memory_content=memory_content,
        model_fps=fps,
        tier_num_frames=tier,
        multiview=str(multiview),
        visual_layout=layout,
        rope=rope,
        dt_bias_init=float(options.get("dt_bias_init", DEFAULT_DT_BIAS_INIT)),
        a_log_init=float(options.get("a_log_init", DEFAULT_A_LOG_INIT)),
        reset_legacy_beta=bool(options.get("reset_legacy_beta", False)),
        aligned_phase_window=aligned_phase_window,
        gdn_output_normalizer=resolve_gdn_output_normalizer(model_cfg.get("memory_gdn_output_normalizer")),
        causal_observation=causal_observation,
        causal_state=causal_state,
        physical_rope=physical_rope,
    )


__all__ = [
    "ACTION_TEMPORAL_COMPRESSION",
    "ALIGNED_PHASE_FIRST_FRAME",
    "ALIGNED_PHASE_PADDED",
    "ALIGNED_PHASE_WINDOWS",
    "CAUSAL_DATASET_TYPES",
    "CAUSAL_POLICY_FACTORY",
    "CausalContract",
    "MEMORY_CONTENTS",
    "MEMORY_VIDEO_AND_ACTION",
    "MEMORY_VIDEO_ONLY",
    "PROTOCOL_EPISODE",
    "RUN_RECORD_NAMES",
    "SERVED_VISUAL_LAYOUTS",
    "bidirectional_view",
    "check_run_record",
    "chunk_causal_options",
    "describe_run_records",
    "is_causal_policy_config",
    "load_run_records",
    "resolve_actions_per_chunk",
    "resolve_aligned_phase_window",
    "refuse_chunk_observation",
    "resolve_causal_contract",
    "resolve_causal_observation",
    "resolve_causal_state",
    "resolve_gdn_output_normalizer",
    "resolve_physical_rope",
    "resolve_memory_content",
]
