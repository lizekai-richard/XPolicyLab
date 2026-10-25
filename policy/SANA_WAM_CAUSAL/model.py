"""XPolicyLab adapter of the SANA chunk-causal policy (RoboDojo ARX-X5, EEF-only absolute targets, EE-pose actions).

The policy runs in-process through the vendored, Sana-free ``sana_wam_causal`` package (on top of SANA_WAM's
``sana_wam_min`` runtime) and is deployed teacher-forced, exactly as it trains (two-stream teacher forcing, the episode
protocol of Sana ``d37817788`` / ``4bd1b674c`` in its single form since ``4bd6eca83``: a chunk carries no observation
latent and reads its observation from the memory, the last frame of the most recent latent in it):

* the first ``get_action`` of an episode opens one causal VAE stream per view at the first frame (the sink: the first
  frame encoded alone) and the memory as the training's phase window aligned on the first frame
  (``data.extra.aligned_phase_window``). ``padded``: the streams encode the first frame repeated over one chunk's
  frames after the sink, and two commits follow, the sink and that frozen first chunk (under ``memory_content``
  video_and_action its rows hold the robot at its first state); chunk 1 is generated at the first frame.
  ``first_frame`` (Sana ``04c8acbee`` / deploy ``9fca24879``): the sink alone is committed and chunk 0 is generated
  at the first frame, reading the sink as its observation;
* every ``get_action`` denoises its chunk as the window [targets] + [state | actions] against the memory: the GDN state
  of everything committed, and on the softmax side the sink plus the newest ``sliding_window_chunks`` committed chunks
  (the training forward's rule);
* while a chunk executes, every ``update_obs`` advances the chunk clock tau (RoboDojo calls it once after every
  executed action, the last one right before the next ``get_action``) and keeps the camera frames of ticks s, 2s, .., C
  (s = the video frame stride, C = the chunk);
* every later ``get_action`` first COMMITS the executed chunk -- its C / s observed frames encoded on in the per-view
  streams (the last latent ends on the new chunk's start frame, the observation it reads) and, under video_and_action,
  the committed actions and the chunk-start state -- then generates the new chunk;
* ``reset`` (RoboDojo sends it twice per episode) drops the memory and the streams; the episode counter advances on the
  first ``update_obs`` of the next episode.

Returned actions: C dicts ``{left,right}_ee_pose`` (link6 in the env-relative world frame, xyz + wxyz) and
``{left,right}_ee_joint_state`` (normalized opening), the evaluator solving IK per tick (``action_type: ee``).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"
if str(POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(POLICY_DIR))

from sana_wam_causal import shared  # noqa: E402,F401  (puts policy/SANA_WAM on sys.path)
from sana_wam_causal.attention_probe import AttentionProbe  # noqa: E402
from sana_wam_causal.contract import MEMORY_VIDEO_AND_ACTION  # noqa: E402
from sana_wam_causal.runtime import CausalRuntime, timed  # noqa: E402

from sana_wam_min.actions import model_action_to_absolute  # noqa: E402
from sana_wam_min.config import ROBOT_BASE_EEF_ONLY  # noqa: E402
from sana_wam_min.eef import (  # noqa: E402
    ARM_FIELDS,
    ArxX5Kinematics,
    eef_state_slot_mask,
    fill_eef_state_slots,
    matrix_to_rot6d,
    native_ee_actions_from_action80,
    observed_link6_discrepancy,
    root_transforms,
    rot6d_to_matrix,
)
from sana_wam_min.robodojo_io import (  # noqa: E402
    ROBODOJO_ARM_DIM,
    ROBODOJO_FROZEN_IMAGE_HEIGHT,
    ROBODOJO_FROZEN_IMAGE_WIDTH,
    ROBODOJO_VIEW_ORDER,
    ROBOT80_JOINT_SLOTS_12,
    instruction_from_obs,
    state80_from_obs,
)
from sana_wam_min.robot80 import normalize_action, normalize_state, normalized_gripper_bounds  # noqa: E402

TAG = "[SANA_WAM_CAUSAL]"
CHECKPOINT_EXPLICIT_KEYS = ("checkpoint_dir", "checkpoint_path", "ckpt_dir", "model_dir")
# What memory keeps as an executed chunk's 32 action rows (memory_content video_and_action only): executed = the EE
# commands sent (the default); predicted = the raw model output; reached = the poses the arms actually reached (FK of
# the measured joints after each action, gripper as observed) -- training's rows are both at once (the dataset state is
# the previous command)
COMMIT_ACTIONS = ("executed", "predicted", "reached")
# full = the deployment (memory of the whole episode); none = the memory ablation: every chunk reopens the memory at
# its own start frame (the sink, and the frozen first chunk of a padded window, of that frame) and is generated as the
# first emission of an episode starting there (chunk 1, chunk 0 on a first-frame window) -- the episode protocol
# trains no memoryless window
MEMORY_MODES = ("full", "none")
SEED_DOMAIN = "sana_wam_causal_xpolicylab.diffusion_seed.v1"
MAX_JSON_SAFE_INTEGER = 2**53 - 1
DEFAULT_INSTRUCTION = "follow the instruction"
EEF_POSE_CHECK_WARN_M = 0.005
EEF_POSE_CHECK_WARN_DEG = 1.0


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    number = _optional_float(value)
    return None if number is None else int(number)


def check_commit_actions(commit_actions: str, memory_content: str) -> None:
    """A video_only memory keeps no action rows, so only the default ``commit_actions`` is meaningful there."""

    if commit_actions not in COMMIT_ACTIONS:
        raise ValueError(f"commit_actions must be one of {COMMIT_ACTIONS}, got {commit_actions!r}")
    if memory_content != MEMORY_VIDEO_AND_ACTION and commit_actions != "executed":
        raise ValueError(
            f"commit_actions={commit_actions!r} needs memory_content {MEMORY_VIDEO_AND_ACTION}: this checkpoint's memory "
            f"({memory_content}) keeps the target latents of a committed chunk alone, no action rows"
        )


def relative_rmse(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """RMS(estimate - reference) / RMS(reference) in fp32."""

    estimate, reference = estimate.float(), reference.float()
    return float(((estimate - reference) ** 2).mean().sqrt() / (reference**2).mean().sqrt().clamp_min(1e-8))


def derive_diffusion_seed(seed_base: int, eval_seed: int, episode_index: int, chunk_index: int) -> int:
    """Deterministic per-(eval seed, episode, chunk) seed: sha256 of the four integers, 53-bit."""

    preimage = b"\x00".join(
        str(part).encode("utf-8")
        for part in (SEED_DOMAIN, int(seed_base), int(eval_seed), int(episode_index), int(chunk_index))
    )
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big") & MAX_JSON_SAFE_INTEGER


def _rgb_frame(color: Any, camera: str, strict_image_size: bool) -> np.ndarray:
    """A C-contiguous uint8 HxWx3 COPY of a decoded RGB array (the ws arrays are read-only and get reused)."""

    frame = np.asarray(color)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{camera}: color must be an HxWx3 array, got shape {frame.shape} dtype {frame.dtype}")
    if frame.dtype != np.uint8:
        raise ValueError(f"{camera}: color must be a uint8 RGB array, got dtype {frame.dtype}; frames are never rescaled")
    expected = (ROBODOJO_FROZEN_IMAGE_HEIGHT, ROBODOJO_FROZEN_IMAGE_WIDTH, 3)
    if strict_image_size and tuple(frame.shape) != expected:
        raise ValueError(f"{camera}: strict_image_size requires {expected}, got {tuple(frame.shape)}")
    return np.array(frame, dtype=np.uint8, copy=True, order="C")


def executed_rows(action80_raw: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """The raw ``[C, 80]`` rows the evaluator actually receives: every active rot6d pair replaced by the rotation it
    encodes (Gram-Schmidt, ``rot6d_to_matrix``, as ``native_ee_actions_from_action80`` builds the EE pose); positions
    and the already clipped gripper closedness unchanged."""

    rows = action80_raw.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    mask = np.asarray(action_mask.detach().cpu().numpy(), dtype=np.bool_)
    for _side, _joint_slice, _position_slice, rotation_slice, _gripper in ARM_FIELDS:
        active = mask[:, rotation_slice].all(axis=1)
        for k in np.flatnonzero(active):
            rotation = rot6d_to_matrix(rows[k, rotation_slice].astype(np.float64))
            rows[k, rotation_slice] = matrix_to_rot6d(rotation).astype(np.float32)
    if not np.isfinite(rows).all():
        raise ValueError("executed rows must be finite")
    return torch.from_numpy(rows)


def command_vs_reached(commanded_raw: np.ndarray, reached_raw: np.ndarray, mask: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    """Per arm, over a chunk's active rows: (mean position gap [m], max position gap [m], max rotation gap [deg],
    max gripper closedness gap) between the commanded EE targets and the poses reached after them."""

    commanded = np.asarray(commanded_raw, dtype=np.float64)
    reached = np.asarray(reached_raw, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    gaps = {}
    for side, _joints, position_slice, rotation_slice, gripper in ARM_FIELDS:
        rows = np.flatnonzero(mask[:, position_slice].all(axis=1) & mask[:, rotation_slice].all(axis=1))
        if rows.size == 0:
            continue
        position = np.linalg.norm(commanded[rows, position_slice] - reached[rows, position_slice], axis=1)
        angles = []
        for k in rows:
            relative = rot6d_to_matrix(commanded[k, rotation_slice]).T @ rot6d_to_matrix(reached[k, rotation_slice])
            angles.append(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
        grip_rows = rows[mask[rows, gripper]]
        grip = float(np.abs(commanded[grip_rows, gripper] - reached[grip_rows, gripper]).max()) if grip_rows.size else 0.0
        gaps[side] = (float(position.mean()), float(position.max()), float(max(angles)), grip)
    return gaps


def format_command_vs_reached(gaps: Mapping[str, tuple[float, float, float, float]]) -> str:
    return " | ".join(
        f"{side[0].upper()} pos {1000 * mean:.1f}/{1000 * peak:.1f} mm rot {deg:.2f} deg grip {grip:.2f}"
        for side, (mean, peak, deg, grip) in gaps.items()
    )


class Model(ModelTemplate):
    """SANA_WAM_CAUSAL policy: one teacher-forced commit + one chunk generation per ``get_action``."""

    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        cfg = self.model_cfg

        action_type = str(cfg.get("action_type") or "ee").strip().lower()
        if action_type != "ee":
            raise ValueError(
                "SANA_WAM_CAUSAL serves the EEF-only causal line, which predicts EE poses only: action_type must be "
                f"'ee', got {action_type!r}"
            )
        self.action_type = action_type
        env_cfg_type = cfg.get("env_cfg_type")
        if not env_cfg_type:
            raise ValueError("env_cfg_type is required")
        self.env_cfg_type = str(env_cfg_type)
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        arm_dims = [int(v) for v in self.robot_action_dim_info["arm_dim"]]
        ee_dims = [int(v) for v in self.robot_action_dim_info["ee_dim"]]
        if arm_dims != [ROBODOJO_ARM_DIM, ROBODOJO_ARM_DIM] or ee_dims != [1, 1]:
            raise ValueError(
                "SANA_WAM_CAUSAL supports dual ARX-X5 arms only (arm_dim [6, 6], ee_dim [1, 1]); "
                f"env_cfg_type={self.env_cfg_type!r} resolves to {self.robot_action_dim_info!r}"
            )
        weight_dtype = str(cfg.get("weight_dtype") or "bfloat16")
        if weight_dtype not in ("bfloat16", "bf16"):
            raise ValueError(f"only bfloat16 weights are validated, got weight_dtype={weight_dtype!r}")
        self.device = torch.device(str(cfg.get("device") or "cuda"))
        self.strict_image_size = _is_true(cfg.get("strict_image_size", False))
        self.diffusion_seed_base = int(cfg.get("diffusion_seed_base") or 20261003)
        self.eval_seed = _optional_int(cfg.get("seed")) or 0
        self.default_instruction = str(cfg.get("default_instruction") or DEFAULT_INSTRUCTION)
        self.eef_pose_check = _is_true(cfg.get("eef_pose_check", True))
        self.commit_actions = str(cfg.get("commit_actions") or "executed").strip().lower()
        if self.commit_actions not in COMMIT_ACTIONS:
            raise ValueError(f"commit_actions must be one of {COMMIT_ACTIONS}, got {self.commit_actions!r}")
        self.memory_mode = str(cfg.get("memory_mode") or "full").strip().lower()
        if self.memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}, got {self.memory_mode!r}")
        if cfg.get("softmax_window") is not None:
            raise ValueError(
                "softmax_window was retired with the chunk-0-only observation protocol: the per-chunk observation "
                "protocol reads the newest sliding_window_chunks committed chunks, as training does"
            )
        self.max_chunks_per_episode = _optional_int(cfg.get("max_chunks_per_episode"))
        self.log_world_model_error = _is_true(cfg.get("log_world_model_error", True))
        self.attention_dump_dir = str(cfg.get("attention_dump_dir") or "").strip() or None
        # Diagnostic predicted-video dump (null = off): per decision the chunk's generated target latents and, at the next
        # decision, its observed (streamed) target latents, each after the chunk's observation latent (its start frame
        # encoded alone) so the causal VAE decodes them on their own -> <dir>/ep<E>_env00_chunk<C>.npz /
        # ep<E>_env00_chunk<C>_observed.npz (fp16, packed and normalized like training); decode offline with
        # tools/decode_prediction_latents.py (no decode inside the timed inference)
        raw_dump = str(cfg.get("prediction_latent_dir") or "").strip()
        self.prediction_latent_dir = None if raw_dump.lower() in ("", "null", "none", "false", "off") else Path(raw_dump)
        if self.prediction_latent_dir is not None:
            self.prediction_latent_dir.mkdir(parents=True, exist_ok=True)
        self.allow_donor_checkpoint = _is_true(cfg.get("allow_donor_checkpoint", False))
        self.kinematics = ArxX5Kinematics(cfg.get("urdf_path") or None)
        self.world_from_base = root_transforms(cfg.get("robot_root_poses") or None)

        self.ckpt_dir = resolve_checkpoint_root(
            dict(cfg), CHECKPOINTS_DIR, policy_dir=POLICY_DIR, explicit_keys=CHECKPOINT_EXPLICIT_KEYS, must_exist=True
        )
        self.runtime = CausalRuntime.from_paths(
            str(self.ckpt_dir),
            text_encoder_path=str(
                cfg.get("text_encoder_path") or os.environ.get("SANA_WAM_TEXT_ENCODER_PATH") or "google/gemma-2-2b-it"
            ),
            vae_path=str(cfg.get("vae_path") or os.environ.get("SANA_WAM_VAE_PATH") or "Efficient-Large-Model/LTX-2.3-Diffusers"),
            device=self.device,
            normalization_path=cfg.get("normalization_path") or None,
            expected_normalization_sha256=cfg.get("normalization_sha256") or None,
            steps=_optional_int(cfg.get("sampling_steps")),
            cfg_scale=_optional_float(cfg.get("cfg_scale")),
            video_cfg_scale=_optional_float(cfg.get("video_cfg_scale")),
            action_cfg_scale=_optional_float(cfg.get("action_cfg_scale")),
            flow_shift=_optional_float(cfg.get("flow_shift")),
            action_flow_shift=_optional_float(cfg.get("action_flow_shift")),
            view_resize=cfg.get("view_resize"),
            allow_donor_checkpoint=self.allow_donor_checkpoint,
        )
        frontend = self.runtime.frontend
        if frontend.action_mode != "robot_base_eef" or frontend.robot_base_eef_layout != ROBOT_BASE_EEF_ONLY:
            raise ValueError(
                "SANA_WAM_CAUSAL serves the EEF-only robot_base_eef line; this checkpoint resolves to action mode "
                f"{frontend.action_mode!r}, robot_base_eef_layout {frontend.robot_base_eef_layout!r}"
            )
        if frontend.eef_target_mode != "absolute":
            raise ValueError(f"SANA_WAM_CAUSAL serves absolute EEF targets; this checkpoint is {frontend.eef_target_mode!r}")
        self.normalization = frontend.normalization
        self.gripper_bounds = normalized_gripper_bounds(self.normalization)
        contract = self.runtime.contract
        self.chunk_actions = int(contract.actions_per_chunk)
        self.frame_stride = int(contract.video_frame_stride)
        self.memory_content = str(contract.memory_content)
        self.first_chunk = int(contract.first_chunk)   # the episode's first emission: 1 (padded window) or 0 (first frame)
        # causal_observation: every chunk is generated from its own observation latent (its start frame encoded alone)
        self.causal_observation = bool(contract.causal_observation)
        check_commit_actions(self.commit_actions, self.memory_content)

        # diagnostics: per-chunk memory-read aggregates of every generation forward (attention_probe.py)
        self._probe = AttentionProbe(self.runtime.model) if self.attention_dump_dir else None
        self._session = None
        self._streams = None
        self._episode_open = False
        self._episode_index = 0
        self._chunk_index = 0
        self._latest_obs: Optional[Mapping[str, Any]] = None
        self._tau: Optional[int] = None
        self._frames: dict[int, list[np.ndarray]] = {}
        self._reached: dict[int, np.ndarray] = {}   # tick tau -> raw measured row (the pose reached after action tau - 1)
        self._pending: Optional[dict[str, Any]] = None
        self._instruction: Optional[str] = None
        self._size_warned = False
        report = self.runtime.load_report
        print(
            f"{TAG} ready: ckpt={self.ckpt_dir} {contract.describe()} softmax_positions=rank+sink "
            f"steps={frontend.steps} video_cfg_scale={frontend.video_cfg_scale} action_cfg_scale={frontend.action_cfg_scale} "
            f"flow_shift={frontend.flow_shift} action_flow_shift={frontend.action_flow_shift} "
            f"view_resize={frontend.view_resize} ({frontend.view_resize_source}) commit_actions={self.commit_actions} "
            f"memory_mode={self.memory_mode} attention_dump_dir={self.attention_dump_dir} "
            f"prediction_latent_dir={self.prediction_latent_dir} "
            f"gdn_recurrence={report.get('gdn_recurrence')} normalization_sha256={self.normalization.sha256} "
            f"run_record={report.get('run_record')} device={self.device} action_type={self.action_type}",
            flush=True,
        )
        print(f"{TAG} prompt Action Mode sentence (from the training yaml): {frontend.action_mode_text!r}", flush=True)
        print(f"{TAG} text contract: {frontend.text_contract}; canvas Observation View: {frontend.canvas_view_text!r}", flush=True)

    # -- observations ------------------------------------------------------------------------------------------------

    def _frames_from_obs(self, obs: Mapping[str, Any]) -> list[np.ndarray]:
        vision = obs["vision"]
        frames = [_rgb_frame(vision[cam]["color"], cam, self.strict_image_size) for cam in ROBODOJO_VIEW_ORDER]
        expected = (ROBODOJO_FROZEN_IMAGE_HEIGHT, ROBODOJO_FROZEN_IMAGE_WIDTH, 3)
        if not self._size_warned and any(tuple(f.shape) != expected for f in frames):
            warnings.warn(f"camera frames are not {expected} (got {[tuple(f.shape) for f in frames]})", stacklevel=2)
            self._size_warned = True
        return frames

    def _instruction_from_obs(self, obs: Mapping[str, Any]) -> str:
        try:
            text = instruction_from_obs(obs)
        except (KeyError, IndexError, TypeError):
            return self.default_instruction
        text = text.strip()
        return text if text else self.default_instruction

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        if not self._episode_open:
            self._episode_open = True
            self._episode_index += 1
        self._latest_obs = obs
        if self._tau is None:
            return  # the observation the next get_action starts from (episode start)
        self._tau += 1
        if self._tau > self.chunk_actions:
            raise RuntimeError(
                f"{TAG} {self._tau} observations after a chunk of {self.chunk_actions} actions: the client must ask for "
                "the next chunk after executing the last one"
            )
        if self._tau % self.frame_stride == 0:
            self._frames[self._tau] = self._frames_from_obs(obs)
        self._reached[self._tau] = self._measured_row(obs)

    def update_obs_batch(self, obs_list) -> None:
        if isinstance(obs_list, Mapping):
            obs_list = [obs_list]
        if len(obs_list) != 1:
            raise ValueError("SANA_WAM_CAUSAL serves one environment (eval_batch: false)")
        self.update_obs(obs_list[0])

    # -- actions -----------------------------------------------------------------------------------------------------

    def _measured_row(self, obs: Mapping[str, Any]) -> np.ndarray:
        """The raw Robot80 row of an observation: measured joints, their URDF-FK base-frame E poses (position +
        rot6d) and the observed gripper closedness -- the chunk-start state, and under ``commit_actions: reached`` the
        action row of the action that led to it."""

        state80, _ = state80_from_obs(obs["state"])
        state80, _ = fill_eef_state_slots(state80, eef_state_slot_mask(True), self.kinematics)
        return state80

    def _anchor(self, obs: Mapping[str, Any], *, check_pose: bool = False) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """(raw state80, EEF-only mask80, normalized state [80]) of the chunk-start observation: measured joints ->
        URDF FK -> base-frame E pose (position + rot6d), gripper closedness from the observed opening; joints masked.
        ``check_pose`` (the first decision of an episode) also runs the observed-pose check."""

        state80, mask80 = self._measured_row(obs), eef_state_slot_mask(True)
        mask80 = np.array(mask80, dtype=np.bool_, copy=True)
        mask80[list(ROBOT80_JOINT_SLOTS_12)] = False
        if check_pose and self.eef_pose_check:
            self._check_observed_eef_pose(obs["state"], state80_from_obs(obs["state"])[0])
        normalized = normalize_state(state80, mask80, self.normalization)
        return state80, mask80, normalized

    def _check_observed_eef_pose(self, obs_state: Mapping[str, Any], state80_measured: np.ndarray) -> None:
        """Once per episode: the evaluator's ``*_ee_pose`` vs FK(measured joints) through the root poses (sub-mm when
        the URDF / root pose / axis conventions match the simulator), as SANA_WAM does."""

        try:
            state_fk, _ = fill_eef_state_slots(state80_measured, eef_state_slot_mask(True), self.kinematics)
            gaps = observed_link6_discrepancy(obs_state, state_fk, self.world_from_base)
        except (KeyError, ValueError, TypeError) as error:
            print(f"{TAG} eef check ep {self._episode_index}: skipped ({error})", flush=True)
            return
        if not gaps:
            return
        text = " | ".join(f"{side} {1000 * pos:.2f} mm / {deg:.3f} deg" for side, (pos, deg) in gaps.items())
        print(f"{TAG} eef check ep {self._episode_index}: observed *_ee_pose vs FK(measured joints): {text}", flush=True)
        if any(pos > EEF_POSE_CHECK_WARN_M or deg > EEF_POSE_CHECK_WARN_DEG for pos, deg in gaps.values()):
            warnings.warn(f"observed end-effector pose disagrees with FK(measured joints): {text}", stacklevel=2)

    def _chunk_views(self) -> list[list[torch.Tensor]]:
        """The executed chunk's C / s observed frames (ticks s, 2s, .., C) as per-view model tensors (``view_frames``)."""

        ticks = list(range(self.frame_stride, self.chunk_actions + 1, self.frame_stride))
        missing = [t for t in ticks if t not in self._frames]
        if missing:
            raise RuntimeError(f"{TAG} frames of ticks {missing} were not observed for chunk {self._pending['chunk']}")
        per_tick = [self.runtime.view_frames(self._frames[t]) for t in ticks]
        return [[views[v] for views in per_tick] for v in range(len(per_tick[0]))]

    def _commit_pending(self, timings: dict, *, commit: bool = True) -> tuple[Optional[float], str]:
        """Encode the executed chunk on in the per-view streams and fold it into memory (``commit``); return the
        world-model error of its prediction and the commanded-vs-reached gap of its actions. ``commit=False``
        (``memory_mode: none``) encodes it on the stream its memory reopened at, for the error alone."""

        pending = self._pending
        if self._tau != self.chunk_actions:
            raise RuntimeError(
                f"{TAG} chunk {pending['chunk']} was executed for {self._tau} of {self.chunk_actions} ticks; only a "
                "complete chunk can be committed (the causal policy never trained on a partial context chunk)"
            )
        # the pose reached after action j is the one measured at tick j + 1
        reached = np.stack([self._reached[t] for t in range(1, self.chunk_actions + 1)])
        mask_cpu = pending["mask_cpu"]
        gap = format_command_vs_reached(command_vs_reached(pending["executed_raw"].numpy(), reached, mask_cpu.numpy()))
        if commit and self.commit_actions == "reached":
            committed = normalize_action(torch.from_numpy(reached), mask_cpu, self.normalization).masked_fill(~mask_cpu, 0)
            pending["committed_action"] = committed.unsqueeze(0).to(device=self.runtime.device)
        if not commit and not self.log_world_model_error:
            return None, gap
        latent, timings["stream_s"] = timed(self._streams.encode_chunk, self._chunk_views())
        if self.prediction_latent_dir is not None and pending.get("dump_observation") is not None:
            self._dump_latent(pending["decision"], pending["dump_observation"], latent, observed=True)
        wm_error = None
        if self.log_world_model_error and pending.get("predicted_latent") is not None:
            wm_error = relative_rmse(pending["predicted_latent"], latent)
        if not commit:
            return wm_error, gap
        _, timings["commit_s"] = timed(
            self._session.commit_chunk,
            latent,
            pending["committed_action"],
            anchor_state=pending["anchor_norm"],
            anchor_state_mask=pending["anchor_mask"],
            action_mask=pending["action_mask"],
        )
        return wm_error, gap

    def _dump_latent(self, decision: int, observation: torch.Tensor, target: torch.Tensor, *, observed: bool) -> None:
        """``prediction_latent_dir``: write one chunk's latents [observation | targets] (fp16, packed and normalized like
        training) with what the offline decoder needs: the decision, the control tick the chunk starts at, its executed
        ticks, the per-view latent shapes, the layout and the video frame stride. ``observed`` selects the streamed
        (real) targets over the generated ones."""

        name = f"ep{int(self._episode_index):04d}_env00_chunk{int(decision):03d}" + ("_observed" if observed else "")
        latent = torch.cat((observation.to(target.dtype), target), dim=2).detach().to(device="cpu", dtype=torch.float16)
        shapes = self._streams.view_shapes if self.runtime.strip else ((int(latent.shape[-2]), int(latent.shape[-1])),)
        out, tmp = self.prediction_latent_dir / f"{name}.npz", self.prediction_latent_dir / f"{name}.tmp.npz"
        np.savez_compressed(
            tmp,
            video_latent=latent.numpy(),
            episode=np.int64(self._episode_index),
            env=np.int64(0),
            chunk=np.int64(decision),
            tick=np.int64(decision * self.chunk_actions),
            executed=np.int64(self.chunk_actions),
            view_latent_shapes=np.asarray(shapes, dtype=np.int64),
            visual_layout=np.asarray(str(self.runtime.contract.visual_layout)),
            video_frame_stride=np.int64(self.frame_stride),
            observed=np.bool_(observed),
            instruction=np.asarray(str(self._instruction)),
        )
        os.replace(tmp, out)

    def _open_memory(self, views: list[torch.Tensor], anchor_raw: np.ndarray, anchor_norm: torch.Tensor,
                     action_mask: torch.Tensor, timings: dict) -> torch.Tensor:
        """(Re)open the streams and the memory at this frame: the sink (the first emitted chunk's observation), then a
        padded window's frozen first chunk (its rows hold the robot at this state under video_and_action); a
        first-frame window's streams open on the sink alone. Returns the sink's latent."""

        device = self.runtime.device
        self._streams = self.runtime.new_streams()
        (sink, frozen), timings["open_stream_s"] = timed(self._streams.start, views)
        if self._session is None:
            raise RuntimeError(f"{TAG} no session to open")
        self._session.reset_memory()
        self._session.view_shapes = self._streams.view_shapes if self.runtime.strip else None
        if self.first_chunk == 0:
            _, timings["open_commit_s"] = timed(self._session.commit_sink, sink)
            return sink
        mask_cpu = action_mask[0].detach().cpu()
        hold_raw = torch.from_numpy(np.repeat(anchor_raw[None].astype(np.float32), self.chunk_actions, axis=0))
        hold = normalize_action(hold_raw, mask_cpu, self.normalization).masked_fill(~mask_cpu, 0)

        def opening():
            self._session.commit_sink(sink)
            self._session.commit_chunk(
                frozen, hold.unsqueeze(0).to(device), anchor_state=anchor_norm.to(device),
                anchor_state_mask=action_mask[0, 0], action_mask=action_mask,
            )

        _, timings["open_commit_s"] = timed(opening)
        return sink

    def get_action(self) -> list[dict[str, np.ndarray]]:
        obs = self._latest_obs
        if obs is None:
            raise ValueError("no observation stored; call update_obs first")
        start = time.perf_counter()
        timings: dict[str, float] = {}
        frames = self._frames_from_obs(obs)
        instruction = self._instruction_from_obs(obs)
        first = self._session is None
        wm_error, gap = None, None
        if first:
            self._session = self.runtime.new_session(instruction)
            self._session.probe = self._probe
            self._instruction = instruction
            print(f"{TAG} episode {self._episode_index} instruction: {instruction!r}", flush=True)
        else:
            if instruction != self._instruction:
                raise RuntimeError(
                    f"{TAG} the instruction changed within episode {self._episode_index} ({self._instruction!r} -> "
                    f"{instruction!r}); the memory is caption-conditioned"
                )
            wm_error, gap = self._commit_pending(timings, commit=self.memory_mode == "full")
        decision = self._chunk_index   # the episode's decision count (0-based)
        if self.max_chunks_per_episode is not None and decision >= self.max_chunks_per_episode:
            raise RuntimeError(f"{TAG} episode {self._episode_index} reached max_chunks_per_episode={self.max_chunks_per_episode}")

        anchor_raw, anchor_mask, anchor_norm = self._anchor(obs, check_pose=first)
        device = self.runtime.device
        anchor_mask_t = torch.as_tensor(anchor_mask, dtype=torch.bool, device=device)
        action_mask = anchor_mask_t.reshape(1, 1, -1).expand(1, self.chunk_actions, -1).contiguous()
        views = self.runtime.view_frames(frames)
        sink = None
        if first or self.memory_mode == "none":
            # the episode opens here (memory none: every chunk reopens it at its own start frame); the first emitted
            # chunk's observation is the sink, the first frame encoded alone
            sink = self._open_memory(views, anchor_raw, anchor_norm, action_mask, timings)
        observation_kwargs = {}
        dump_observation = None
        if self.causal_observation:
            # the chunk's own observation latent: its start frame encoded alone, i.e. the sink's latent at the frame the
            # memory (re)opened at, else this frame on fresh streams of its own (Sana encode_observation)
            observation = sink
            if observation is None:
                observation, timings["observation_encode_s"] = timed(self._streams.encode_observation, views)
            observation_kwargs["observation"] = observation
            dump_observation = observation
        if self.prediction_latent_dir is not None and dump_observation is None:
            # the decoder's first latent: the chunk's start frame encoded alone (the sink at an opening)
            dump_observation = sink if sink is not None else self._streams.encode_observation(views)
        chunk = self._session.chunk_idx   # the training chunk this window is (the first emitted one at the first frame)
        expected = self.first_chunk if self.memory_mode == "none" else decision + self.first_chunk
        if chunk != expected:
            raise RuntimeError(f"{TAG} the session is at chunk {chunk} for decision {decision} (expected {expected})")
        read_ids = self._session.read_chunk_ids()
        origin = self._session.softmax_origin()   # the read is numbered from 0 here, the sink one frame before it
        seed = derive_diffusion_seed(self.diffusion_seed_base, self.eval_seed, self._episode_index, decision)
        generator = torch.Generator(device=device).manual_seed(seed)
        (action_norm, predicted_latent), timings["generate_s"] = timed(
            self._session.generate_chunk,
            anchor_state=anchor_norm.to(device),
            anchor_state_mask=anchor_mask_t,
            action_mask=action_mask,
            generator=generator,
            gripper_bounds=self.gripper_bounds,
            **observation_kwargs,
        )
        if self._probe is not None:
            self._probe.write(
                self.attention_dump_dir, self._episode_index,
                {"instruction": instruction, "memory_mode": self.memory_mode, "memory_content": self.memory_content,
                 "chunk_observation": False, "decision": int(decision), "diffusion_seed": int(seed),
                 "checkpoint": str(self.ckpt_dir)},
            )
        if self.prediction_latent_dir is not None:
            self._dump_latent(decision, dump_observation, predicted_latent, observed=False)
        mask_cpu = action_mask[0].detach().cpu()
        action_raw = model_action_to_absolute(
            action_norm[0].detach().to(device="cpu", dtype=torch.float32), mask_cpu, self.normalization, anchor_raw, anchor_mask
        )
        executed_raw = executed_rows(action_raw, mask_cpu)
        if self.commit_actions == "executed":
            committed = normalize_action(executed_raw, mask_cpu, self.normalization)
        else:
            committed = action_norm[0].detach().to(device="cpu", dtype=torch.float32).masked_fill(~mask_cpu, 0)
        actions = native_ee_actions_from_action80(executed_raw, self.world_from_base)
        if len(actions) != self.chunk_actions:
            raise RuntimeError(f"{TAG} produced {len(actions)} actions for a chunk of {self.chunk_actions}")

        self._pending = {
            "chunk": chunk,
            "anchor_norm": anchor_norm.to(device),
            "anchor_mask": anchor_mask_t,
            "action_mask": action_mask,
            "committed_action": committed.unsqueeze(0).to(device=device),
            "predicted_latent": predicted_latent.detach() if self.log_world_model_error else None,
            "executed_raw": executed_raw,
            "mask_cpu": mask_cpu,
            "decision": decision,
            "dump_observation": None if dump_observation is None else dump_observation.detach(),
        }
        self._frames = {}
        self._reached = {}
        self._tau = 0
        self._chunk_index = decision + 1
        total = time.perf_counter() - start
        detail = " ".join(f"{k}={v:.2f}" for k, v in timings.items())
        print(
            f"{TAG} ep {self._episode_index} decision {decision} chunk {chunk}: read_chunks={read_ids} origin={origin} "
            f"{detail} total_s={total:.2f}"
            + ("" if wm_error is None else f" world_model_rel_rmse(decision {decision - 1})={wm_error:.4f}")
            + ("" if gap is None else f" cmd_vs_reached(decision {decision - 1}): {gap}"),
            flush=True,
        )
        return [
            {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in action.items()} for action in actions
        ]

    def get_action_batch(self, env_idx_list=None) -> list[list[dict[str, np.ndarray]]]:
        if env_idx_list is not None:
            ids = np.asarray(env_idx_list).reshape(-1).tolist() if not isinstance(env_idx_list, (int, np.integer)) else [int(env_idx_list)]
            if len(ids) != 1:
                raise ValueError("SANA_WAM_CAUSAL serves one environment (eval_batch: false)")
        return [self.get_action()]

    def reset(self) -> None:
        """Drop the episode's memory; idempotent (RoboDojo sends two resets per episode)."""

        self._session = None
        self._streams = None
        self._episode_open = False
        self._latest_obs = None
        self._tau = None
        self._frames = {}
        self._reached = {}
        self._pending = None
        self._instruction = None
        self._chunk_index = 0

    def prepare_case(self, case_meta=None) -> None:
        return None

    def on_trial_end(self, result=None) -> None:
        return None
