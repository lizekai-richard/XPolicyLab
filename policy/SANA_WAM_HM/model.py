"""XPolicyLab adapter of the SANA hierarchical-memory policy (RoboDojo ARX-X5, EEF-only absolute targets, EE-pose actions).

The policy runs in-process through the vendored, Sana-free ``sana_wam_hm`` package (on top of SANA_WAM's
``sana_wam_min`` runtime). Its memory is two streams of causal-VAE latents of the episode's own camera canvases
(Sana rwm/sparse-memory, ``hierarchical_memory: {gdn: dense|sparse, softmax: dense|sparse, sparse_frames: m}``):

* decision c anchors at tick t = 32c (tick 0 = the episode's first observation, training's first paired row r0); every
  ``update_obs`` advances the tick (RoboDojo calls it once after every executed action, the last one right before the
  next ``get_action``) and keeps that tick's canvas, quantized to uint8 as the training payload -- every tick, since the
  sparse memory samples arbitrary rows of the whole history;
* the dense memory (read by blocks configured ``dense``): one persistent causal-VAE stream, L_0 = the first frame alone
  at the episode start, then L_c from the 8 stride-4 canvases of the executed chunk (ticks 32(c-1)+4 .. 32c);
* the sparse memory (read by blocks configured ``sparse``): every decision encodes [tick 0, s_0..s_{m-1}] (m ticks
  sampled uniformly over [1, t], ``memory.sparse_rows``) as one causal sequence -> [A_0, A_1..];
* every ``get_action``: (decision c >= 1) the executed chunk's dense latent is absorbed and the previous prediction is
  scored against the chunk it executed; the row-t canvas is encoded alone as the observation; the chunk
  [obs | target] + [state | 32 actions] is denoised against the memory (``deploy_form`` streaming: the GDN state of
  the committed dense latents + the sparse memory riding every step; full: the training forward of the decision's
  sample) and returned;
* ``reset`` (RoboDojo sends it twice per episode) drops the memory and the canvases; the episode counter advances on the
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

from sana_wam_hm import shared  # noqa: E402,F401  (puts policy/SANA_WAM on sys.path)
from sana_wam_hm import memory as hm  # noqa: E402
from sana_wam_hm.runtime import HMRuntime  # noqa: E402

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
from sana_wam_min.robot80 import normalize_state, normalized_gripper_bounds  # noqa: E402

TAG = "[SANA_WAM_HM]"
CHECKPOINT_EXPLICIT_KEYS = ("checkpoint_dir", "checkpoint_path", "ckpt_dir", "model_dir")
SEED_DOMAIN = "sana_wam_hm_xpolicylab.diffusion_seed.v1"
MAX_JSON_SAFE_INTEGER = 2**53 - 1
DEFAULT_INSTRUCTION = "follow the instruction"
EEF_POSE_CHECK_WARN_M = 0.005
EEF_POSE_CHECK_WARN_DEG = 1.0
# the frame-anchor recipes train K = ceil((t - r0) / 32) <= 47 (the longest RoboDojo span); a later decision runs the
# GDN state / positions past anything trained
TRAINED_MAX_K = 47


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


def _optional_str(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
        return None
    return str(value).strip()


def relative_rmse(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """RMS(estimate - reference) / RMS(reference) in fp32."""

    estimate, reference = estimate.float(), reference.float()
    return float(((estimate - reference) ** 2).mean().sqrt() / (reference**2).mean().sqrt().clamp_min(1e-8))


def derive_diffusion_seed(seed_base: int, eval_seed: int, episode_index: int, decision: int) -> int:
    """Deterministic per-(eval seed, episode, decision) seed: sha256 of the four integers, 53-bit."""

    preimage = b"\x00".join(
        str(part).encode("utf-8")
        for part in (SEED_DOMAIN, int(seed_base), int(eval_seed), int(episode_index), int(decision))
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
    encodes (Gram-Schmidt, as ``native_ee_actions_from_action80`` builds the EE pose)."""

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


def timed(fn, *args, **kwargs):
    """``(result, seconds)`` with a CUDA sync when a GPU is in use (per-decision latency logging)."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


class Model(ModelTemplate):
    """SANA_WAM_HM policy: one decision (dense absorb + observation + sparse memory + chunk generation) per
    ``get_action``."""

    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        cfg = self.model_cfg

        action_type = str(cfg.get("action_type") or "ee").strip().lower()
        if action_type != "ee":
            raise ValueError(
                "SANA_WAM_HM serves the EEF-only hierarchical-memory line, which predicts EE poses only: action_type must "
                f"be 'ee', got {action_type!r}"
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
                "SANA_WAM_HM supports dual ARX-X5 arms only (arm_dim [6, 6], ee_dim [1, 1]); "
                f"env_cfg_type={self.env_cfg_type!r} resolves to {self.robot_action_dim_info!r}"
            )
        weight_dtype = str(cfg.get("weight_dtype") or "bfloat16")
        if weight_dtype not in ("bfloat16", "bf16"):
            raise ValueError(f"only bfloat16 weights are validated, got weight_dtype={weight_dtype!r}")
        self.device = torch.device(str(cfg.get("device") or "cuda"))
        self.strict_image_size = _is_true(cfg.get("strict_image_size", False))
        self.diffusion_seed_base = int(cfg.get("diffusion_seed_base") or 20261012)
        self.eval_seed = _optional_int(cfg.get("seed")) or 0
        self.default_instruction = str(cfg.get("default_instruction") or DEFAULT_INSTRUCTION)
        self.eef_pose_check = _is_true(cfg.get("eef_pose_check", True))
        self.max_chunks_per_episode = _optional_int(cfg.get("max_chunks_per_episode"))
        self.log_world_model_error = _is_true(cfg.get("log_world_model_error", True))
        self.allow_donor_checkpoint = _is_true(cfg.get("allow_donor_checkpoint", False))
        self.kinematics = ArxX5Kinematics(cfg.get("urdf_path") or None)
        self.world_from_base = root_transforms(cfg.get("robot_root_poses") or None)

        self.ckpt_dir = resolve_checkpoint_root(
            dict(cfg), CHECKPOINTS_DIR, policy_dir=POLICY_DIR, explicit_keys=CHECKPOINT_EXPLICIT_KEYS, must_exist=True
        )
        self.runtime = HMRuntime.from_paths(
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
            deploy_form=_optional_str(cfg.get("deploy_form")),
            allow_donor_checkpoint=self.allow_donor_checkpoint,
        )
        frontend = self.runtime.frontend
        if frontend.action_mode != "robot_base_eef" or frontend.robot_base_eef_layout != ROBOT_BASE_EEF_ONLY:
            raise ValueError(
                "SANA_WAM_HM serves the EEF-only robot_base_eef line; this checkpoint resolves to action mode "
                f"{frontend.action_mode!r}, robot_base_eef_layout {frontend.robot_base_eef_layout!r}"
            )
        if frontend.eef_target_mode != "absolute":
            raise ValueError(f"SANA_WAM_HM serves absolute EEF targets; this checkpoint is {frontend.eef_target_mode!r}")
        self.normalization = frontend.normalization
        self.gripper_bounds = normalized_gripper_bounds(self.normalization)
        contract = self.runtime.contract
        self.contract = contract
        self.chunk_actions = int(contract.actions_per_chunk)

        self._session = None
        self._episode_open = False
        self._episode_index = 0
        self._latest_obs: Optional[Mapping[str, Any]] = None
        self._tau: Optional[int] = None
        self._tick = 0
        self._canvases: list[torch.Tensor] = []
        self._reached: dict[int, np.ndarray] = {}
        self._dense_stream = None
        self._pending: Optional[dict[str, Any]] = None
        self._instruction: Optional[str] = None
        self._size_warned = False
        self._k_warned = False
        report = self.runtime.load_report
        print(
            f"{TAG} ready: ckpt={self.ckpt_dir} {contract.describe()} deploy_form={self.runtime.deploy_form} "
            f"steps={frontend.steps} video_cfg_scale={frontend.video_cfg_scale} action_cfg_scale={frontend.action_cfg_scale} "
            f"flow_shift={frontend.flow_shift} action_flow_shift={frontend.action_flow_shift} "
            f"view_resize={frontend.view_resize} ({frontend.view_resize_source}) "
            f"gdn_recurrence={report.get('gdn_recurrence')} normalization_sha256={self.normalization.sha256} "
            f"device={self.device} action_type={self.action_type}",
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
        self._tick += 1
        if len(self._canvases) != self._tick:
            raise RuntimeError(f"{TAG} tick {self._tick} arrives with {len(self._canvases)} canvases kept")
        self._canvases.append(self.runtime.canvas(self._frames_from_obs(obs)))
        self._reached[self._tau] = self._measured_row(obs)

    def update_obs_batch(self, obs_list) -> None:
        if isinstance(obs_list, Mapping):
            obs_list = [obs_list]
        if len(obs_list) != 1:
            raise ValueError("SANA_WAM_HM serves one environment (eval_batch: false)")
        self.update_obs(obs_list[0])

    # -- robot rows ----------------------------------------------------------------------------------------------------

    def _measured_row(self, obs: Mapping[str, Any]) -> np.ndarray:
        """The raw Robot80 row of an observation: measured joints, their URDF-FK base-frame EE poses and the observed
        gripper closedness."""

        state80, _ = state80_from_obs(obs["state"])
        state80, _ = fill_eef_state_slots(state80, eef_state_slot_mask(True), self.kinematics)
        return state80

    def _anchor(self, obs: Mapping[str, Any], *, check_pose: bool = False) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """(raw state80, EEF-only mask80, normalized state [80]) of the decision's observation."""

        state80, mask80 = self._measured_row(obs), eef_state_slot_mask(True)
        mask80 = np.array(mask80, dtype=np.bool_, copy=True)
        mask80[list(ROBOT80_JOINT_SLOTS_12)] = False
        if check_pose and self.eef_pose_check:
            self._check_observed_eef_pose(obs["state"], state80_from_obs(obs["state"])[0])
        normalized = normalize_state(state80, mask80, self.normalization)
        return state80, mask80, normalized

    def _check_observed_eef_pose(self, obs_state: Mapping[str, Any], state80_measured: np.ndarray) -> None:
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

    # -- the memory ----------------------------------------------------------------------------------------------------

    def _absorb_executed_chunk(self, timings: dict) -> tuple[Optional[float], str]:
        """Close decision c - 1: the dense latent of the chunk it executed enters the memory; its prediction is scored
        against the latent training would have regressed it to. Returns (world-model error, commanded-vs-reached gap)."""

        pending = self._pending
        if self._tau != self.chunk_actions:
            raise RuntimeError(
                f"{TAG} decision {pending['decision']} was executed for {self._tau} of {self.chunk_actions} ticks; the "
                "memory takes complete chunks only"
            )
        reached = np.stack([self._reached[t] for t in range(1, self.chunk_actions + 1)])
        gap = format_command_vs_reached(command_vs_reached(pending["executed_raw"].numpy(), reached, pending["mask_cpu"].numpy()))
        session = self._session
        c = session.decision
        reference = None
        if session.reads_dense:
            ticks = hm.dense_chunk_ticks(c)
            latent, timings["dense_encode_s"] = timed(self._dense_stream.extend, [self._canvases[t] for t in ticks])
            _, timings["commit_s"] = timed(session.add_dense, latent)
            reference = latent
        elif self.log_world_model_error and pending.get("sparse_stream") is not None:
            # the sparse-only recipes encode the target after the sparse rows (Sana ``stream_rows``)
            ticks = hm.target_rows(pending["t"])
            reference, timings["target_encode_s"] = timed(pending["sparse_stream"].extend, [self._canvases[t] for t in ticks])
        wm_error = None
        if self.log_world_model_error and reference is not None and pending.get("predicted_latent") is not None:
            wm_error = relative_rmse(pending["predicted_latent"], reference)
        return wm_error, gap

    # -- actions -----------------------------------------------------------------------------------------------------

    def get_action(self) -> list[dict[str, np.ndarray]]:
        obs = self._latest_obs
        if obs is None:
            raise ValueError("no observation stored; call update_obs first")
        start = time.perf_counter()
        timings: dict[str, float] = {}
        instruction = self._instruction_from_obs(obs)
        wm_error, gap = None, None
        if self._session is None:
            session = self.runtime.new_session(instruction)
            self._session = session
            self._instruction = instruction
            self._tick = 0
            self._canvases = [self.runtime.canvas(self._frames_from_obs(obs))]
            print(f"{TAG} episode {self._episode_index} instruction: {instruction!r}", flush=True)
            if session.reads_dense:
                self._dense_stream = self.runtime.new_dense_stream()
                first, timings["dense_encode_s"] = timed(self._dense_stream.start, self._canvases[0])
                _, timings["commit_s"] = timed(session.add_dense, first)
        else:
            if instruction != self._instruction:
                raise RuntimeError(
                    f"{TAG} the instruction changed within episode {self._episode_index} ({self._instruction!r} -> "
                    f"{instruction!r}); the memory is caption-conditioned"
                )
            wm_error, gap = self._absorb_executed_chunk(timings)
        session = self._session
        c = session.decision
        t = hm.decision_row(c)
        if t != self._tick:
            raise RuntimeError(f"{TAG} decision {c} anchors at tick {t}, the episode is at tick {self._tick}")
        if self.max_chunks_per_episode is not None and c >= self.max_chunks_per_episode:
            raise RuntimeError(f"{TAG} episode {self._episode_index} reached max_chunks_per_episode={self.max_chunks_per_episode}")
        if c > TRAINED_MAX_K and not self._k_warned:
            warnings.warn(f"decision {c} is past the trained K <= {TRAINED_MAX_K}", stacklevel=2)
            self._k_warned = True

        observation_latent, timings["obs_encode_s"] = timed(self.runtime.encode_observation, self._canvases[t])
        sparse_latents, sparse_stream, ticks = None, None, None
        if session.reads_sparse:
            ticks = hm.sparse_ticks(c, self.contract.sparse_frames)
            (sparse_latents, sparse_stream), timings["sparse_encode_s"] = timed(
                self.runtime.encode_sparse, [self._canvases[tick] for tick in ticks]
            )

        anchor_raw, anchor_mask, anchor_norm = self._anchor(obs, check_pose=c == 0)
        device = self.runtime.device
        anchor_mask_t = torch.as_tensor(anchor_mask, dtype=torch.bool, device=device)
        action_mask = anchor_mask_t.reshape(1, 1, -1).expand(1, self.chunk_actions, -1).contiguous()
        seed = derive_diffusion_seed(self.diffusion_seed_base, self.eval_seed, self._episode_index, c)
        generator = torch.Generator(device=device).manual_seed(seed)
        (action_norm, predicted_latent), timings["generate_s"] = timed(
            session.generate,
            observation_latent,
            sparse_latents,
            anchor_state=anchor_norm.to(device),
            anchor_state_mask=anchor_mask_t,
            action_mask=action_mask,
            generator=generator,
            gripper_bounds=self.gripper_bounds,
        )
        mask_cpu = action_mask[0].detach().cpu()
        action_raw = model_action_to_absolute(
            action_norm[0].detach().to(device="cpu", dtype=torch.float32), mask_cpu, self.normalization, anchor_raw, anchor_mask
        )
        executed_raw = executed_rows(action_raw, mask_cpu)
        actions = native_ee_actions_from_action80(executed_raw, self.world_from_base)
        if len(actions) != self.chunk_actions:
            raise RuntimeError(f"{TAG} produced {len(actions)} actions for a chunk of {self.chunk_actions}")

        self._pending = {
            "decision": c,
            "t": t,
            "predicted_latent": predicted_latent.detach() if self.log_world_model_error else None,
            "sparse_stream": sparse_stream if (self.log_world_model_error and not session.reads_dense) else None,
            "executed_raw": executed_raw,
            "mask_cpu": mask_cpu,
        }
        self._reached = {}
        self._tau = 0
        total = time.perf_counter() - start
        detail = " ".join(f"{k}={v:.2f}" for k, v in timings.items())
        memory = []
        if session.reads_dense:
            memory.append(f"dense=L_0..L_{session.dense_count - 1}")
        if session.reads_sparse:
            memory.append(f"sparse_ticks={ticks}")
        print(
            f"{TAG} ep {self._episode_index} decision {c} t={t}: {' '.join(memory)} form={session.deploy_form} {detail} "
            f"total_s={total:.2f}"
            + ("" if wm_error is None else f" world_model_rel_rmse(decision {c - 1})={wm_error:.4f}")
            + ("" if gap is None else f" cmd_vs_reached(decision {c - 1}): {gap}"),
            flush=True,
        )
        return [
            {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in action.items()} for action in actions
        ]

    def get_action_batch(self, env_idx_list=None) -> list[list[dict[str, np.ndarray]]]:
        if env_idx_list is not None:
            ids = np.asarray(env_idx_list).reshape(-1).tolist() if not isinstance(env_idx_list, (int, np.integer)) else [int(env_idx_list)]
            if len(ids) != 1:
                raise ValueError("SANA_WAM_HM serves one environment (eval_batch: false)")
        return [self.get_action()]

    def reset(self) -> None:
        """Drop the episode's memory and canvases; idempotent (RoboDojo sends two resets per episode)."""

        self._session = None
        self._episode_open = False
        self._latest_obs = None
        self._tau = None
        self._tick = 0
        self._canvases = []
        self._reached = {}
        self._dense_stream = None
        self._pending = None
        self._instruction = None

    def prepare_case(self, case_meta=None) -> None:
        return None

    def on_trial_end(self, result=None) -> None:
        return None
