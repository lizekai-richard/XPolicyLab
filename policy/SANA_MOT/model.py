"""XPolicyLab adapter of the SANA MoT dual-expert world-action policy (RoboDojo ARX-X5, joint or EE space).

The policy runs in-process: ``Model.__init__`` builds a :class:`MoTInferenceSession` from the vendored, Sana-free
package ``sana_mot_min`` (the bidirectional MoT policy mirror in every video layout -- the 2x2 multiview strip with one
prompt group per view, the OpenWAM head-over-wrists canvas and the sana_pixel 2x2 pixel canvas -- and every checkpoint
text layout, detected from the weights: separate caption embedders, the canvas modes' ONE shared caption embedder, the
606e48dd9 legacy) on top of the sibling SANA_WAM adapter's runtime (Flow-Euler sampler, LTX-2.3 causal VAE encoder,
Gemma-2-2B text conditioning, Robot80 normalization, the RoboDojo codecs and the ARX-X5 EEF kinematics), and every
``get_action`` call performs one diffusion inference on the stored observation. The websocket ``PolicyServer`` decodes
camera colors before ``update_obs``; this module never decodes images and never swaps channels (the checkpoint was
trained on RGB frames).

Contract: ``update_obs`` stores the observation, ``get_action`` returns the predicted chunk -- one target per
source-frame transition (24 for the 25-frame tier, 32 for the 33-frame tier), or only its first ``n_action_steps`` --
and ``reset`` clears the observation and advances the episode counter used for diffusion seeding. Joint lines
(``joint_only``) return joint-target dicts (``left_arm_joint_state`` (6,), ``left_ee_joint_state`` (1,),
``right_arm_joint_state`` (6,), ``right_ee_joint_state`` (1,)) conditioned on the measured joint state. The EEF-only
``robot_base_eef`` line (2026-09-20 contract: EEF pose + grippers, joints neither fed nor supervised) is served with
``action_type: ee`` -- ``{left,right}_ee_pose`` (link6 in the env-relative world frame) solved to joints by the
evaluator's IK -- with its EEF state slots derived from the measured joints by URDF forward kinematics exactly as the
corpus did (``sana_wam_min/eef.py``).
"""

from __future__ import annotations

import hashlib
import os
import sys
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
# The vendored package lives next to this file; inserting POLICY_DIR lets tools and tests import it as plain
# ``sana_mot_min`` regardless of how XPolicyLab itself was put on sys.path. Its own import puts the sibling
# policy/SANA_WAM on sys.path for the shared ``sana_wam_min`` runtime.
if str(POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(POLICY_DIR))

from sana_mot_min.config import declared_multiview, resolve_video_layout  # noqa: E402
from sana_mot_min.mot_model.model import (  # noqa: E402
    CONTEXT_EMBEDDER,
    LEGACY_ACTION_MLP,
    SHARED_CAPTION_EMBEDDER,
    VIDEO_LAYOUTS,
)
from sana_mot_min.session import MoTInferenceSession, find_train_config, load_train_config  # noqa: E402
from sana_wam_min.actions import apply_joint_limits  # noqa: E402
from sana_wam_min.robot80 import normalized_gripper_bounds  # noqa: E402
from sana_wam_min.config import (  # noqa: E402
    ROBOT_BASE_EEF_LAYOUTS,
    ROBOT_BASE_EEF_ONLY,
    action_mode_from_train_config,
    robot_base_eef_layout_from_train_config,
)
from sana_wam_min.eef import (  # noqa: E402
    ArxX5Kinematics,
    eef_state_slot_mask,
    fill_eef_state_slots,
    native_ee_actions_from_action80,
    reconstruct_absolute_eef,
    root_transforms,
)
from sana_wam_min.robodojo_io import (  # noqa: E402
    ROBODOJO_ARM_DIM,
    ROBODOJO_FROZEN_IMAGE_HEIGHT,
    ROBODOJO_FROZEN_IMAGE_WIDTH,
    ROBODOJO_NATIVE_ACTION_KEYS,
    ROBODOJO_VIEW_ORDER,
    ROBOT80_JOINT_SLOTS_12,
    instruction_from_obs,
    state80_from_obs,
    upstream_actions_from_action80,
)

CHECKPOINT_EXPLICIT_KEYS = ("checkpoint_dir", "checkpoint_path", "ckpt_dir", "model_dir")
# Robot80 slots kept by the diagnostic trajectory dump (same layout as policy/SANA_WAM): 12 arm joints + 2 gripper slots
ROBOT80_GRIPPER_SLOTS_2 = (16, 45)  # left / right gripper closedness slots of the Robot80 row (as in policy/SANA_WAM)
TRAJ_SLOTS = tuple(ROBOT80_JOINT_SLOTS_12) + tuple(ROBOT80_GRIPPER_SLOTS_2)
ACTION_TYPES = ("joint", "ee")
ROBODOJO_EE_ACTION_KEYS = ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state")
# video_layout: auto (default) takes the layout the checkpoint was trained with; an explicit value must agree with it.
VIDEO_LAYOUT_CHOICES = ("auto",) + tuple(VIDEO_LAYOUTS)
JOINT_LIMIT_MODES = ("clip", "reject", "none")
SEED_DOMAIN = "sana_mot_xpolicylab.diffusion_seed.v1"
MAX_JSON_SAFE_INTEGER = 2**53 - 1
DEFAULT_INSTRUCTION = "follow the instruction"


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


def derive_diffusion_seed(seed_base: int, eval_seed: int, episode_index: int, chunk_index: int) -> int:
    """Deterministic per-(eval seed, episode, chunk) seed: sha256 of the four integers, 53-bit."""

    preimage = b"\x00".join(
        str(part).encode("utf-8") for part in (SEED_DOMAIN, int(seed_base), int(eval_seed), int(episode_index), int(chunk_index))
    )
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big") & MAX_JSON_SAFE_INTEGER


def _rgb_frame(color: Any, camera: str, strict_image_size: bool) -> np.ndarray:
    """Return a C-contiguous uint8 HxWx3 copy of a decoded color array; channel order and value scale are never touched."""

    frame = np.asarray(color)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{camera}: color must be an HxWx3 array, got shape {frame.shape} dtype {frame.dtype}")
    if frame.dtype != np.uint8:
        raise ValueError(f"{camera}: color must be a uint8 RGB array, got dtype {frame.dtype}; frames are never rescaled")
    expected = (ROBODOJO_FROZEN_IMAGE_HEIGHT, ROBODOJO_FROZEN_IMAGE_WIDTH, 3)
    if strict_image_size and tuple(frame.shape) != expected:
        raise ValueError(f"{camera}: strict_image_size requires {expected}, got {tuple(frame.shape)}")
    return np.array(frame, dtype=np.uint8, copy=True, order="C")


class Model(ModelTemplate):
    """SANA_MOT policy: one in-process diffusion inference per ``get_action``."""

    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        cfg = self.model_cfg

        action_type = str(cfg.get("action_type") or "joint").strip().lower()
        if action_type not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES} for SANA_MOT, got {action_type!r}")
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
                "SANA_MOT RoboDojo checkpoints support dual ARX-X5 arms only (arm_dim [6, 6], ee_dim [1, 1]); "
                f"env_cfg_type={self.env_cfg_type!r} resolves to {self.robot_action_dim_info!r}"
            )

        self.ckpt_dir = self._resolve_ckpt_dir(cfg)
        self.train_config = load_train_config(str(find_train_config(self.ckpt_dir)))
        self.action_mode = self._check_action_mode(self.train_config)
        self.include_eef = self.action_mode == "robot_base_eef"
        # robot_base_eef: full (32 slots, before 2026-09-20) | eef_only (EEF + grippers); resolved from the yaml here and
        # again with the normalization artifact's scheme stamps once the session has loaded it
        self.requested_robot_base_eef_layout = self._requested_robot_base_eef_layout(cfg)
        if not self.include_eef and self.requested_robot_base_eef_layout != "auto":
            raise ValueError("robot_base_eef_layout applies to robot_base_eef checkpoints only")
        self.robot_base_eef_layout = (
            (
                robot_base_eef_layout_from_train_config(self.train_config)
                if self.requested_robot_base_eef_layout == "auto"
                else self.requested_robot_base_eef_layout
            )
            if self.include_eef
            else None
        )
        if self.action_type == "ee" and not self.include_eef:
            raise ValueError("action_type 'ee' needs a robot_base_eef checkpoint (EEF pose slots supervised)")
        self._check_eef_only_action_type()
        self.eef_pose_check = _is_true(cfg.get("eef_pose_check", True))
        self.kinematics = ArxX5Kinematics(cfg.get("urdf_path") or None) if self.include_eef else None
        self.world_from_base = root_transforms(cfg.get("robot_root_poses") or None)
        # Fail before the weights are read when the yaml alone decides the layout (it always does for real yamls: they
        # carry data.type); the resolved session layout is checked again below.
        self.requested_video_layout = self._requested_video_layout(cfg)
        yaml_layout = self._yaml_video_layout(self.train_config)
        if yaml_layout is not None:
            self._check_video_layout(self.requested_video_layout, yaml_layout, self.train_config)
        weight_dtype = str(cfg.get("weight_dtype") or "bfloat16")
        if weight_dtype not in ("bfloat16", "bf16"):
            raise ValueError(f"only bfloat16 weights are supported by SANA_MOT, got weight_dtype={weight_dtype!r}")
        self.device = torch.device(str(cfg.get("device") or "cuda"))
        self.strict_image_size = _is_true(cfg.get("strict_image_size", False))
        self.diffusion_seed_base = int(cfg.get("diffusion_seed_base") or 20260802)
        self.eval_seed = _optional_int(cfg.get("seed")) or 0
        self.default_instruction = str(cfg.get("default_instruction") or DEFAULT_INSTRUCTION)
        self.n_action_steps = self._resolve_n_action_steps(cfg)
        self.action_chunk_size: Optional[int] = None
        self._logged_instruction: Optional[str] = None
        self._n_action_steps_warned = False

        self.joint_limit_mode = str(cfg.get("joint_limit_mode") or "clip").lower()
        if self.joint_limit_mode not in JOINT_LIMIT_MODES:
            raise ValueError(f"joint_limit_mode must be one of {JOINT_LIMIT_MODES}, got {self.joint_limit_mode!r}")
        self.joint_lower, self.joint_upper = self._resolve_joint_limits(cfg)

        self.session = MoTInferenceSession.from_paths(
            checkpoint_dir=str(self.ckpt_dir),
            text_encoder_path=str(cfg.get("text_encoder_path") or os.environ.get("SANA_WAM_TEXT_ENCODER_PATH") or "google/gemma-2-2b-it"),
            vae_path=str(cfg.get("vae_path") or os.environ.get("SANA_WAM_VAE_PATH") or "Efficient-Large-Model/LTX-2.3-Diffusers"),
            normalization_path=cfg.get("normalization_path") or None,
            device=self.device,
            steps=_optional_int(cfg.get("sampling_steps")),
            cfg_scale=_optional_float(cfg.get("cfg_scale")),
            flow_shift=_optional_float(cfg.get("flow_shift")),
            # the action stream's own sampling shift (null = the checkpoint's inference_action_flow_shift / action_flow_shift
            # when declared, else flow_shift for both streams)
            action_flow_shift=_optional_float(cfg.get("action_flow_shift")),
            # how each view reaches its bucket: null = the yaml's robot_sft.view_resize, else stretch (the only SFT resize
            # since rwm/mot 034e55dca); crop = the legacy resize of every checkpoint trained before it
            view_resize=cfg.get("view_resize"),
            expected_normalization_sha256=cfg.get("normalization_sha256") or None,
            video_cfg_scale=_optional_float(cfg.get("video_cfg_scale")),
            action_cfg_scale=_optional_float(cfg.get("action_cfg_scale")),
            # contracts the yaml cannot always name (auto = resolved from the checkpoint, see deploy.yml)
            rope_mode=cfg.get("rope_mode"),
            text_groups=cfg.get("text_groups"),
            canvas_prompt=cfg.get("canvas_prompt"),
            robot_base_eef_layout=(
                None if self.requested_robot_base_eef_layout == "auto" else self.requested_robot_base_eef_layout
            ),
        )
        self.model = self.session.model
        if self.include_eef:
            self.robot_base_eef_layout = getattr(self.session, "robot_base_eef_layout", None) or self.robot_base_eef_layout
            self._check_eef_only_action_type()
        self.video_layout = str(getattr(self.session, "video_layout", None))
        self._check_video_layout(self.requested_video_layout, self.video_layout, self.train_config)
        self._log_normalization_contract()

        self._obs: dict[int, dict] = {}
        self._order: list[int] = []
        self._episode_index = 0
        self._chunk_index = 0
        self._size_warned = False
        # Diagnostic trajectory dump (trajectory_dump_dir, null = off), same format as policy/SANA_WAM: per control tick
        # the measured Robot80 joints + grippers of every update_obs, per inference the conditioning row and the absolute
        # chunk returned -> <dir>/ep<N>.npz (rewritten after every chunk, flushed at reset / trial end)
        self.trajectory_dump_dir = self._resolve_trajectory_dump_dir(cfg)
        self._traj_obs: list[tuple[int, int, int, np.ndarray]] = []
        self._traj_chunks: list[dict[str, Any]] = []
        self._traj_step = 0
        print(
            f"[SANA_MOT] ready: ckpt={self.ckpt_dir} steps={self.session.steps} cfg_scale={self.session.cfg_scale} "
            f"video_cfg_scale={self.session.video_cfg_scale} action_cfg_scale={self.session.action_cfg_scale} "
            f"flow_shift={self.session.flow_shift} action_flow_shift={getattr(self.session, 'action_flow_shift', None) or 'shared'} "
            f"device={self.device} joint_limit_mode={self.joint_limit_mode} "
            f"n_action_steps={'all' if self.n_action_steps is None else self.n_action_steps} "
            f"action_type={self.action_type} joint_target_mode={self.session.joint_target_mode} "
            f"video_layout={self.video_layout} (requested {self.requested_video_layout}) "
            f"view_resize={getattr(self.session, 'view_resize', 'stretch')} ({getattr(self.session, 'view_resize_source', 'default')}) "
            f"video_fps={self.session.video_fps} video_frame_stride={self.session.video_frame_stride} "
            f"context_layout={(getattr(self.session, 'load_report', None) or {}).get('context_layout')} "
            f"state_as_context={getattr(getattr(self.model, 'action_dit', None), 'state_as_context', None)}"
            f" multiview={declared_multiview(self.train_config)}"
            f" rope={getattr(self.model, 'rope', None)} ({getattr(self.session, 'rope_contract', 'n/a')})"
            f" text_groups={getattr(self.session, 'text_groups', None)} canvas_prompt={getattr(self.session, 'canvas_prompt', None)}"
            f" action_mode={self.action_mode}"
            + (f" robot_base_eef_layout={self.robot_base_eef_layout}" if self.include_eef else "")
            + (f" trajectory_dump_dir={self.trajectory_dump_dir}" if self.trajectory_dump_dir is not None else "")
        )
        sentence = getattr(self.session, "action_mode_text", None)
        if sentence is not None:
            print(f"[SANA_MOT] prompt Action Mode sentence (from the training yaml): {sentence!r}")
        contract = getattr(self.session, "text_contract", None)
        if contract is not None:
            print(f"[SANA_MOT] text contract: {contract}; canvas Observation View: {getattr(self.session, 'canvas_view_text', None)!r}")

    # -- configuration --------------------------------------------------------

    @staticmethod
    def _check_action_mode(train_cfg: Mapping[str, Any]) -> str:
        """The line's single action mode (``data.extra.action_mode_sample_ratio``): joint_only, or the EEF-only
        robot_base_eef line (rwm/mot 2434e9199, e.g. ``sft_robodojo_mot_eefabs_f33fps8_sana_pixel``); qwen_canonical and
        mixed ratios are refused."""

        return action_mode_from_train_config(dict(train_cfg))

    @staticmethod
    def _requested_robot_base_eef_layout(cfg: Mapping[str, Any]) -> str:
        """``robot_base_eef_layout``: ``auto`` (default) | ``full`` | ``eef_only`` (robot_base_eef checkpoints only)."""

        raw = cfg.get("robot_base_eef_layout")
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "auto", "none", "null"}):
            return "auto"
        value = str(raw).strip().lower()
        if value not in ROBOT_BASE_EEF_LAYOUTS:
            raise ValueError(f"robot_base_eef_layout must be one of ('auto',) + {ROBOT_BASE_EEF_LAYOUTS}, got {raw!r}")
        return value

    def _check_eef_only_action_type(self) -> None:
        """An EEF-only checkpoint predicts no joint slot, so it can only drive the evaluator through EE poses."""

        if self.robot_base_eef_layout == ROBOT_BASE_EEF_ONLY and self.action_type != "ee":
            raise ValueError(
                "this robot_base_eef checkpoint is EEF-only (EEF pose + grippers, no joint slot supervised or fed), so it "
                "predicts no joint targets; serve it with action_type 'ee'"
            )

    @staticmethod
    def _requested_video_layout(cfg: Mapping[str, Any]) -> str:
        """``video_layout``: ``auto`` (default, also null / empty) or one of :data:`VIDEO_LAYOUTS`."""

        raw = cfg.get("video_layout")
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}):
            return "auto"
        requested = str(raw).strip().lower()
        if requested not in VIDEO_LAYOUT_CHOICES:
            raise ValueError(f"video_layout must be one of {VIDEO_LAYOUT_CHOICES}, got {raw!r}")
        return requested

    @staticmethod
    def _yaml_video_layout(train_cfg: Mapping[str, Any]) -> Optional[str]:
        """The layout the training yaml decides on its own, or None when it also takes the checkpoint's text layout
        (a yaml with neither ``model.extra.video_layout`` nor ``data.type``)."""

        layouts = {resolve_video_layout(dict(train_cfg), layout) for layout in (CONTEXT_EMBEDDER, SHARED_CAPTION_EMBEDDER, LEGACY_ACTION_MLP)}
        return layouts.pop() if len(layouts) == 1 else None

    @staticmethod
    def _check_video_layout(requested: str, resolved: str, train_cfg: Mapping[str, Any]) -> None:
        """An explicit ``video_layout`` only asserts the checkpoint's layout: the layout is a training-time contract,
        so the key documents the operator's intent and can never re-route a checkpoint through the other front-end."""

        if requested == "auto" or requested == resolved:
            return
        extra = (train_cfg.get("model") or {}).get("extra") or {}
        raise ValueError(
            f"video_layout {requested!r} does not match the checkpoint, which resolves to {resolved!r} "
            f"(model.extra.video_layout {extra.get('video_layout')!r}, data.type {(train_cfg.get('data') or {}).get('type')!r})"
        )

    def _log_normalization_contract(self) -> None:
        """One start-up line naming the normalization artifact actually loaded and a check of its slot layout."""

        norm = getattr(self.session, "normalization", None)
        if norm is None or getattr(norm, "action_normalization_mask80", None) is None:
            return
        active = [int(i) for i in np.flatnonzero(np.asarray(norm.action_normalization_mask80, dtype=bool))]
        state_active = [int(i) for i in np.flatnonzero(np.asarray(norm.state_normalization_mask80, dtype=bool))]
        print(
            f"[SANA_MOT] normalization: path={norm.source_path} sha256={norm.sha256} action_mode={norm.action_mode} "
            f"action_representation={norm.action_representation} joint_target_mode={norm.joint_target_mode} "
            f"num_frames={norm.num_frames} model_fps={norm.model_fps} action_slots_normalized={active} state_slots_normalized={state_active}"
        )
        joint6 = list(range(0, 6)) + list(range(29, 35))
        problems = []
        missing = [i for i in joint6 if i not in active]
        if missing:
            problems.append(f"joint slots without action statistics: {missing}")
        seventh = [i for i in (6, 35) if i in active or i in state_active]
        if seventh:
            problems.append(f"7th-joint slots {seventh} carry statistics (ARX-X5 arms have 6 joints)")
        # 2026-09-20 forced scheme: grippers by statistics (center 0.5, scale 0.5), Rot6D by the fixed [-1, 1] range; the
        # artifact stamps it and the generic masked affine map applies it. Before: grippers identity (mask off).
        gripper_scheme = getattr(norm, "gripper_normalization", None)
        grippers = [i for i in (16, 45) if i in active]
        if grippers and gripper_scheme != "statistics":
            problems.append(f"gripper slots {grippers} are normalized but the artifact does not declare gripper_normalization: statistics")
        if not grippers and gripper_scheme == "statistics":
            problems.append("the artifact declares gripper_normalization: statistics but slots 16 / 45 are not masked")
        scheme = (
            f"grippers 16 / 45 by statistics, rot6d {getattr(norm, 'rotation_normalization', None)} (2026-09-20 scheme)"
            if gripper_scheme == "statistics"
            else "grippers 16 / 45 identity (pre-2026-09-20 scheme)"
        )
        if problems:
            print("[SANA_MOT] WARNING normalization layout does not match the ARX-X5 contract: " + "; ".join(problems))
        else:
            print(f"[SANA_MOT] normalization layout OK: 6 joints per arm (slots 0-5 / 29-34), {scheme}, slots 6 / 35 unused")
        (left_low, left_high), (right_low, right_high) = normalized_gripper_bounds(norm)
        print(
            f"[SANA_MOT] gripper clamp in the model domain = raw closedness [0, 1]: left [{left_low:.3f}, {left_high:.3f}] "
            f"right [{right_low:.3f}, {right_high:.3f}]"
        )

    @staticmethod
    def _resolve_ckpt_dir(cfg: Mapping[str, Any]) -> Path:
        return resolve_checkpoint_root(
            dict(cfg), CHECKPOINTS_DIR, policy_dir=POLICY_DIR, explicit_keys=CHECKPOINT_EXPLICIT_KEYS, must_exist=True
        )

    @staticmethod
    def _resolve_n_action_steps(cfg: Mapping[str, Any]) -> Optional[int]:
        """``n_action_steps``: how many leading targets of each predicted chunk are returned; None = the whole chunk."""

        raw = cfg.get("n_action_steps")
        if raw is None or isinstance(raw, bool):
            if raw is None:
                return None
            raise ValueError(f"n_action_steps must be a positive integer, null or 'all', got {raw!r}")
        if isinstance(raw, str):
            text = raw.strip().lower()
            if text in {"", "none", "null", "all"}:
                return None
            raw = text
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"n_action_steps must be a positive integer, null or 'all', got {raw!r}") from exc
        if not number.is_integer() or number < 0:
            raise ValueError(f"n_action_steps must be a positive integer, null or 'all', got {raw!r}")
        value = int(number)
        return None if value == 0 else value

    def _resolve_joint_limits(self, cfg: Mapping[str, Any]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Per-slot limits for the 12 joint slots from ``joint_lower`` / ``joint_upper`` (12 values each), if given."""

        lower, upper = cfg.get("joint_lower"), cfg.get("joint_upper")
        if self.joint_limit_mode == "none":
            return None, None
        if lower is None or upper is None:
            warnings.warn(
                "deploy.yml has no joint_lower/joint_upper; joint-limit gating is skipped "
                f"(joint_limit_mode={self.joint_limit_mode!r} has no effect)",
                stacklevel=2,
            )
            return None, None
        lower_arr = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper_arr = np.asarray(upper, dtype=np.float64).reshape(-1)
        n_slots = len(ROBOT80_JOINT_SLOTS_12)
        if lower_arr.shape != (n_slots,) or upper_arr.shape != (n_slots,):
            raise ValueError(f"joint_lower/joint_upper must each hold {n_slots} values (left 6 then right 6)")
        return lower_arr, upper_arr

    # -- observations ---------------------------------------------------------

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list) -> None:
        if isinstance(obs_list, Mapping):
            obs_list = [obs_list]
        if not obs_list:
            raise ValueError("update_obs_batch received an empty list")
        self._obs, self._order = {}, []
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", index))
            self._obs[env_idx] = obs
            self._order.append(env_idx)
            self._traj_record_obs(env_idx, obs)
        if self.trajectory_dump_dir is not None:
            self._traj_step += 1  # one control tick per update_obs call

    def _frames_from_obs(self, obs: Mapping[str, Any]) -> list[np.ndarray]:
        vision = obs["vision"]
        frames = [_rgb_frame(vision[cam]["color"], cam, self.strict_image_size) for cam in ROBODOJO_VIEW_ORDER]
        expected = (ROBODOJO_FROZEN_IMAGE_HEIGHT, ROBODOJO_FROZEN_IMAGE_WIDTH, 3)
        if not self._size_warned and any(tuple(f.shape) != expected for f in frames):
            target = {
                "openwam_canvas": "stretched to its OpenWAM canvas slot (head 256x320, wrists 128x160)",
                "sana_pixel_canvas": "resized/cropped to 320x480 and halved into its sana_pixel quadrant",
            }.get(self.video_layout, "resized/cropped to the trained 256x320 bucket")
            warnings.warn(
                f"camera frames are not {expected} (got {[tuple(f.shape) for f in frames]}); each is {target}",
                stacklevel=2,
            )
            self._size_warned = True
        return frames

    def _instruction_from_obs(self, obs: Mapping[str, Any]) -> str:
        try:
            text = instruction_from_obs(obs)
        except (KeyError, IndexError, TypeError):
            return self.default_instruction
        text = text.strip()
        return text if text else self.default_instruction

    # -- actions ----------------------------------------------------------------

    def _predict_one(self, obs: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
        frames = self._frames_from_obs(obs)
        state80_raw, state_mask80 = state80_from_obs(obs["state"])
        if self.include_eef:
            # the corpus pairs each arm's joints with FK(those joints): derive the EEF slots from the measured joints
            state80_raw, state_mask80 = fill_eef_state_slots(state80_raw, eef_state_slot_mask(True), self.kinematics)
            if self.robot_base_eef_layout == ROBOT_BASE_EEF_ONLY:
                # EEF-only: the state token and the action mask are the 20 EEF-pose + gripper slots
                state_mask80 = np.array(state_mask80, dtype=bool, copy=True)
                state_mask80[list(ROBOT80_JOINT_SLOTS_12)] = False
        instruction = self._instruction_from_obs(obs)
        if instruction != self._logged_instruction:
            print(f"[SANA_MOT] episode {self._episode_index} instruction: {instruction!r}", flush=True)
            self._logged_instruction = instruction
        seed = derive_diffusion_seed(self.diffusion_seed_base, self.eval_seed, self._episode_index, self._chunk_index)
        generator = torch.Generator(device=self.session.device).manual_seed(seed)
        result = self.session.predict(frames, state80_raw, state_mask80, instruction, generator)
        action80 = result.action80_raw_absolute
        if self.include_eef and getattr(self.session, "eef_target_mode", "anchor_delta") == "anchor_delta":
            # anchor_delta EEF slots are still anchor-relative; absolute EEF lines predict the base-frame pose itself
            action80 = reconstruct_absolute_eef(action80, result.action_mask, state80_raw, state_mask80)
        if self.joint_lower is not None and self.joint_limit_mode in ("clip", "reject") and self.action_type == "joint":
            action80, clipped = apply_joint_limits(
                action80, ROBOT80_JOINT_SLOTS_12, self.joint_lower, self.joint_upper, mode=self.joint_limit_mode
            )
            if clipped:
                warnings.warn(f"joint targets clipped to configured limits on Robot80 slots {clipped}", stacklevel=2)
        self._chunk_index += 1
        arm_dims, ee_dims = self.robot_action_dim_info["arm_dim"], self.robot_action_dim_info["ee_dim"]
        if self.action_type == "ee":
            # absolute base-frame E poses -> link6 poses in the env-relative world frame; the evaluator solves IK per tick
            actions = native_ee_actions_from_action80(action80, self.world_from_base)
            keys = ROBODOJO_EE_ACTION_KEYS
            for action in actions:
                assert action["left_ee_pose"].shape == (7,) and action["right_ee_pose"].shape == (7,)
                assert action["left_ee_joint_state"].shape == (int(ee_dims[0]),)
                assert action["right_ee_joint_state"].shape == (int(ee_dims[1]),)
        else:
            actions = upstream_actions_from_action80(action80)
            keys = ROBODOJO_NATIVE_ACTION_KEYS
            for action in actions:
                assert action["left_arm_joint_state"].shape == (int(arm_dims[0]),)
                assert action["right_arm_joint_state"].shape == (int(arm_dims[1]),)
                assert action["left_ee_joint_state"].shape == (int(ee_dims[0]),)
                assert action["right_ee_joint_state"].shape == (int(ee_dims[1]),)
        actions = self._select_actions_to_execute(actions)
        # absolute joint targets: the conditioning row is the measured state, no anchor is added
        self._traj_record_chunk(int(obs.get("env_idx", 0)), state80_raw, state80_raw, action80, len(actions))
        return [{k: np.ascontiguousarray(a[k], dtype=np.float32) for k in keys} for a in actions]

    def _select_actions_to_execute(self, actions: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        """Keep the first ``n_action_steps`` targets of a predicted chunk (all of them when unset or n >= chunk)."""

        chunk = len(actions)
        if self.action_chunk_size is None:
            self.action_chunk_size = chunk
            if self.n_action_steps is not None:
                print(
                    f"[SANA_MOT] n_action_steps={self.n_action_steps}: executing the first "
                    f"{min(self.n_action_steps, chunk)} of {chunk} predicted targets per inference, then replanning"
                )
        n = self.n_action_steps
        if n is None:
            return actions
        if n > chunk and not self._n_action_steps_warned:
            warnings.warn(
                f"n_action_steps={n} exceeds the predicted chunk of {chunk} targets; the whole chunk is executed",
                stacklevel=2,
            )
            self._n_action_steps_warned = True
        return actions[:n]

    def get_action(self) -> list[dict[str, np.ndarray]]:
        env_idx = self._order[0] if self._order else 0
        return self.get_action_batch([env_idx])[0]

    def get_action_batch(self, env_idx_list=None) -> list[list[dict[str, np.ndarray]]]:
        if env_idx_list is None:
            env_idx_list = list(self._order)
        elif isinstance(env_idx_list, np.ndarray):
            env_idx_list = env_idx_list.reshape(-1).tolist()
        elif isinstance(env_idx_list, (int, np.integer)):
            env_idx_list = [int(env_idx_list)]
        results = []
        for env_idx in env_idx_list:
            obs = self._obs.get(int(env_idx))
            if obs is None:
                raise ValueError(f"No stored observation for env_idx {env_idx}; call update_obs_batch first")
            results.append(self._predict_one(obs))
        return results

    def reset(self) -> None:
        self._traj_flush()
        self._traj_obs, self._traj_chunks, self._traj_step = [], [], 0
        self._obs, self._order = {}, []
        self._chunk_index = 0
        self._logged_instruction = None
        self._episode_index += 1

    def prepare_case(self, case_meta=None) -> None:
        return None

    def on_trial_end(self, result=None) -> None:
        self._traj_flush()
        return None

    # -- diagnostic trajectory dump (mirror of policy/SANA_WAM/model.py) ------------

    @staticmethod
    def _resolve_trajectory_dump_dir(cfg: Mapping[str, Any]) -> Optional[Path]:
        """``trajectory_dump_dir``: null/empty = off; otherwise a directory (created) for per-episode ``ep<N>.npz`` dumps."""

        raw = cfg.get("trajectory_dump_dir")
        if raw is None or raw is False or str(raw).strip().lower() in ("", "null", "none", "false", "off"):
            return None
        path = Path(str(raw)).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _traj_record_obs(self, env_idx: int, obs: Mapping[str, Any]) -> None:
        if self.trajectory_dump_dir is None:
            return
        state80, _ = state80_from_obs(obs["state"])
        row = np.asarray(state80, dtype=np.float32)[list(TRAJ_SLOTS)]
        self._traj_obs.append((int(self._episode_index), int(self._traj_step), int(env_idx), row))

    def _traj_record_chunk(self, env_idx: int, state80_measured: Any, state80_raw: Any, action80: Any, executed: int) -> None:
        if self.trajectory_dump_dir is None:
            return
        rows = torch.as_tensor(action80).detach().to(device="cpu", dtype=torch.float32).numpy()
        sl = list(TRAJ_SLOTS)
        self._traj_chunks.append(
            dict(
                episode=int(self._episode_index),
                chunk=int(self._chunk_index - 1),  # _chunk_index was advanced by the caller
                step=int(max(self._traj_step - 1, 0)),  # the update_obs tick this chunk was predicted from
                env=int(env_idx),
                measured=np.asarray(state80_measured, dtype=np.float32)[sl],
                anchor=np.asarray(state80_raw, dtype=np.float32)[sl],
                actions=np.ascontiguousarray(rows[:, sl], dtype=np.float32),
                executed=int(executed),
            )
        )
        self._traj_flush()

    def _traj_flush(self) -> None:
        if self.trajectory_dump_dir is None or (not self._traj_obs and not self._traj_chunks):
            return
        obs, ch = self._traj_obs, self._traj_chunks
        n_slots = len(TRAJ_SLOTS)
        if ch:
            width = max(c["actions"].shape[0] for c in ch)
            acts = np.full((len(ch), width, n_slots), np.nan, dtype=np.float32)
            for i, c in enumerate(ch):
                acts[i, : c["actions"].shape[0]] = c["actions"]
        else:
            acts = np.zeros((0, 0, n_slots), dtype=np.float32)
        arrays = dict(
            slots=np.asarray(TRAJ_SLOTS, dtype=np.int64),
            obs_episode=np.asarray([o[0] for o in obs], dtype=np.int64),
            obs_step=np.asarray([o[1] for o in obs], dtype=np.int64),
            obs_env=np.asarray([o[2] for o in obs], dtype=np.int64),
            obs_state=np.stack([o[3] for o in obs]) if obs else np.zeros((0, n_slots), dtype=np.float32),
            chunk_episode=np.asarray([c["episode"] for c in ch], dtype=np.int64),
            chunk_index=np.asarray([c["chunk"] for c in ch], dtype=np.int64),
            chunk_step=np.asarray([c["step"] for c in ch], dtype=np.int64),
            chunk_env=np.asarray([c["env"] for c in ch], dtype=np.int64),
            chunk_executed=np.asarray([c["executed"] for c in ch], dtype=np.int64),
            chunk_measured=np.stack([c["measured"] for c in ch]) if ch else np.zeros((0, n_slots), dtype=np.float32),
            chunk_anchor=np.stack([c["anchor"] for c in ch]) if ch else np.zeros((0, n_slots), dtype=np.float32),
            chunk_actions=acts,
        )
        out = self.trajectory_dump_dir / f"ep{int(self._episode_index):04d}.npz"
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
