"""XPolicyLab adapter of the SANA unified world-action policy (RoboDojo ARX-X5, joint space).

The policy runs in-process: ``Model.__init__`` builds a :class:`PolicyInferenceSession`
from the vendored, Sana-free package ``sana_wam_min`` (policy transformer, Flow-Euler
sampler, LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning, Robot80 normalization
and the RoboDojo codecs) and every ``get_action`` call performs one 50-step diffusion
inference on the stored observation. The websocket ``PolicyServer`` decodes camera colors
before ``update_obs``; this module never decodes images and never swaps channels (the
checkpoint was trained on RGB frames).

Contract: ``update_obs`` stores the observation, ``get_action`` returns the predicted chunk of
joint-target dicts (``left_arm_joint_state`` (6,), ``left_ee_joint_state`` (1,),
``right_arm_joint_state`` (6,), ``right_ee_joint_state`` (1,); float32) -- 24 per inference for
the 25-frame tier, or only its first ``n_action_steps`` targets when that key is set, so the
environment loop re-observes and asks for a fresh chunk after ``n`` ticks (receding-horizon
replanning) -- and ``reset`` clears the observation and the last-command anchor and advances the
episode counter used for diffusion seeding. ``anchor_source`` picks the joint state the model is
conditioned on and its joint deltas are added to: the evaluator's measured joints (default) or the
last target this adapter returned for the previous chunk (the training corpus records the previous
command as the state, so a measured, lagging anchor is a train/eval mismatch). ``action_type`` is
``joint`` (24 joint-target dicts) or ``ee`` (24 ``left_ee_pose`` / ``right_ee_pose`` dicts: the flange ``link6``
pose in the env-relative world frame, solved to joints by the evaluator's IK; needs a robot_base_eef
checkpoint). Checkpoints of the robot_base_eef line (``state_profile`` auto-detected from the training
yaml) are conditioned on the flange EEF pose too: the adapter derives it from the row's joints by URDF forward
kinematics exactly as the corpus did (``sana_wam_min/eef.py``). ``visual_layout`` (auto-detected from the training
yaml) is ``three_view_strip`` for the 320px lines (each camera 256x320, encoded on its own, packed as a strip) or
``openwam_canvas`` for the ``rwm/openwam`` canvas line (the three cameras stretched into one 384x320 L-shaped RGB
canvas, encoded once, one prompt shared by the video and action tokens; ``sana_wam_min/openwam_canvas.py``). The
robot must resolve to dual six-joint arms with one gripper channel each.
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
# The vendored package lives next to this file; inserting POLICY_DIR lets tools and tests
# import it as plain ``sana_wam_min`` regardless of how XPolicyLab itself was put on sys.path.
if str(POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(POLICY_DIR))

from sana_wam_min.actions import apply_joint_limits  # noqa: E402
from sana_wam_min.eef import (  # noqa: E402
    ArxX5Kinematics,
    eef_state_slot_mask,
    fill_eef_state_slots,
    native_ee_actions_from_action80,
    observed_link6_discrepancy,
    reconstruct_absolute_eef,
    root_transforms,
)
from sana_wam_min.robodojo_io import (  # noqa: E402
    LEFT_GRIPPER_SLOT,
    ROBODOJO_ARM_DIM,
    ROBODOJO_FROZEN_IMAGE_HEIGHT,
    ROBODOJO_FROZEN_IMAGE_WIDTH,
    ROBODOJO_NATIVE_ACTION_KEYS,
    ROBODOJO_VIEW_ORDER,
    RIGHT_GRIPPER_SLOT,
    ROBOT80_JOINT_SLOTS_12,
    instruction_from_obs,
    state80_from_obs,
    upstream_actions_from_action80,
)
from sana_wam_min.session import PolicyInferenceSession, find_train_config  # noqa: E402
from sana_wam_min.config import (  # noqa: E402
    VISUAL_LAYOUT_OPENWAM_CANVAS,
    VISUAL_LAYOUTS,
    load_train_config,
    resolve_visual_layout,
    state_as_cross_attention_from_train_config,
)
from sana_wam_min.openwam_canvas import OPENWAM_CANVAS_HEIGHT, OPENWAM_CANVAS_WIDTH  # noqa: E402

CHECKPOINT_EXPLICIT_KEYS = ("checkpoint_dir", "checkpoint_path", "ckpt_dir", "model_dir")
ACTION_TYPES = ("joint", "ee")
# state_profile: which Robot80 slots the model is conditioned on / supervised. joint_only = 6 joints + gripper per arm
# (the epoch10 / s50k lines); robot_base_eef = those plus the flange EEF position + rot6d slots (the eef SFT line);
# auto reads data.extra.action_mode_sample_ratio of the checkpoint's training yaml ((qwen_canonical, robot_base_eef, joint_only)).
STATE_PROFILES = ("auto", "joint_only", "robot_base_eef")
# visual_layout: auto (default) reads the checkpoint's training yaml; an explicit value must agree with it.
VISUAL_LAYOUT_CHOICES = ("auto",) + tuple(VISUAL_LAYOUTS)
ROBODOJO_EE_ACTION_KEYS = ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state")
EEF_POSE_CHECK_WARN_M = 0.005
EEF_POSE_CHECK_WARN_DEG = 1.0
JOINT_LIMIT_MODES = ("clip", "reject", "none")
# anchor_source: the Robot80 row the model is conditioned on AND its joint deltas are added to (one row in training).
ANCHOR_SOURCES = ("measured", "last_command", "last_command_clamped")
DEFAULT_ANCHOR_CLAMP_RAD = 0.05
ROBOT80_GRIPPER_SLOTS_2 = (LEFT_GRIPPER_SLOT, RIGHT_GRIPPER_SLOT)
SEED_DOMAIN = "sana_wam_xpolicylab.diffusion_seed.v1"
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
    """Return a C-contiguous uint8 HxWx3 copy of a decoded color array; channel order and value scale are never touched.

    Non-uint8 frames are rejected: the value range of a float frame ([0, 1] or [0, 255]) is
    unknowable here and a silent rescale would feed the VAE the wrong pixels.
    """

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
    """SANA_WAM policy: one in-process diffusion inference per ``get_action``."""

    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        cfg = self.model_cfg

        action_type = str(cfg.get("action_type") or "joint").strip().lower()
        if action_type not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES}, got {action_type!r}")
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
                "SANA_WAM RoboDojo checkpoint supports dual ARX-X5 arms only (arm_dim [6, 6], ee_dim [1, 1]); "
                f"env_cfg_type={self.env_cfg_type!r} resolves to {self.robot_action_dim_info!r}"
            )

        self.ckpt_dir = self._resolve_ckpt_dir(cfg)
        self.train_config = load_train_config(str(find_train_config(self.ckpt_dir)))
        self.visual_layout = self._resolve_visual_layout(cfg, self.train_config)
        self.state_as_cross_attention = state_as_cross_attention_from_train_config(self.train_config)
        self.state_profile, self.include_eef = self._resolve_state_profile(cfg, self.train_config)
        if self.action_type == "ee" and not self.include_eef:
            raise ValueError(
                "action_type 'ee' needs a robot_base_eef checkpoint (EEF position/rotation slots supervised); "
                f"this checkpoint's state profile is {self.state_profile!r}"
            )
        self.eef_pose_check = _is_true(cfg.get("eef_pose_check", True))
        self.kinematics = ArxX5Kinematics(cfg.get("urdf_path") or None) if self.include_eef else None
        self.world_from_base = root_transforms(cfg.get("robot_root_poses") or None)
        weight_dtype = str(cfg.get("weight_dtype") or "bfloat16")
        if weight_dtype not in ("bfloat16", "bf16"):
            raise ValueError(f"only bfloat16 weights are validated for SANA_WAM, got weight_dtype={weight_dtype!r}")
        self.device = torch.device(str(cfg.get("device") or "cuda"))
        self.strict_image_size = _is_true(cfg.get("strict_image_size", False))
        self.diffusion_seed_base = int(cfg.get("diffusion_seed_base") or 20260802)
        self.eval_seed = _optional_int(cfg.get("seed")) or 0
        self.default_instruction = str(cfg.get("default_instruction") or DEFAULT_INSTRUCTION)
        self.n_action_steps = self._resolve_n_action_steps(cfg)
        self.action_chunk_size: Optional[int] = None  # length of the predicted chunk, learned from the first inference
        self._logged_instruction: Optional[str] = None  # provenance: the instruction text is printed once per episode
        self._n_action_steps_warned = False
        self.anchor_source, self.anchor_clamp_rad = self._resolve_anchor_source(cfg)
        if self.action_type == "ee" and self.anchor_source != "measured":
            # under ee actions the evaluator executes IK solutions of our EEF targets, so the joints the adapter returned
            # were never the executed command; a last-command anchor would mix executed EEF with unexecuted joints
            raise ValueError("anchor_source must be 'measured' when action_type is 'ee'")
        # per env: the last absolute Robot80 row this adapter RETURNED for the previous chunk (= the last command the
        # evaluator executed); the state input + delta anchor of the next chunk under anchor_source last_command*;
        # cleared by reset() so the first chunk of every episode anchors on the measured state
        self._last_command80: dict[int, np.ndarray] = {}

        self.joint_limit_mode = str(cfg.get("joint_limit_mode") or "clip").lower()
        if self.joint_limit_mode not in JOINT_LIMIT_MODES:
            raise ValueError(f"joint_limit_mode must be one of {JOINT_LIMIT_MODES}, got {self.joint_limit_mode!r}")
        self.joint_lower, self.joint_upper = self._resolve_joint_limits(cfg)

        self.session = PolicyInferenceSession.from_paths(
            checkpoint_dir=str(self.ckpt_dir),
            text_encoder_path=str(cfg.get("text_encoder_path") or os.environ.get("SANA_WAM_TEXT_ENCODER_PATH") or "google/gemma-2-2b-it"),
            vae_path=str(cfg.get("vae_path") or os.environ.get("SANA_WAM_VAE_PATH") or "Efficient-Large-Model/LTX-2.3-Diffusers"),
            normalization_path=cfg.get("normalization_path") or None,
            device=self.device,
            steps=_optional_int(cfg.get("sampling_steps")),
            cfg_scale=_optional_float(cfg.get("cfg_scale")),
            flow_shift=_optional_float(cfg.get("flow_shift")),
            expected_normalization_sha256=cfg.get("normalization_sha256") or None,
            # per-stream guidance: null inherits cfg_scale; action_cfg_scale=1 with cfg_scale>1 guides the video only
            video_cfg_scale=_optional_float(cfg.get("video_cfg_scale")),
            action_cfg_scale=_optional_float(cfg.get("action_cfg_scale")),
        )
        self.model = self.session.model
        session_layout = getattr(self.session, "visual_layout", self.visual_layout)
        if session_layout != self.visual_layout:
            raise RuntimeError(f"session visual layout {session_layout!r} != adapter {self.visual_layout!r}")
        self._log_normalization_contract()

        self._obs: dict[int, dict] = {}
        self._order: list[int] = []
        self._episode_index = 0
        self._chunk_index = 0
        self._size_warned = False
        print(
            f"[SANA_WAM] ready: ckpt={self.ckpt_dir} steps={self.session.steps} cfg_scale={self.session.cfg_scale} "
            f"video_cfg_scale={self.session.video_cfg_scale} action_cfg_scale={self.session.action_cfg_scale} "
            f"flow_shift={self.session.flow_shift} device={self.device} joint_limit_mode={self.joint_limit_mode} "
            f"n_action_steps={'all' if self.n_action_steps is None else self.n_action_steps} "
            f"action_type={self.action_type} state_profile={self.state_profile} include_eef={self.include_eef} "
            f"anchor_source={self.anchor_source}"
            + (f" anchor_clamp_rad={self.anchor_clamp_rad}" if self.anchor_clamp_rad is not None else "")
            + f" visual_layout={self.visual_layout} state_as_cross_attention={self.state_as_cross_attention}"
        )
        prompt_sentence = getattr(self.session, "action_mode_text", None)
        if prompt_sentence is not None:
            print(f"[SANA_WAM] prompt Action Mode sentence (from the training yaml): {prompt_sentence!r}")
        if self.visual_layout == VISUAL_LAYOUT_OPENWAM_CANVAS:
            print(
                f"[SANA_WAM] visual layout openwam_canvas: the 3 cameras are stretched into one "
                f"{OPENWAM_CANVAS_HEIGHT}x{OPENWAM_CANVAS_WIDTH} L-shaped RGB canvas (head 256x320 above the left/right "
                "wrists 128x160), encoded ONCE into a 12x10 latent grid (480 video tokens); one prompt shared by the "
                "video and action tokens"
            )

    def _log_normalization_contract(self) -> None:
        """One start-up line naming the normalization artifact actually loaded and the Robot80 slots it normalizes, and
        a check that they match the ARX-X5 joint-only contract (6 joints per arm in slots 0-5 / 29-34, grippers 16 / 45
        identity, the 7th-joint slots 6 / 35 unused). A layout mismatch -- e.g. a 7-joint artifact -- is reported loudly."""
        norm = getattr(self.session, "normalization", None)
        if norm is None or getattr(norm, "action_normalization_mask80", None) is None:
            return          # session doubles in the unit tests carry no artifact; the real session always does
        active = [int(i) for i in np.flatnonzero(np.asarray(norm.action_normalization_mask80, dtype=bool))]
        state_active = [int(i) for i in np.flatnonzero(np.asarray(norm.state_normalization_mask80, dtype=bool))]
        print(
            f"[SANA_WAM] normalization: path={norm.source_path} sha256={norm.sha256} action_mode={norm.action_mode} "
            f"action_representation={norm.action_representation} joint_target_mode={norm.joint_target_mode} "
            f"num_frames={norm.num_frames} model_fps={norm.model_fps} action_slots_normalized={active} state_slots_normalized={state_active}"
        )
        joint6 = list(range(0, 6)) + list(range(29, 35))
        problems = []
        if getattr(self, "include_eef", False):
            eef_position = [7, 8, 9, 36, 37, 38]
            eef_rotation = list(range(10, 16)) + list(range(39, 45))
            missing_pos = [i for i in eef_position if i not in active or i not in state_active]
            if missing_pos:
                problems.append(f"EEF position slots without statistics for a robot_base_eef checkpoint: {missing_pos}")
            rot_norm = [i for i in eef_rotation if i in active]
            print(f"[SANA_WAM] EEF slots served (state profile robot_base_eef): rot6d slots normalized={'yes' if rot_norm else 'no (identity)'}")
        missing = [i for i in joint6 if i not in active]
        if missing:
            problems.append(f"joint slots without action statistics: {missing}")
        seventh = [i for i in (6, 35) if i in active or i in state_active]
        if seventh:
            problems.append(f"7th-joint slots {seventh} carry statistics (ARX-X5 arms have 6 joints; the adapter fills slots 0-5 / 29-34 only)")
        grippers = [i for i in (16, 45) if i in active]
        if grippers:
            problems.append(f"gripper slots {grippers} are normalized (the contract keeps closedness as identity)")
        if problems:
            print("[SANA_WAM] WARNING normalization layout does not match the ARX-X5 joint-only contract: " + "; ".join(problems))
        else:
            print("[SANA_WAM] normalization layout OK: 6 joints per arm (slots 0-5 / 29-34), grippers 16 / 45 identity, slots 6 / 35 unused")

    # -- configuration --------------------------------------------------------

    @staticmethod
    def _resolve_ckpt_dir(cfg: Mapping[str, Any]) -> Path:
        root = resolve_checkpoint_root(
            dict(cfg), CHECKPOINTS_DIR, policy_dir=POLICY_DIR, explicit_keys=CHECKPOINT_EXPLICIT_KEYS, must_exist=True
        )
        return root

    @staticmethod
    def _resolve_visual_layout(cfg: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> str:
        """``visual_layout``: ``auto`` (default) reads the checkpoint's training yaml (model factory + dataset type); an
        explicit ``three_view_strip`` / ``openwam_canvas`` must agree with it. The key documents the operator's intent
        and can never re-route a checkpoint through the other front-end (the layout is a training-time contract)."""

        detected = resolve_visual_layout(train_cfg)
        raw = cfg.get("visual_layout")
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}):
            requested = "auto"
        else:
            requested = str(raw).strip().lower()
        if requested not in VISUAL_LAYOUT_CHOICES:
            raise ValueError(f"visual_layout must be one of {VISUAL_LAYOUT_CHOICES}, got {raw!r}")
        if requested != "auto" and requested != detected:
            raise ValueError(
                f"visual_layout {requested!r} does not match the checkpoint: its training yaml declares {detected!r} "
                f"(model.model {train_cfg['model']['model']}, data.type {train_cfg['data'].get('type')!r})"
            )
        return detected

    @staticmethod
    def _resolve_n_action_steps(cfg: Mapping[str, Any]) -> Optional[int]:
        """``n_action_steps``: how many leading targets of each predicted chunk are returned; None = the whole chunk.

        Accepts a positive integer (also as a string, the form ``--overrides`` may deliver), or
        ``null`` / ``0`` / ``"all"`` for the unchanged full-chunk behaviour.
        """

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

    @staticmethod
    def _resolve_anchor_source(cfg: Mapping[str, Any]) -> tuple[str, Optional[float]]:
        """``anchor_source``: the joint state the model is conditioned on and the anchor its joint deltas are added to.

        ``measured`` (default) = the evaluator's measured joints, the historical behaviour. ``last_command`` = the
        last joint target this adapter returned for the previous chunk, i.e. the convention of the training corpus,
        whose recorded state is the previous frame's command (``state[i+1] == action[i]``) while a simulator's
        measured joints lag the command it is tracking. ``last_command_clamped`` = measured + clip(last_command -
        measured, +-``anchor_clamp_rad``), a guard against command wind-up under blocking contact. Grippers follow
        the same source (last commanded closedness); the first chunk of every episode uses the measured state.
        """

        raw = cfg.get("anchor_source")
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}):
            source = "measured"
        elif isinstance(raw, str) and raw.strip().lower() in ANCHOR_SOURCES:
            source = raw.strip().lower()
        else:
            raise ValueError(f"anchor_source must be one of {ANCHOR_SOURCES}, got {raw!r}")
        raw_clamp = cfg.get("anchor_clamp_rad")
        clamp = DEFAULT_ANCHOR_CLAMP_RAD if _optional_float(raw_clamp) is None else float(raw_clamp)
        if not np.isfinite(clamp) or clamp <= 0:
            raise ValueError(f"anchor_clamp_rad must be a positive number of radians, got {raw_clamp!r}")
        return source, (clamp if source == "last_command_clamped" else None)

    @staticmethod
    def _resolve_state_profile(cfg: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> tuple[str, bool]:
        """``state_profile``: ``joint_only`` | ``robot_base_eef`` | ``auto`` (default) = read from the training yaml's
        ``data.extra.action_mode_sample_ratio`` (ACTION_MODES order qwen_canonical, robot_base_eef, joint_only): a
        positive robot_base_eef weight means the checkpoint was conditioned on and supervised with the EEF slots
        (``include_eef=True`` in Sana's ``pack_robot_base_eef_state80``). Returns ``(profile, include_eef)``."""

        raw = cfg.get("state_profile")
        profile = "auto" if raw is None else str(raw).strip().lower()
        if profile not in STATE_PROFILES:
            raise ValueError(f"state_profile must be one of {STATE_PROFILES}, got {raw!r}")
        if profile == "auto":
            extra = (train_cfg.get("data") or {}).get("extra") or {}
            ratios = extra.get("action_mode_sample_ratio")
            if ratios is None:
                profile = "joint_only"
            else:
                weights = [float(x) for x in ratios]
                if len(weights) != 3 or any(w < 0 for w in weights):
                    raise ValueError(f"unexpected data.extra.action_mode_sample_ratio {ratios!r} in the training yaml")
                if weights[0] > 0:
                    raise ValueError("qwen_canonical (camera-frame EEF) checkpoints are not supported by this adapter")
                profile = "robot_base_eef" if weights[1] > 0 else "joint_only"
        return profile, profile == "robot_base_eef"

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

    def _frames_from_obs(self, obs: Mapping[str, Any]) -> list[np.ndarray]:
        vision = obs["vision"]
        frames = [_rgb_frame(vision[cam]["color"], cam, self.strict_image_size) for cam in ROBODOJO_VIEW_ORDER]
        expected = (ROBODOJO_FROZEN_IMAGE_HEIGHT, ROBODOJO_FROZEN_IMAGE_WIDTH, 3)
        if not self._size_warned and any(tuple(f.shape) != expected for f in frames):
            target = (
                f"stretched into the {OPENWAM_CANVAS_HEIGHT}x{OPENWAM_CANVAS_WIDTH} OpenWAM canvas slots"
                if self.visual_layout == VISUAL_LAYOUT_OPENWAM_CANVAS
                else "resized/cropped to the trained 256x320 bucket"
            )
            warnings.warn(
                f"camera frames are not {expected} (got {[tuple(f.shape) for f in frames]}); they are {target}",
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
        env_idx = int(obs.get("env_idx", 0))
        frames = self._frames_from_obs(obs)
        state80_measured, state_mask80 = state80_from_obs(obs["state"])
        state80_raw = self._anchor_state(env_idx, state80_measured)
        if self.include_eef:
            # the training row pairs each arm's joints with FK(those joints) -> derive the EEF slots from the row's own
            # joints (measured, or the last command under anchor_source last_command), never from the observation's pose
            state80_raw, state_mask80 = fill_eef_state_slots(state80_raw, eef_state_slot_mask(True), self.kinematics)
            if self.eef_pose_check and self._chunk_index == 0:
                self._check_observed_eef_pose(obs["state"], state80_measured)
        instruction = self._instruction_from_obs(obs)
        if instruction != self._logged_instruction:
            # Tasks such as *_by_language render a different instruction per episode; log each text once so the
            # server log shows what the policy was actually conditioned on.
            print(f"[SANA_WAM] episode {self._episode_index} instruction: {instruction!r}", flush=True)
            self._logged_instruction = instruction
        seed = derive_diffusion_seed(self.diffusion_seed_base, self.eval_seed, self._episode_index, self._chunk_index)
        generator = torch.Generator(device=self.session.device).manual_seed(seed)
        result = self.session.predict(frames, state80_raw, state_mask80, instruction, generator)
        action80 = result.action80_raw_absolute
        if self.include_eef:
            # joints are already anchor + delta (session); the EEF slots are still anchor-relative deltas
            action80 = reconstruct_absolute_eef(action80, result.action_mask, state80_raw, state_mask80)
        if self.joint_lower is not None and self.joint_limit_mode in ("clip", "reject"):
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
        self._remember_last_command(env_idx, action80, len(actions))
        return [{k: np.ascontiguousarray(a[k], dtype=np.float32) for k in keys} for a in actions]

    def _check_observed_eef_pose(self, obs_state: Mapping[str, Any], state80_measured: np.ndarray) -> None:
        """Once per episode: compare the evaluator's ``*_ee_pose`` observation with FK(measured joints) through the
        configured root poses. Agreement (sub-mm) proves the URDF / root pose / axis conventions match the simulator;
        a large gap means the EEF state the model sees is in the wrong frame."""

        try:
            state_fk, _ = fill_eef_state_slots(state80_measured, eef_state_slot_mask(True), self.kinematics)
            gaps = observed_link6_discrepancy(obs_state, state_fk, self.world_from_base)
        except (KeyError, ValueError, TypeError) as exc:
            print(f"[SANA_WAM] eef check ep {self._episode_index}: skipped ({exc})", flush=True)
            return
        if not gaps:
            return
        text = " | ".join(f"{side} {1000 * pos:.2f} mm / {deg:.3f} deg" for side, (pos, deg) in gaps.items())
        bad = any(pos > EEF_POSE_CHECK_WARN_M or deg > EEF_POSE_CHECK_WARN_DEG for pos, deg in gaps.values())
        print(f"[SANA_WAM] eef check ep {self._episode_index}: observed *_ee_pose vs FK(measured joints): {text}", flush=True)
        if bad:
            warnings.warn(
                "observed end-effector pose disagrees with FK(measured joints) beyond "
                f"{1000 * EEF_POSE_CHECK_WARN_M:.0f} mm / {EEF_POSE_CHECK_WARN_DEG} deg: {text}; check robot_root_poses / urdf_path",
                stacklevel=2,
            )

    def _anchor_state(self, env_idx: int, state80_measured: np.ndarray) -> np.ndarray:
        """The Robot80 row this chunk is conditioned on and anchored to, per ``anchor_source``.

        Without a previous command (first chunk of the episode) every mode uses the measured row. Otherwise the
        lag ``last_command - measured`` of the 12 joints is logged per chunk (diagnostic in every mode), and under
        ``last_command`` / ``last_command_clamped`` the joint and gripper slots of the row are replaced by the last
        executed command (clamped to +-anchor_clamp_rad around the measurement in the clamped mode); every other
        slot stays as measured.
        """

        last = self._last_command80.get(env_idx)
        if last is None:
            return state80_measured
        joints = list(ROBOT80_JOINT_SLOTS_12)
        residual = last[joints] - state80_measured[joints]
        left, right = np.abs(residual[:ROBODOJO_ARM_DIM]), np.abs(residual[ROBODOJO_ARM_DIM:])
        clamped = 0
        if self.anchor_source == "measured":
            anchor = state80_measured
        else:
            anchor = np.array(state80_measured, dtype=np.float32, copy=True)
            if self.anchor_source == "last_command_clamped":
                tau = float(self.anchor_clamp_rad)
                clamped = int(np.count_nonzero(np.abs(residual) > tau))
                anchor[joints] = state80_measured[joints] + np.clip(residual, -tau, tau)
            else:
                anchor[joints] = last[joints]
            for slot in ROBOT80_GRIPPER_SLOTS_2:
                anchor[slot] = last[slot]
        # under ee actions the evaluator executes IK solutions of the EE targets, so "last_cmd" is the model's own joint
        # prediction rather than an executed command: the residual then measures prediction-vs-IK, not controller lag
        what = "|pred_joints-measured| (ee mode: joints were not executed)" if self.action_type == "ee" else "|last_cmd-measured|"
        print(
            f"[SANA_WAM] anchor ep {self._episode_index} chunk {self._chunk_index} src={self.anchor_source}: "
            f"{what} rad left max {left.max():.4f} mean {left.mean():.4f} | "
            f"right max {right.max():.4f} mean {right.mean():.4f}"
            + (f" | clamped {clamped}/12 at {self.anchor_clamp_rad}" if self.anchor_source == "last_command_clamped" else ""),
            flush=True,
        )
        return anchor

    def _remember_last_command(self, env_idx: int, action80: Any, executed: int) -> None:
        """Keep row ``executed - 1`` of the absolute chunk -- the last target actually returned, hence the last
        command the evaluator executes before it re-observes -- as the next chunk's last command."""

        rows = torch.as_tensor(action80).detach().to(device="cpu", dtype=torch.float32).numpy()
        self._last_command80[env_idx] = np.array(rows[int(executed) - 1], dtype=np.float32, copy=True)

    def _select_actions_to_execute(self, actions: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        """Keep the first ``n_action_steps`` targets of a predicted chunk (all of them when unset or n >= chunk).

        The RoboDojo loop executes every returned target, re-observing after each one, and only
        then asks for the next chunk; returning ``n < chunk`` targets therefore re-plans every
        ``n`` control ticks at the price of ``chunk / n`` times more inferences per episode.
        """

        chunk = len(actions)
        if self.action_chunk_size is None:
            self.action_chunk_size = chunk
            if self.n_action_steps is not None:
                print(
                    f"[SANA_WAM] n_action_steps={self.n_action_steps}: executing the first "
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
        self._obs, self._order = {}, []
        self._chunk_index = 0
        self._last_command80 = {}
        self._logged_instruction = None
        self._episode_index += 1

    def prepare_case(self, case_meta=None) -> None:
        return None

    def on_trial_end(self, result=None) -> None:
        return None
