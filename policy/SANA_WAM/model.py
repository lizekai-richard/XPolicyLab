"""XPolicyLab adapter of the SANA unified world-action policy (RoboDojo ARX-X5, joint space).

The policy runs in-process: ``Model.__init__`` builds a :class:`PolicyInferenceSession`
from the vendored, Sana-free package ``sana_wam_min`` (policy transformer, Flow-Euler
sampler, LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning, Robot80 normalization
and the RoboDojo codecs) and every ``get_action`` call performs one 50-step diffusion
inference on the stored observation. The websocket ``PolicyServer`` decodes camera colors
before ``update_obs``; this module never decodes images and never swaps channels (the
checkpoint was trained on RGB frames).

Contract: ``update_obs`` stores the observation, ``get_action`` returns 24 joint-target dicts
(``left_arm_joint_state`` (6,), ``left_ee_joint_state`` (1,), ``right_arm_joint_state`` (6,),
``right_ee_joint_state`` (1,); float32), ``reset`` clears the observation and advances the
episode counter used for diffusion seeding. ``action_type`` must be ``joint`` and the robot
must resolve to dual six-joint arms with one gripper channel each.
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
from sana_wam_min.session import PolicyInferenceSession  # noqa: E402

CHECKPOINT_EXPLICIT_KEYS = ("checkpoint_dir", "checkpoint_path", "ckpt_dir", "model_dir")
JOINT_LIMIT_MODES = ("clip", "reject", "none")
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

        action_type = str(cfg.get("action_type") or "joint")
        if action_type != "joint":
            raise ValueError(f"SANA_WAM is a joint-space policy; action_type must be 'joint', got {action_type!r}")
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
        weight_dtype = str(cfg.get("weight_dtype") or "bfloat16")
        if weight_dtype not in ("bfloat16", "bf16"):
            raise ValueError(f"only bfloat16 weights are validated for SANA_WAM, got weight_dtype={weight_dtype!r}")
        self.device = torch.device(str(cfg.get("device") or "cuda"))
        self.strict_image_size = _is_true(cfg.get("strict_image_size", False))
        self.diffusion_seed_base = int(cfg.get("diffusion_seed_base") or 20260802)
        self.eval_seed = _optional_int(cfg.get("seed")) or 0
        self.default_instruction = str(cfg.get("default_instruction") or DEFAULT_INSTRUCTION)

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
        )
        self.model = self.session.model

        self._obs: dict[int, dict] = {}
        self._order: list[int] = []
        self._episode_index = 0
        self._chunk_index = 0
        self._size_warned = False
        print(
            f"[SANA_WAM] ready: ckpt={self.ckpt_dir} steps={self.session.steps} cfg_scale={self.session.cfg_scale} "
            f"flow_shift={self.session.flow_shift} device={self.device} joint_limit_mode={self.joint_limit_mode}"
        )

    # -- configuration --------------------------------------------------------

    @staticmethod
    def _resolve_ckpt_dir(cfg: Mapping[str, Any]) -> Path:
        root = resolve_checkpoint_root(
            dict(cfg), CHECKPOINTS_DIR, policy_dir=POLICY_DIR, explicit_keys=CHECKPOINT_EXPLICIT_KEYS, must_exist=True
        )
        return root

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
            warnings.warn(
                f"camera frames are not {expected} (got {[tuple(f.shape) for f in frames]}); they are resized/cropped "
                "to the trained 256x320 bucket",
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
        instruction = self._instruction_from_obs(obs)
        seed = derive_diffusion_seed(self.diffusion_seed_base, self.eval_seed, self._episode_index, self._chunk_index)
        generator = torch.Generator(device=self.session.device).manual_seed(seed)
        result = self.session.predict(frames, state80_raw, state_mask80, instruction, generator)
        action80 = result.action80_raw_absolute
        if self.joint_lower is not None and self.joint_limit_mode in ("clip", "reject"):
            action80, clipped = apply_joint_limits(
                action80, ROBOT80_JOINT_SLOTS_12, self.joint_lower, self.joint_upper, mode=self.joint_limit_mode
            )
            if clipped:
                warnings.warn(f"joint targets clipped to configured limits on Robot80 slots {clipped}", stacklevel=2)
        self._chunk_index += 1
        actions = upstream_actions_from_action80(action80)
        arm_dims, ee_dims = self.robot_action_dim_info["arm_dim"], self.robot_action_dim_info["ee_dim"]
        for action in actions:
            assert action["left_arm_joint_state"].shape == (int(arm_dims[0]),)
            assert action["right_arm_joint_state"].shape == (int(arm_dims[1]),)
            assert action["left_ee_joint_state"].shape == (int(ee_dims[0]),)
            assert action["right_ee_joint_state"].shape == (int(ee_dims[1]),)
        return [{k: np.ascontiguousarray(a[k], dtype=np.float32) for k in ROBODOJO_NATIVE_ACTION_KEYS} for a in actions]

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
        self._episode_index += 1

    def prepare_case(self, case_meta=None) -> None:
        return None

    def on_trial_end(self, result=None) -> None:
        return None
