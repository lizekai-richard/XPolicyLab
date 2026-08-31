# Copyright 2025 FlashVLA team. All rights reserved.
"""XPolicyLab adapter for FlashVLA checkpoints (pi05 baseline and pi05-flashvla).

The checkpoint declares its own policy type, so one adapter serves both:

  * ``pi05``          -- the chunked baseline. ``predict_action_chunk`` is called
    directly instead of ``select_action`` so no action queue lives inside the
    policy; the queue would be shared across RoboDojo's parallel envs.
  * ``pi05-flashvla`` -- action streaming. Its streaming buffer is per-episode
    and single-env, so it is only accepted when the client drives one env.

RoboDojo hands out one observation per env and expects a whole action chunk back
per env (``deploy.eval_one_episode_batch`` executes the chunk open-loop), which
maps onto a single batched forward across every running env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from flashvla.policies.factory import make_pre_post_processors

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

# RoboDojo names its overhead camera cam_head; the LeRobot datasets it exports
# name the same stream cam_high, which is what the checkpoints are trained on.
DEFAULT_CAMERA_MAP = {
    "observation.images.cam_high": "cam_head",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}

_STREAMING_TYPES = {"pi05-flashvla"}


def _load_policy_class(policy_type: str):
    if policy_type == "pi05":
        from flashvla.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    if policy_type == "pi05-flashvla":
        # Importing the config registers the "pi05-flashvla" type with PreTrainedConfig.
        from flashvla.policies.pi05.configuration_pi05 import PI05FlashVLAConfig  # noqa: F401
        from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy

        return PI05FlashVLAPolicy
    raise ValueError(
        f"Unsupported FlashVLA policy type {policy_type!r}. "
        "Expected 'pi05' or 'pi05-flashvla'."
    )


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg.get("task_name")
        self.action_type = model_cfg.get("action_type", "joint")
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        self.camera_map = dict(model_cfg.get("camera_map") or DEFAULT_CAMERA_MAP)

        device = model_cfg.get("device", "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[FlashVLA] CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

        self._observations: list[dict[str, Any]] = []
        self._env_idx_list: list[int] = []
        self._instruction: str | None = None

        self.policy = self.get_model(model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg: dict[str, Any]):
        checkpoint_root = resolve_checkpoint_root(
            model_cfg, _CHECKPOINTS_DIR, policy_dir=_POLICY_DIR
        )
        config = PreTrainedConfig.from_pretrained(checkpoint_root)
        config.device = str(self.device)

        self.policy_type = config.type
        self.is_streaming = self.policy_type in _STREAMING_TYPES
        if self.is_streaming:
            config.cold_start_mode = model_cfg.get("cold_start_mode", "current_state")

        n_action_steps = model_cfg.get("n_action_steps")
        if n_action_steps is not None:
            config.n_action_steps = int(n_action_steps)

        policy_class = _load_policy_class(self.policy_type)
        policy = policy_class.from_pretrained(checkpoint_root, config=config)
        policy.to(self.device)
        policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(checkpoint_root),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        self.n_action_steps = int(policy.config.n_action_steps)
        self._check_camera_map(policy.config)

        if self.is_streaming:
            from flashvla.async_manager import AsyncStreamingActionManager

            self.manager = AsyncStreamingActionManager(
                policy=policy,
                overlap_steps=int(model_cfg.get("inference_overlap_steps", 0)),
                skip_stale_actions=bool(model_cfg.get("skip_stale_actions", False)),
            )
        else:
            self.manager = None

        print(
            f"[FlashVLA] loaded {checkpoint_root} (type={self.policy_type}, "
            f"device={self.device}, n_action_steps={self.n_action_steps})"
        )
        return policy

    def _check_camera_map(self, config) -> None:
        """Fail at load time when the map does not cover the policy's cameras."""
        expected = {key for key in config.input_features if key.startswith("observation.images.")}
        mapped = set(self.camera_map)
        if expected != mapped:
            raise ValueError(
                "camera_map does not match the checkpoint's image features.\n"
                f"  checkpoint expects: {sorted(expected)}\n"
                f"  camera_map covers:  {sorted(mapped)}\n"
                "Set camera_map in deploy.yml to map each feature onto a RoboDojo camera."
            )

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if self.is_streaming and len(obs_list) > 1:
            raise ValueError(
                f"policy type {self.policy_type!r} keeps a per-episode streaming buffer and "
                f"cannot be driven with {len(obs_list)} parallel envs. Run the sim with "
                "num_envs=1 and eval_batch=false, or evaluate a 'pi05' checkpoint."
            )
        self._observations = list(obs_list)
        self._env_idx_list = [
            obs.get("env_idx", index) for index, obs in enumerate(obs_list)
        ]

    def get_action(self, **kwargs):
        return self.get_action_batch(env_idx_list=self._env_idx_list[:1], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._observations:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = list(env_idx_list) if env_idx_list else list(self._env_idx_list)
        observations = self._select_observations(env_idx_list)

        batch = self.preprocessor(self._encode_batch(observations))
        with torch.inference_mode():
            if self.is_streaming:
                # The manager returns one action per call; stack a chunk so the
                # client's open-loop replay length matches the baseline path.
                chunk = torch.stack(
                    [self.manager.act(batch) for _ in range(self.n_action_steps)], dim=1
                )
            else:
                chunk = self.policy.predict_action_chunk(batch)[:, : self.n_action_steps]

        actions = self._postprocess_chunk(chunk)
        return [
            unpack_robot_state(
                env_actions, self.action_type, self.robot_action_dim_info, source_type="obs"
            )
            for env_actions in actions
        ]

    def _select_observations(self, env_idx_list: list[int]) -> list[dict[str, Any]]:
        """Reorder the cached observations to match the requested envs.

        The client drops finished envs between steps, so the env list handed to
        get_action_batch can be a subset of the last update_obs_batch call.
        """
        by_env_idx = dict(zip(self._env_idx_list, self._observations))
        missing = [env_idx for env_idx in env_idx_list if env_idx not in by_env_idx]
        if missing:
            raise KeyError(
                f"no observation for env(s) {missing}; update_obs_batch was called with "
                f"{self._env_idx_list}"
            )
        return [by_env_idx[env_idx] for env_idx in env_idx_list]

    def _postprocess_chunk(self, chunk: torch.Tensor) -> np.ndarray:
        """Unnormalize a [B, T, D] chunk through the 2-D postprocessor contract."""
        batch_size, horizon, action_dim = chunk.shape
        flat = self.postprocessor(chunk.reshape(batch_size * horizon, action_dim))
        return (
            flat.detach()
            .to("cpu")
            .to(torch.float32)
            .numpy()
            .reshape(batch_size, horizon, action_dim)
        )

    def _encode_batch(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            feature_key: torch.stack(
                [
                    _to_chw_float(_extract_image(obs, camera_name))
                    for obs in observations
                ]
            ).to(self.device)
            for feature_key, camera_name in self.camera_map.items()
        }

        states = np.stack(
            [
                pack_robot_state(
                    obs, self.action_type, self.robot_action_dim_info, source_type="obs"
                )
                for obs in observations
            ]
        ).astype(np.float32)
        batch["observation.state"] = torch.from_numpy(states).to(self.device)
        batch["task"] = [self._resolve_instruction(obs) for obs in observations]
        return batch

    def _resolve_instruction(self, obs: dict[str, Any]) -> str:
        instruction = obs.get("instruction") or self.task_name or ""
        if instruction != self._instruction:
            self._instruction = instruction
            print(f"[FlashVLA] instruction: {instruction[:80]}")
        return instruction

    def reset(self):
        self._observations = []
        self._env_idx_list = []
        self._instruction = None
        if self.manager is not None:
            self.manager.reset()
        else:
            self.policy.reset()
        print("[FlashVLA] model reset")


def _extract_image(obs: dict[str, Any], camera_name: str) -> np.ndarray:
    """Pull one camera's color image out of a RoboDojo observation.

    The policy server decodes colors before calling the model, so this only has
    to locate the field. The array is RGB -- see process_data.decode_image_bit,
    do not swap channels here.
    """
    vision = obs.get("vision")
    if not isinstance(vision, dict) or camera_name not in vision:
        available = sorted(vision) if isinstance(vision, dict) else []
        raise KeyError(
            f"camera {camera_name!r} is not in the observation; available: {available}"
        )

    camera = vision[camera_name]
    if not isinstance(camera, dict):
        return np.asarray(camera)
    for image_key in ("color", "rgb", "image"):
        if image_key in camera:
            return np.asarray(camera[image_key])
    raise KeyError(f"camera {camera_name!r} carries no color field; keys: {sorted(camera)}")


def _to_chw_float(hwc_uint8: np.ndarray) -> torch.Tensor:
    if hwc_uint8.ndim != 3:
        raise ValueError(f"expected an HWC image, got shape {hwc_uint8.shape}")
    if hwc_uint8.shape[0] in (1, 3) and hwc_uint8.shape[-1] not in (1, 3):
        hwc_uint8 = np.transpose(hwc_uint8, (1, 2, 0))

    image = torch.from_numpy(np.ascontiguousarray(hwc_uint8))
    if image.dtype == torch.uint8:
        image = image.to(dtype=torch.float32) / 255.0
    else:
        image = image.to(dtype=torch.float32).clamp_(0.0, 1.0)
    return image.permute(2, 0, 1)
