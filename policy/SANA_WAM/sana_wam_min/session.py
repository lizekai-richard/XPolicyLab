"""In-process inference session of the SANA unified world-action policy (RoboDojo joint-only line).

The session owns the four loaded components (policy transformer, LTX-2.3 VAE bundle, Gemma
tokenizer/encoder, Robot80 normalization) and reproduces the deployment ``predict`` of Sana's
``PolicyInferenceSession`` in the same order: token-group prompt encoding, per-view frame-0
VAE encode into a zero-filled latent window packed as one strip, ``data_info`` assembly,
video-then-action noise from one generator, Flow-Euler sampling under bf16 autocast, then
denormalize -> anchor + delta -> gripper clip. ``predict_from_latent`` replays the same
sampler call from a pre-encoded strip (validation bundles).
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .actions import model_action_to_absolute
from .checkpoint import build_policy_model, load_policy_weights, resolve_checkpoint_file
from .config import load_train_config, policy_config_from_train_config, sampling_defaults_from_train_config
from .multiview import pack_multiview_latents
from .pixels import frame_to_model_tensor, frames_to_vae_input, latent_frame_count, observation_window, target_size_hw, tier
from .robodojo_io import ROBODOJO_MODEL_FPS_HZ, ROBODOJO_VIEW_ORDER, VIEW_SLOT_IDS
from .robot80 import ROBOT80_DIM, Normalization, load_normalization, normalize_state
from .sampler import sample_policy
from .text import encode_prompt_rows, load_text_encoder, render_token_group_rows, unconditional_rows_from_conditional
from .vae import VaeBundle, encode_video, load_vae

TRAIN_CONFIG_NAME = "config.yaml"
NORMALIZATION_FILE_NAME = "robodojo_arx_x5_model_fps_25_f25_normalization.json"
# sha256 of the normalization artifact the checkpoint line was trained and validated with.
TRAINING_NORMALIZATION_SHA256 = "983fbd46df6af34e2048ed6806ae9ce49cd8bd3af72cfba7d8610069959ea1da"
# Fallback copy shipped next to the package (policy/SANA_WAM/normalization/).
PACKAGED_NORMALIZATION_PATH = Path(__file__).resolve().parent.parent / "normalization" / NORMALIZATION_FILE_NAME
PROMPT_CACHE_SIZE = 8


@dataclass
class PredictResult:
    """One chunk prediction: raw absolute Robot80 rows, the model-domain rows, and the sampled video strip."""

    action80_raw_absolute: torch.Tensor
    action80_model: torch.Tensor
    action_mask: torch.Tensor
    video_latent: torch.Tensor
    receipt: dict


def checkpoint_root(checkpoint_dir_or_file: str | Path) -> Path:
    """Return the checkpoint directory for a dir or its ``model/pytorch_model_fsdp.bin`` file path."""

    path = Path(checkpoint_dir_or_file).expanduser().resolve()
    if path.is_file() or path.suffix == ".bin":
        return path.parent.parent if path.parent.name == "model" else path.parent
    return path


def find_train_config(checkpoint_dir: str | Path) -> Path:
    """Return the training yaml next to the checkpoint dir, its parent, or its grandparent (first hit)."""

    start = checkpoint_root(checkpoint_dir)
    candidates = [start / TRAIN_CONFIG_NAME, start.parent / TRAIN_CONFIG_NAME, start.parent.parent / TRAIN_CONFIG_NAME]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no {TRAIN_CONFIG_NAME} found for checkpoint {start}; checked " + ", ".join(str(c) for c in candidates)
    )


def resolve_normalization_path(checkpoint_dir: str | Path, normalization_path: Optional[str | Path] = None) -> Path:
    """Explicit path, else ``<ckpt>/normalization/<canonical name>``, else the packaged copy; never a directory scan."""

    if normalization_path is not None:
        return Path(normalization_path).expanduser().resolve()
    canonical = checkpoint_root(checkpoint_dir) / "normalization" / NORMALIZATION_FILE_NAME
    if canonical.is_file():
        return canonical
    return PACKAGED_NORMALIZATION_PATH


def load_checked_normalization(
    checkpoint_dir: str | Path,
    train_cfg: dict,
    normalization_path: Optional[str | Path] = None,
    expected_sha256: Optional[str] = None,
) -> Normalization:
    """Resolve, sha-pin and load the normalization artifact, then cross-check it against the training yaml.

    The pin is enforced whenever ``expected_sha256`` is given; the packaged copy is pinned to the
    training artifact by default. The artifact's affine statistics and its declared mode are one
    construction; a mismatch with the yaml would silently reconstruct wrong absolute joints.
    """

    norm_path = resolve_normalization_path(checkpoint_dir, normalization_path)
    if expected_sha256 is None and norm_path == PACKAGED_NORMALIZATION_PATH:
        expected_sha256 = TRAINING_NORMALIZATION_SHA256
    normalization = load_normalization(norm_path, expected_sha256=expected_sha256)
    trained_mode = str(train_cfg["data"]["extra"]["joint_target_mode"])
    if normalization.joint_target_mode != trained_mode:
        raise ValueError(
            f"normalization joint_target_mode {normalization.joint_target_mode!r} differs from "
            f"the training yaml {trained_mode!r} ({norm_path})"
        )
    num_frames = int(train_cfg["data"]["num_frames"])
    if normalization.num_frames is not None and normalization.num_frames != num_frames:
        raise ValueError(f"normalization num_frames {normalization.num_frames} != training {num_frames} ({norm_path})")
    return normalization


def resolve_sampling_knobs(
    train_cfg: dict,
    steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    flow_shift: Optional[float] = None,
) -> tuple[int, float, float]:
    """Explicit values win; ``None`` falls back to the validated defaults of the training yaml."""

    defaults = sampling_defaults_from_train_config(train_cfg)
    steps = int(defaults["steps"] if steps is None else steps)
    cfg_scale = float(defaults["cfg_scale"] if cfg_scale is None else cfg_scale)
    flow_shift = float(defaults["flow_shift"] if flow_shift is None else flow_shift)
    if steps <= 0:
        raise ValueError("sampling steps must be positive")
    if not math.isfinite(cfg_scale) or cfg_scale < 1.0:
        raise ValueError("cfg_scale must be finite and >= 1.0")
    if not math.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError("flow_shift must be finite and positive")
    return steps, cfg_scale, flow_shift


class PolicyInferenceSession:
    """The loaded policy and its conditioning encoders, driven one observation chunk at a time."""

    def __init__(
        self,
        model: torch.nn.Module,
        vae: VaeBundle,
        tokenizer: Any,
        text_encoder: Any,
        normalization: Normalization,
        train_config: dict,
        device: str | torch.device = "cuda",
        steps: int = 50,
        cfg_scale: float = 1.0,
        flow_shift: float = 3.5,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.normalization = normalization
        self.train_config = train_config
        self.device = torch.device(device)
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.flow_shift = float(flow_shift)
        self.checkpoint_path = checkpoint_path
        self.image_size = int(train_config["model"]["image_size"])
        self.multi_fps = train_config["data"].get("multi_fps") or None
        self.fps = ROBODOJO_MODEL_FPS_HZ
        self.view_order = ROBODOJO_VIEW_ORDER
        self.view_slot_ids = VIEW_SLOT_IDS
        self.caption_max_length = int(train_config["text_encoder"]["model_max_length"])
        self._prompt_cache: OrderedDict = OrderedDict()

    @classmethod
    def from_paths(
        cls,
        checkpoint_dir: str,
        text_encoder_path: str,
        vae_path: str,
        normalization_path: Optional[str] = None,
        device: str | torch.device = "cuda",
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        flow_shift: Optional[float] = None,
        expected_normalization_sha256: Optional[str] = None,
    ) -> "PolicyInferenceSession":
        """Load every component from disk: train yaml -> normalization -> bf16 model + weights -> VAE -> Gemma.

        The normalization artifact is loaded and cross-checked before the 4.47B model is built so a
        wrong artifact fails before the multi-minute weight load. The model is converted to bf16
        before the fp32 checkpoint tensors are copied in (the validated order).
        """

        device = torch.device(device)
        config_path = find_train_config(checkpoint_dir)
        train_cfg = load_train_config(str(config_path))
        policy_config = policy_config_from_train_config(train_cfg)
        steps, cfg_scale, flow_shift = resolve_sampling_knobs(train_cfg, steps, cfg_scale, flow_shift)
        normalization = load_checked_normalization(
            checkpoint_dir, train_cfg, normalization_path, expected_normalization_sha256
        )

        model = build_policy_model(policy_config, dtype=torch.bfloat16, device=device)
        load_report = load_policy_weights(model, str(checkpoint_dir), device=device)

        vae = load_vae(vae_path, device=device, dtype=torch.bfloat16)
        tokenizer, text_encoder = load_text_encoder(text_encoder_path, device=device)
        session = cls(
            model=model,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            normalization=normalization,
            train_config=train_cfg,
            device=device,
            steps=steps,
            cfg_scale=cfg_scale,
            flow_shift=flow_shift,
            checkpoint_path=resolve_checkpoint_file(str(checkpoint_dir)),
        )
        session.load_report = load_report
        return session

    # -- conditioning --------------------------------------------------------

    def encode_rows(
        self,
        conditional_rows: Sequence[str],
        unconditional_rows: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode the G conditional rows (+ the unconditional rows in the same batch when cfg_scale > 1)."""

        rows = [tuple(conditional_rows)]
        if self.cfg_scale > 1:
            if unconditional_rows is None:
                unconditional_rows = unconditional_rows_from_conditional(conditional_rows)
            rows.append(tuple(unconditional_rows))
        y, mask = encode_prompt_rows(
            rows, self.tokenizer, self.text_encoder, self.device, max_length=self.caption_max_length
        )
        if self.cfg_scale > 1:
            return y[:1], mask[:1], y[1:2], mask[1:2]
        return y, mask, None, None

    def encode_instruction(self, instruction: str):
        """Render the token-group rows for ``instruction`` and encode them (LRU-cached per instruction)."""

        cached = self._prompt_cache.get(instruction)
        if cached is not None:
            return cached
        rows = render_token_group_rows(instruction, include_instruction=True, view_order=self.view_order)
        result = self.encode_rows(rows, None)
        if len(self._prompt_cache) >= PROMPT_CACHE_SIZE:
            self._prompt_cache.popitem(last=False)
        self._prompt_cache[instruction] = result
        return result

    def encode_observation(self, frames_rgb: Sequence[np.ndarray], latent_frames: int) -> tuple[torch.Tensor, tuple]:
        """Encode frame 0 of every view independently and pack the zero-filled windows into one strip."""

        if len(frames_rgb) != len(self.view_slot_ids):
            raise ValueError(f"expected {len(self.view_slot_ids)} views in order {self.view_order}, got {len(frames_rgb)}")
        windows = []
        for frame in frames_rgb:
            frame = np.asarray(frame)
            target = target_size_hw(self.image_size, frame_hw=(int(frame.shape[0]), int(frame.shape[1])))
            pixels = frame_to_model_tensor(frame, target)
            video = frames_to_vae_input(pixels.unsqueeze(0)).to(device=self.vae.device, dtype=self.vae.dtype)
            latent = encode_video(self.vae, video)
            windows.append(observation_window(latent, latent_frames))
        view_shapes = tuple((int(w.shape[-2]), int(w.shape[-1])) for w in windows)
        if len(set(view_shapes)) != 1:
            raise ValueError(f"single-resolution contract violated: view latent shapes {view_shapes}")
        strip, _ = pack_multiview_latents(windows)
        return strip, view_shapes

    def build_data_info(
        self,
        view_shapes: Sequence[tuple[int, int]],
        initial_state80: torch.Tensor,
        initial_state_mask80: torch.Tensor,
        action80: torch.Tensor,
        action_mask80: torch.Tensor,
    ) -> dict:
        """Assemble the conditioning dict the policy forward consumes (the sampler adds the per-step keys)."""

        num_views = len(view_shapes)
        if num_views != len(self.view_slot_ids):
            raise ValueError(f"view count {num_views} disagrees with the trained view plan {self.view_order}")
        device = self.device
        return {
            "rwm_task": "policy",
            "model_fps": torch.tensor([float(self.fps)], device=device),
            "num_views_per_sample": num_views,
            "sample_batch_size": 1,
            "view_count": torch.tensor([num_views], dtype=torch.int64, device=device),
            "view_latent_shape": torch.tensor([[list(s) for s in view_shapes]], dtype=torch.int64, device=device),
            "view_slot_ids": torch.tensor(self.view_slot_ids, dtype=torch.int64, device=device),
            "initial_state80": initial_state80.reshape(1, ROBOT80_DIM).to(device=device, dtype=torch.float32),
            "initial_state_condition_mask80": initial_state_mask80.reshape(1, ROBOT80_DIM).to(device=device, dtype=torch.bool),
            "action80": action80.to(device=device, dtype=torch.float32),
            "action_mask80": action_mask80.to(device=device, dtype=torch.bool),
            "camera_conditioning_enabled": False,
        }

    # -- sampling -----------------------------------------------------------

    def _sample(
        self,
        clean_video: torch.Tensor,
        clean_action: torch.Tensor,
        action_mask: torch.Tensor,
        text: tuple,
        data_info: dict,
        generator: Optional[torch.Generator],
        video_noise: Optional[torch.Tensor],
        action_noise: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        caption_embeds, caption_mask, uncond_embeds, uncond_mask = text
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            if video_noise is None:
                video_noise = torch.randn(clean_video.shape, device=self.device, dtype=clean_video.dtype, generator=generator)
            if action_noise is None:
                action_noise = torch.randn(clean_action.shape, device=self.device, dtype=clean_action.dtype, generator=generator)
            return sample_policy(
                self.model,
                clean_video,
                video_noise,
                action_noise,
                clean_action,
                action_mask,
                caption_embeds,
                caption_mask,
                uncond_embeds,
                uncond_mask,
                self.cfg_scale,
                data_info,
                self.steps,
                self.flow_shift,
            )

    @torch.inference_mode()
    def predict(
        self,
        frames_rgb: Sequence[np.ndarray],
        state80_raw: np.ndarray,
        state_mask80: np.ndarray,
        instruction: str,
        generator: Optional[torch.Generator] = None,
    ) -> PredictResult:
        """Run one chunk prediction from RGB uint8 HxWx3 frames (view order), raw Robot80 state and instruction."""

        start = time.perf_counter()
        frames, k_actions = tier(self.fps, None, self.multi_fps)
        latent_frames = latent_frame_count(frames, self.vae.temporal_compression)

        text = self.encode_instruction(instruction)
        window, view_shapes = self.encode_observation(frames_rgb, latent_frames)

        mask80 = torch.as_tensor(np.asarray(state_mask80)).to(device=self.device, dtype=torch.bool)
        action_mask = mask80.reshape(1, 1, ROBOT80_DIM).expand(1, k_actions, ROBOT80_DIM).contiguous()
        clean_action = torch.zeros((1, k_actions, ROBOT80_DIM), device=self.device, dtype=torch.float32)
        initial_state80 = normalize_state(state80_raw, state_mask80, self.normalization)
        data_info = self.build_data_info(view_shapes, initial_state80, mask80, clean_action, action_mask)

        video, action = self._sample(window, clean_action, action_mask, text, data_info, generator, None, None)

        action80_model = action.reshape(k_actions, ROBOT80_DIM).detach().to(device="cpu", dtype=torch.float32)
        action_mask_cpu = action_mask[0].detach().cpu()
        action80_raw = model_action_to_absolute(
            action80_model, action_mask_cpu, self.normalization, np.asarray(state80_raw), np.asarray(state_mask80)
        )
        receipt = {
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "flow_shift": self.flow_shift,
            "seed": None if generator is None else int(generator.initial_seed()),
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "checkpoint": self.checkpoint_path,
            "normalization_sha256": self.normalization.sha256,
            "view_latent_shapes": view_shapes,
            "frames": frames,
            "k_actions": k_actions,
        }
        return PredictResult(
            action80_raw_absolute=action80_raw,
            action80_model=action80_model,
            action_mask=action_mask_cpu,
            video_latent=video,
            receipt=receipt,
        )

    @torch.inference_mode()
    def predict_from_latent(
        self,
        clean_video_strip: torch.Tensor,
        view_latent_shapes: Sequence[tuple[int, int]],
        prompt_rows_cond: Sequence[str],
        prompt_rows_uncond: Optional[Sequence[str]],
        initial_state80_normalized: torch.Tensor,
        state_mask: torch.Tensor,
        clean_action_normalized: torch.Tensor,
        action_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        video_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay the sampler on a pre-encoded strip; noise comes from ``generator`` unless both tensors are given."""

        clean_video = clean_video_strip.to(self.device)
        clean_action = clean_action_normalized.to(device=self.device, dtype=torch.float32)
        mask = action_mask.to(device=self.device, dtype=torch.bool)
        text = self.encode_rows(prompt_rows_cond, prompt_rows_uncond)
        data_info = self.build_data_info(
            tuple((int(h), int(w)) for h, w in view_latent_shapes),
            torch.as_tensor(initial_state80_normalized),
            torch.as_tensor(state_mask),
            clean_action,
            mask,
        )
        noise = (None if video_noise is None else video_noise.to(self.device),
                 None if action_noise is None else action_noise.to(self.device))
        video, action = self._sample(clean_video, clean_action, mask, text, data_info, generator, *noise)
        return action, video


__all__ = [
    "NORMALIZATION_FILE_NAME",
    "checkpoint_root",
    "PACKAGED_NORMALIZATION_PATH",
    "PolicyInferenceSession",
    "PredictResult",
    "TRAINING_NORMALIZATION_SHA256",
    "find_train_config",
    "load_checked_normalization",
    "resolve_normalization_path",
    "resolve_sampling_knobs",
]
