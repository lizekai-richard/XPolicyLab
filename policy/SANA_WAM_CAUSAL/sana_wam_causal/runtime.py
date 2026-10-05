"""Everything a causal deploy loads once: the causal model, the shared front-end, the encoders, the normalization.

The non-model half is ``sana_wam_min``'s ``PolicyInferenceSession`` (the bidirectional SANA_WAM front-end) built on the
causal checkpoint's *bidirectional view* (``contract.bidirectional_view``): the sana_pixel canvas and its single-frame
VAE encode, the G = 1 canvas prompt rows and their Gemma encoding (conditional + unconditional), the normalization
artifact resolved and selected exactly as for the parent SFT checkpoint (f33 file name, yaml sha pin, absolute-EEF
statistics), and the action denormalization. This module adds what only the causal deploy needs: the causal model,
the 9-frame encode of an executed chunk and the per-episode ``CausalPolicySession``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.config import (  # noqa: E402
    load_train_config,
    policy_config_from_train_config,
    resolve_rope_contract,
)
from sana_wam_min.pixels import frames_to_vae_input  # noqa: E402
from sana_wam_min.robodojo_io import VIEW_SLOT_IDS  # noqa: E402
from sana_wam_min.sana_pixel_canvas import expected_sana_pixel_latent_hw, sana_pixel_canvas_from_frames  # noqa: E402
from sana_wam_min.session import (  # noqa: E402
    PolicyInferenceSession,
    find_train_config,
    load_checked_normalization,
    resolve_action_flow_shift,
    resolve_branch_cfg_scales,
    resolve_sampling_knobs,
)
from sana_wam_min.text import load_text_encoder  # noqa: E402
from sana_wam_min.vae import encode_video, load_vae  # noqa: E402

from .checkpoint import load_causal_weights  # noqa: E402
from .contract import CausalContract, bidirectional_view, resolve_causal_contract  # noqa: E402
from .model import CausalPolicyModel, build_causal_policy  # noqa: E402
from .session import WINDOW_RULES, Caption, CausalPolicySession  # noqa: E402


@dataclass
class CausalRuntime:
    model: CausalPolicyModel
    frontend: PolicyInferenceSession
    contract: CausalContract
    train_config: dict
    window_rule: str = "training"
    load_report: dict = field(default_factory=dict)

    # -- loading ------------------------------------------------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        checkpoint_dir: str,
        *,
        text_encoder_path: str,
        vae_path: str,
        device: str | torch.device = "cuda",
        normalization_path: Optional[str] = None,
        expected_normalization_sha256: Optional[str] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        video_cfg_scale: Optional[float] = None,
        action_cfg_scale: Optional[float] = None,
        flow_shift: Optional[float] = None,
        action_flow_shift: Optional[float] = None,
        view_resize: Optional[str] = None,
        allow_donor_checkpoint: bool = False,
        window_rule: str = "training",
    ) -> "CausalRuntime":
        """Load in the validated order: yaml -> contract (refusals before any weight) -> normalization -> bf16 causal
        model + weights -> VAE -> Gemma -> the shared front-end."""

        if window_rule not in WINDOW_RULES:
            raise ValueError(f"softmax_window must be one of {WINDOW_RULES}, got {window_rule!r}")
        device = torch.device(device)
        train_cfg = load_train_config(str(find_train_config(checkpoint_dir)))
        contract = resolve_causal_contract(train_cfg)
        view = bidirectional_view(train_cfg)
        rope, legacy_origin, rope_label = resolve_rope_contract(view, None)
        policy_config = policy_config_from_train_config(view, rope=rope, legacy_origin=legacy_origin)
        steps, cfg_scale, flow_shift = resolve_sampling_knobs(view, steps, cfg_scale, flow_shift)
        video_cfg_scale, action_cfg_scale = resolve_branch_cfg_scales(cfg_scale, video_cfg_scale, action_cfg_scale)
        normalization = load_checked_normalization(
            checkpoint_dir, view, normalization_path, expected_normalization_sha256
        )

        model = build_causal_policy(policy_config, contract, dtype=torch.bfloat16, device="cpu")
        load_report = load_causal_weights(
            model, str(checkpoint_dir), allow_missing_recurrence=allow_donor_checkpoint, device=device
        )
        vae = load_vae(vae_path, device=device, dtype=torch.bfloat16)
        tokenizer, text_encoder = load_text_encoder(text_encoder_path, device=device)
        frontend = PolicyInferenceSession(
            model=model,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            normalization=normalization,
            train_config=view,
            device=device,
            steps=steps,
            cfg_scale=cfg_scale,
            flow_shift=flow_shift,
            checkpoint_path=str(checkpoint_dir),
            video_cfg_scale=video_cfg_scale,
            action_cfg_scale=action_cfg_scale,
            action_flow_shift=action_flow_shift,
            view_resize=view_resize,
        )
        frontend.rope_contract = rope_label
        runtime = cls(
            model=model,
            frontend=frontend,
            contract=contract,
            train_config=train_cfg,
            window_rule=window_rule,
            load_report=load_report,
        )
        runtime._check_front_end()
        return runtime

    def _check_front_end(self) -> None:
        frontend = self.frontend
        if frontend.visual_layout != self.contract.visual_layout:
            raise RuntimeError(f"front-end layout {frontend.visual_layout} != contract {self.contract.visual_layout}")
        if int(frontend.video_frame_stride) != int(self.contract.video_frame_stride):
            raise RuntimeError("front-end and contract disagree on the video frame stride")
        if int(frontend.text_groups) != 1:
            raise RuntimeError(f"the one-view canvas carries one prompt row, the front-end resolved G={frontend.text_groups}")

    # -- per episode -------------------------------------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.frontend.device

    def new_session(self, instruction: str) -> CausalPolicySession:
        """A fresh memory for one episode, conditioned on ``instruction`` (conditional + unconditional rows)."""

        y, mask, null_y, null_mask = self.frontend.encode_instruction(instruction)
        frontend = self.frontend
        return CausalPolicySession(
            self.model,
            caption=Caption(y, mask),
            unconditional_caption=None if null_y is None else Caption(null_y, null_mask),
            steps=frontend.steps,
            flow_shift=frontend.flow_shift,
            action_flow_shift=frontend.action_flow_shift,
            video_cfg_scale=frontend.video_cfg_scale,
            action_cfg_scale=frontend.action_cfg_scale,
            model_fps=self.contract.model_fps,
            window=self.contract.sliding_window_chunks,
            window_rule=self.window_rule,
        )

    def _canvas(self, frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        frontend = self.frontend
        return sana_pixel_canvas_from_frames(
            [np.asarray(frame) for frame in frames_rgb],
            VIEW_SLOT_IDS,
            frontend.sana_pixel_canvas_hw,
            view_resize=frontend.view_resize,
        )

    @torch.inference_mode()
    def encode_frames(self, canvases: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode ``[3, H, W]`` canvases as ONE clip (causal LTX-2.3: frame 0 alone, then groups of 8)."""

        vae = self.frontend.vae
        clip = torch.stack(list(canvases), dim=0)
        video = frames_to_vae_input(clip).to(device=vae.device, dtype=vae.dtype)
        latent = encode_video(vae, video)
        expected = expected_sana_pixel_latent_hw(vae.spatial_compression, self.frontend.sana_pixel_canvas_hw)
        if (int(latent.shape[-2]), int(latent.shape[-1])) != tuple(expected):
            raise ValueError(f"canvas latent grid {tuple(latent.shape[-2:])} differs from the trained {expected}")
        return latent

    def encode_observation(self, frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        """The episode's first frame: ``[1, C_lat, 1, H, W]`` (the single-frame encode, latent 0 of a training window)."""

        return self.encode_frames([self._canvas(frames_rgb)])

    def encode_executed_chunk(self, frames_per_view: Sequence[Sequence[np.ndarray]]) -> torch.Tensor:
        """The executed chunk's target latents: the 1 + C/s observed frames (ticks 0, s, .., C of the chunk) encoded as
        one clip, the anchor latent dropped -> ``[1, C_lat, f_c, H, W]`` (training: store member ``w{s + cC}``,
        latent 1..)."""

        expected = 1 + self.contract.actions_per_chunk // self.contract.video_frame_stride
        if len(frames_per_view) != expected:
            raise ValueError(f"an executed chunk needs {expected} observed frames, got {len(frames_per_view)}")
        latent = self.encode_frames([self._canvas(frames) for frames in frames_per_view])
        targets = latent[:, :, 1:]
        if targets.shape[2] != self.contract.latent_frames_per_chunk:
            raise ValueError(
                f"the chunk clip encoded to {latent.shape[2]} latent frames; expected 1 + {self.contract.latent_frames_per_chunk}"
            )
        return targets


def timed(fn, *args, **kwargs):
    """``(result, seconds)`` with a CUDA sync when a GPU is in use (per-chunk latency logging)."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


__all__ = ["CausalRuntime", "timed"]
