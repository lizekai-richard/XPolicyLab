"""Everything a causal deploy loads once: the causal model, the shared front-end, the encoders, the normalization.

The non-model half is ``sana_wam_min``'s ``PolicyInferenceSession`` (the bidirectional SANA_WAM front-end) built on the
causal checkpoint's *bidirectional view* (``contract.bidirectional_view``): the view pixels (the three-view strip, each
camera resized to its bucket, or the sana_pixel canvas), the prompt rows (G = V + 1 for the strip, G = 1 for the
canvas) and their Gemma encoding (conditional + unconditional), the normalization artifact resolved and selected
exactly as for the parent SFT checkpoint (f33 file name, yaml sha pin, absolute-EEF statistics), and the action
denormalization. This module adds what only the causal deploy needs: the causal model, the single-frame observation
encode, the per-view causal VAE streams of an episode (``vae_stream.ViewStreams``: the sink, the frozen first chunk
and every executed chunk's targets) and the per-episode ``CausalPolicySession``.
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
from sana_wam_min.config import VISUAL_LAYOUT_SANA_PIXEL_CANVAS, VISUAL_LAYOUT_THREE_VIEW_STRIP  # noqa: E402
from sana_wam_min.multiview import pack_multiview_latents  # noqa: E402
from sana_wam_min.pixels import frame_to_model_tensor, target_size_hw  # noqa: E402
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
from sana_wam_min.vae import load_vae  # noqa: E402

from .checkpoint import load_causal_weights  # noqa: E402
from .contract import CausalContract, bidirectional_view, resolve_causal_contract  # noqa: E402
from .model import CausalPolicyModel, build_causal_policy  # noqa: E402
from .session import Caption, CausalPolicySession  # noqa: E402
from .vae_stream import ViewStreams, encode_window  # noqa: E402


@dataclass
class CausalRuntime:
    model: CausalPolicyModel
    frontend: PolicyInferenceSession
    contract: CausalContract
    train_config: dict
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
    ) -> "CausalRuntime":
        """Load in the validated order: yaml -> contract (refusals before any weight) -> normalization -> bf16 causal
        model + weights -> VAE -> Gemma -> the shared front-end."""

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
        views = len(frontend.view_slot_ids)
        groups = 1 if frontend.visual_layout == VISUAL_LAYOUT_SANA_PIXEL_CANVAS else views + 1
        if int(frontend.text_groups) != groups:
            raise RuntimeError(
                f"the {frontend.visual_layout} front-end carries G = {groups} prompt rows, it resolved G={frontend.text_groups}"
            )

    # -- per episode -------------------------------------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.frontend.device

    @property
    def strip(self) -> bool:
        """The three-view strip (several views on the batch's width axis) rather than the one-view canvas."""

        return self.frontend.visual_layout == VISUAL_LAYOUT_THREE_VIEW_STRIP

    @property
    def view_slot_ids(self) -> tuple[int, ...]:
        return tuple(int(slot) for slot in self.frontend.view_slot_ids) if self.strip else (0,)

    def new_session(self, instruction: str, view_shapes=None) -> CausalPolicySession:
        """A fresh memory for one episode, conditioned on ``instruction`` (conditional + unconditional rows);
        ``view_shapes`` are the strip's per-view latent shapes (None for the canvas)."""

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
            view_shapes=view_shapes if self.strip else None,
            view_slot_ids=self.view_slot_ids,
        )

    def view_frames(self, frames_rgb: Sequence[np.ndarray]) -> list[torch.Tensor]:
        """One tick's cameras (``ROBODOJO_VIEW_ORDER``) as the float ``[3, H, W]`` model tensors the VAE encodes: every
        view resized to its bucket (the strip), or the three composited into the sana_pixel canvas (one view)."""

        frontend = self.frontend
        frames = [np.asarray(frame) for frame in frames_rgb]
        if len(frames) != len(frontend.view_order):
            raise ValueError(f"expected {len(frontend.view_order)} camera frames in order {frontend.view_order}")
        if not self.strip:
            return [
                sana_pixel_canvas_from_frames(
                    frames, VIEW_SLOT_IDS, frontend.sana_pixel_canvas_hw, view_resize=frontend.view_resize
                )
            ]
        return [
            frame_to_model_tensor(
                frame,
                target_size_hw(frontend.image_size, frame_hw=(int(frame.shape[0]), int(frame.shape[1]))),
                frontend.view_resize,
            )
            for frame in frames
        ]

    def _check_grid(self, latent: torch.Tensor, view_shapes) -> None:
        vae = self.frontend.vae
        if not self.strip:
            expected = tuple(expected_sana_pixel_latent_hw(vae.spatial_compression, self.frontend.sana_pixel_canvas_hw))
            if (int(latent.shape[-2]), int(latent.shape[-1])) != expected:
                raise ValueError(f"canvas latent grid {tuple(latent.shape[-2:])} differs from the trained {expected}")
        elif len(set(view_shapes or ())) != 1:
            raise ValueError(f"single-resolution contract violated: view latent shapes {view_shapes}")

    @torch.inference_mode()
    def encode_observation(self, views: Sequence[torch.Tensor]) -> tuple[torch.Tensor, Optional[tuple]]:
        """A chunk's observation: its start frame encoded alone, every view on its own (``view_frames``) and packed ->
        (``[1, C_lat, 1, H, W]`` canvas or ``[1, C_lat, 1, 1, sum(h * w)]`` strip, the strip's view shapes or None).
        By the causal VAE it is the window encode's latent of that frame alone (the first emitted chunk's: the
        sink)."""

        latents = [encode_window(self.frontend.vae, [view]) for view in views]
        packed, shapes = pack_multiview_latents(latents)
        shapes = tuple(shapes) if self.strip else None
        self._check_grid(packed, shapes)
        return packed, shapes

    def new_streams(self) -> ViewStreams:
        """The per-view causal VAE streams of one episode (``start`` at its first frame, then one chunk at a time)."""

        return ViewStreams(
            self.frontend.vae, frames_per_chunk=self.contract.actions_per_chunk // self.contract.video_frame_stride,
            frozen_first_chunk=self.contract.first_chunk == 1,   # a first-frame window opens on the sink alone
        )


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
