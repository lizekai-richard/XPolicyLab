"""Everything a sparse-memory deploy loads once: the model, the shared front-end, the encoders, the normalization.

The non-model half is ``sana_wam_min``'s ``PolicyInferenceSession`` (the bidirectional SANA_WAM front-end) built on the
checkpoint's *bidirectional view* (``contract.bidirectional_view``): the sana_pixel canvas, the G = 1 canvas prompt rows
built for the one-chunk window (every training sample's prompt is the C + 1 = 33-row window's, whatever its n) and their
Gemma encoding (conditional + unconditional), the normalization artifact resolved and selected exactly as for the
parent SFT checkpoint, the action denormalization. This module adds the sparse memory's pixel path (canvases quantized
to uint8 as the training payload) and its one causal encode per decision (``vae_encode.encode_causal``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.config import load_train_config, policy_config_from_train_config, resolve_rope_contract  # noqa: E402
from sana_wam_min.robodojo_io import VIEW_SLOT_IDS  # noqa: E402
from sana_wam_min.sana_pixel_canvas import expected_sana_pixel_latent_hw, sana_pixel_canvas_from_frames  # noqa: E402
from sana_wam_min.session import (  # noqa: E402
    PolicyInferenceSession,
    find_train_config,
    load_checked_normalization,
    resolve_branch_cfg_scales,
    resolve_sampling_knobs,
)
from sana_wam_min.text import load_text_encoder  # noqa: E402
from sana_wam_min.vae import load_vae  # noqa: E402

from .checkpoint import load_sm_weights  # noqa: E402
from .contract import CANVAS_LATENT_HW, SMContract, bidirectional_view, resolve_sm_contract  # noqa: E402
from .model import SMPolicyModel, build_sm_policy  # noqa: E402
from .session import Caption, SMSession  # noqa: E402
from .vae_encode import canvas_uint8, encode_causal  # noqa: E402


@dataclass
class SMRuntime:
    model: SMPolicyModel
    frontend: PolicyInferenceSession
    contract: SMContract
    train_config: dict
    deploy_form: str
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
        deploy_form: Optional[str] = None,
        allow_donor_checkpoint: bool = False,
    ) -> "SMRuntime":
        """Load in the validated order: yaml -> contract (refusals before any weight) -> normalization -> bf16 model +
        weights -> VAE -> Gemma -> the shared front-end."""

        device = torch.device(device)
        train_cfg = load_train_config(str(find_train_config(checkpoint_dir)))
        contract = resolve_sm_contract(train_cfg)
        form = contract.resolve_deploy_form(deploy_form)
        view = bidirectional_view(train_cfg)
        rope, legacy_origin, rope_label = resolve_rope_contract(view, None)
        policy_config = policy_config_from_train_config(view, rope=rope, legacy_origin=legacy_origin)
        steps, cfg_scale, flow_shift = resolve_sampling_knobs(view, steps, cfg_scale, flow_shift)
        video_cfg_scale, action_cfg_scale = resolve_branch_cfg_scales(cfg_scale, video_cfg_scale, action_cfg_scale)
        normalization = load_checked_normalization(checkpoint_dir, view, normalization_path, expected_normalization_sha256)

        model = build_sm_policy(policy_config, contract, dtype=torch.bfloat16, device="cpu")
        load_report = load_sm_weights(
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
            deploy_form=form,
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
        latent_hw = tuple(expected_sana_pixel_latent_hw(self.vae.spatial_compression, frontend.sana_pixel_canvas_hw))
        if latent_hw != CANVAS_LATENT_HW:
            raise RuntimeError(f"the canvas latent grid {latent_hw} is not the sparse memory's {CANVAS_LATENT_HW}")

    # -- per episode -------------------------------------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.frontend.device

    @property
    def vae(self):
        return self.frontend.vae

    def new_session(self, instruction: str) -> SMSession:
        """The decisions of one episode, conditioned on ``instruction`` (conditional + unconditional rows)."""

        y, mask, null_y, null_mask = self.frontend.encode_instruction(instruction)
        frontend = self.frontend
        return SMSession(
            self.model,
            caption=Caption(y, mask),
            unconditional_caption=None if null_y is None else Caption(null_y, null_mask),
            steps=frontend.steps,
            flow_shift=frontend.flow_shift,
            action_flow_shift=frontend.action_flow_shift,
            video_cfg_scale=frontend.video_cfg_scale,
            action_cfg_scale=frontend.action_cfg_scale,
            model_fps=self.contract.model_fps,
            deploy_form=self.deploy_form,
        )

    def canvas(self, frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        """The uint8 ``[3, H, W]`` training canvas of one observation (view order of the SFT recipe)."""

        frontend = self.frontend
        canvas = sana_pixel_canvas_from_frames(
            [np.asarray(frame) for frame in frames_rgb],
            VIEW_SLOT_IDS,
            frontend.sana_pixel_canvas_hw,
            view_resize=frontend.view_resize,
        )
        return canvas_uint8(canvas)

    def encode(self, canvases: Sequence[torch.Tensor]) -> torch.Tensor:
        """One causal encode of ``1 + 8k`` uint8 canvases -> ``[1, C, 1 + k, h, w]``: a decision's prefix (Sana
        ``encode_prefix``), or a whole sample [sink | memory rows | target rows] (the trainer's encode)."""

        latent = encode_causal(self.vae, canvases)
        if (int(latent.shape[-2]), int(latent.shape[-1])) != CANVAS_LATENT_HW:
            raise ValueError(f"canvas latent grid {tuple(latent.shape[-2:])} differs from the trained {CANVAS_LATENT_HW}")
        return latent


__all__ = ["SMRuntime"]
