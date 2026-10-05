"""Inference-only mirror of the chunk-causal policy (Sana ``sana_qwennext_policy_causal.py`` at c391260a4).

The model is the bidirectional policy trunk (``sana_wam_min.policy_model.PolicyModel``: embedders, AdaLN table,
Block AttnRes V2, text cross-attention, final layer, action head) with every block rebuilt around the chunk-causal
shells of ``layers.py``. It evaluates ONE deploy window per call, always a single chunk segment, which is every form
the streaming session needs (Sana ``CausalPolicyChunkSession`` + ``forward_chunk``):

* the observation prefill: the episode's first frame as a video-only clean entry, t = 0 (``commit_observation``);
* the chunk-0 window under ``obs_in_first_chunk``: [observation (t = 0) | chunk-0 targets] + [state | C actions], no
  cache -- the bidirectional single-chunk window with the causal GDN formula;
* the cached chunk window: [chunk-c targets] + [state | C actions] reading the session memory, either a denoising
  step (reads only) or the commit of the executed chunk at t = 0 (reads, then writes its GDN state and softmax K/V).

RoPE (``rope: aligned``): latent frame phi sits at ``base_fps * phi / (fps / s)`` (s = the video frame stride) and
robot step n at ``base_fps * n / (8 * fps)``; chunk c's targets are latent frame ``c * f_chunk + 1`` onwards, its state
row step ``c * C`` and its actions steps ``c * C + 1 .. c * C + C`` (the chunk-0 window starts both clocks at 0). The
canvas keeps the plain (t, y, x) grid; the robot rows carry spatial ids 0. One text group (G = 1) spans the window.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.policy_model.attnres import BlockAttnResV2  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402
from sana_wam_min.policy_model.embeddings import (  # noqa: E402
    ActionFinalLayer,
    CaptionEmbedder,
    MaskedProjector,
    PatchEmbedMS3D,
    RMSNorm,
    T2IFinalLayer,
    TimestepEmbedder,
)
from sana_wam_min.policy_model.geometry import strip_to_view_tokens, view_to_strip_tokens  # noqa: E402
from sana_wam_min.policy_model.model import PolicyModel  # noqa: E402
from sana_wam_min.policy_model.rope import PhysicalTimeWanRotaryPosEmbed, WanRotaryPosEmbed  # noqa: E402

from .contract import ACTION_TEMPORAL_COMPRESSION, CausalContract  # noqa: E402
from .layers import GDN_KIND, CausalPolicyBlock, LayerCache  # noqa: E402

GDN_RECURRENCE_KEYS = ("gate_proj.weight", "gate_proj.bias", "A_log", "dt_bias", "recall_gate")


class CausalPolicyModel(PolicyModel):
    """The causal policy's deploy windows on the bidirectional trunk (single view canvas, ``rope: aligned``)."""

    def __init__(self, config: PolicyConfig, contract: CausalContract) -> None:
        nn.Module.__init__(self)
        config = config.validate()
        if config.rope not in (None, "aligned"):
            raise NotImplementedError(f"the causal mirror implements rope aligned only, got {config.rope!r}")
        if config.shared_prompt or config.state_as_cross_attention:
            raise NotImplementedError("the rwm/openwam canvas policy class has no causal variant")
        if config.sana_pixel_pad != "unmasked":
            raise NotImplementedError("sana_pixel_pad masked has no causal variant")
        self.policy_config = config
        self.contract = contract
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.patch_size = tuple(config.patch_size)
        self.num_heads = config.num_heads
        self.linear_head_dim = config.linear_head_dim
        self.softmax_head_dim = config.softmax_head_dim
        self.softmax_layer_indices = list(config.softmax_layer_indices)
        self.softmax_attn_type = "GatedSoftmaxAttention"
        self.block_attn_types = list(config.block_attn_types)
        self.attn_res_block_size = config.attn_res_block_size
        self.timestep_norm_scale_factor = config.timestep_norm_scale_factor
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.action_temporal_compression = config.action_temporal_compression
        if int(self.action_temporal_compression) != ACTION_TEMPORAL_COMPRESSION:
            raise ValueError(f"action_temporal_compression must be {ACTION_TEMPORAL_COMPRESSION}")
        self.multiview_spatial_rope_layout = config.multiview_spatial_rope_layout
        self.multiview_spatial_rope_tile_shape = tuple(config.multiview_spatial_rope_tile_shape)
        self.shared_prompt = False
        self.state_as_cross_attention = False
        self.rope_mode = config.rope
        self.legacy_strided_action_origin = int(config.legacy_strided_action_origin)
        self.use_xformers_cross_attention = False
        self.chunk_actions = int(contract.actions_per_chunk)
        self.video_frame_stride = int(contract.video_frame_stride)
        self.latent_frames_per_chunk = int(contract.latent_frames_per_chunk)

        self.x_embedder = PatchEmbedMS3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(config.caption_channels, config.hidden_size)
        self.attention_y_norm = RMSNorm(config.hidden_size, scale_factor=config.y_norm_scale_factor, eps=config.norm_eps)
        self.rope_linear = PhysicalTimeWanRotaryPosEmbed(
            WanRotaryPosEmbed(config.linear_head_dim, max_seq_len=1024), attention_head_dim=config.linear_head_dim
        )
        self.rope_softmax = PhysicalTimeWanRotaryPosEmbed(
            WanRotaryPosEmbed(config.softmax_head_dim, max_seq_len=1024), attention_head_dim=config.softmax_head_dim
        )
        self.blocks = nn.ModuleList(
            CausalPolicyBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                attention_kind=kind,
                linear_head_dim=config.linear_head_dim,
                softmax_head_dim=config.softmax_head_dim,
                fp32_attention=config.fp32_attention,
                dt_bias_init=contract.dt_bias_init,
                a_log_init=contract.a_log_init,
            )
            for kind in self.block_attn_types
        )
        self.set_cross_attention_xformers(True)
        self.attn_res = BlockAttnResV2(config.hidden_size, config.depth, config.attn_res_block_size)
        self.final_layer = T2IFinalLayer(config.hidden_size, config.patch_size, config.out_channels)
        self.state_embed = MaskedProjector(config.state_dim, config.hidden_size)
        self.action_embed = MaskedProjector(config.action_dim, config.hidden_size)
        self.action_head = ActionFinalLayer(config.hidden_size, config.action_dim)
        self.f = self.h = self.w = 0
        self._initialize_weights()
        self.init_gdn_recurrence()
        self.eval()

    def init_gdn_recurrence(self) -> None:
        """Re-initialize every GDN layer's recurrence parameters exactly as Sana does when a checkpoint lacks them."""

        for block in self.blocks:
            if block.attn_type == GDN_KIND:
                block.attn.init_recurrence_parameters()

    def gdn_recurrence_keys(self) -> list[str]:
        """The state-dict keys a bidirectional SFT donor does not carry (the GDN recurrence of every GDN block)."""

        keys = []
        for index, block in enumerate(self.blocks):
            if block.attn_type == GDN_KIND:
                keys.extend(f"blocks.{index}.attn.{name}" for name in GDN_RECURRENCE_KEYS)
        return keys

    # -- geometry and RoPE ------------------------------------------------------------------------------------------

    def window_rope(
        self,
        rope: PhysicalTimeWanRotaryPosEmbed,
        fps: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        robot_rows: int,
        frame_offset: int,
        step_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Rotary table ``[B, 1, F*H*W + robot_rows, D/2]`` of one window on the physical clock.

        Video token (phi, y, x): time ``base_fps * phi / (fps / s)`` with phi = ``frame_offset + j``; robot row r: time
        ``base_fps * (step_offset + r) / (8 * fps)`` and spatial ids 0 (Sana ``_causal_multiview_rope`` +
        ``_action_rope`` for one chunk: row 0 is the state at the chunk's first step, row j + 1 action j)."""

        batch = fps.numel()
        fps_video = fps.to(device=device, dtype=torch.float64) / int(self.video_frame_stride)
        ids = rope._position_ids(frames, height, width, fps_video, device)
        phi = int(frame_offset) + torch.arange(frames, dtype=torch.float64, device=device)
        frame_of_token = torch.arange(frames, device=device, dtype=torch.long).repeat_interleave(height * width)
        ids = ids.clone()
        ids[..., 0] = rope.base_fps * phi[frame_of_token][None] / fps_video[:, None]
        if robot_rows:
            steps = (int(step_offset) + torch.arange(robot_rows, device=device, dtype=torch.long)).to(torch.float64)
            time_ids = rope.base_fps * steps[None] / (self.action_temporal_compression * fps.to(device=device)[:, None])
            robot = torch.stack((time_ids, torch.zeros_like(time_ids), torch.zeros_like(time_ids)), dim=-1)
            ids = torch.cat((ids, robot.expand(batch, -1, -1)), dim=1)
        return rope.from_position_ids(ids)

    # -- the trunk -----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def _causal_trunk(
        self,
        tokens: torch.Tensor,
        text: torch.Tensor,
        modulation: torch.Tensor,
        text_mask,
        prompt_group_spans,
        rope_linear: torch.Tensor,
        rope_softmax: torch.Tensor,
        caches: Optional[Sequence[LayerCache]],
    ) -> torch.Tensor:
        """``PolicyModel._forward_trunk`` (Block AttnRes V2 over depth, per token) with each layer's cache view."""

        block_size = self.attn_res_block_size
        num_blocks = math.ceil(len(self.blocks) / block_size)
        value_buffer = torch.empty((num_blocks + 1, *tokens.shape), device=tokens.device, dtype=tokens.dtype)
        key_buffer = torch.empty_like(value_buffer)
        value_buffer[0] = tokens
        key_buffer[0] = self.attn_res.key_norm(tokens.unsqueeze(0)).squeeze(0)
        n_active = 1
        partial = None
        for block_index in range(num_blocks):
            start = block_index * block_size
            end = min(start + block_size, len(self.blocks))
            for layer_index in range(start, end):
                rope = rope_softmax if self.block_attn_types[layer_index] == self.softmax_attn_type else rope_linear
                hidden = self.attn_res._attend_buffer(self.attn_res.attn_proj, value_buffer, key_buffer, n_active, partial)
                attn_delta = self.blocks[layer_index].forward_attn_sublayer(
                    hidden,
                    text,
                    modulation,
                    text_mask=text_mask,
                    rotary_emb=rope,
                    prompt_group_spans=prompt_group_spans,
                    cache=None if caches is None else caches[layer_index],
                )
                partial = attn_delta if partial is None else partial + attn_delta
                hidden = self.attn_res._attend_buffer(self.attn_res.mlp_proj, value_buffer, key_buffer, n_active, partial)
                partial = partial + self.blocks[layer_index].forward_mlp_sublayer(hidden, modulation)
            value_buffer[n_active] = partial
            key_buffer[n_active] = self.attn_res.key_norm(partial.unsqueeze(0)).squeeze(0)
            n_active += 1
            partial = None
        return self.attn_res._attend_buffer(self.attn_res.final_proj, value_buffer, key_buffer, n_active, None)

    # -- one deploy window ---------------------------------------------------------------------------------------------

    @torch.no_grad()
    def forward_window(
        self,
        latent: torch.Tensor,
        video_timesteps: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        model_fps: Any,
        frame_offset: int,
        step_offset: int,
        robot: Optional[Mapping[str, torch.Tensor]] = None,
        caches: Optional[Sequence[LayerCache]] = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate one single-segment window.

        Args:
            latent: ``[B, C, F, H, W]`` canvas latent frames of the window (observation and / or chunk targets).
            video_timesteps: ``[B, F]`` raw per-frame timesteps (0 for clean frames).
            y / mask: ``[B, 1, 1, L, C_text]`` caption embeddings and ``[B, 1, L]`` key mask (G = 1).
            model_fps: the batch fps (25 for RoboDojo).
            frame_offset / step_offset: physical index of the window's first latent frame and first robot step.
            robot: None for the video-only observation prefill, else ``state80`` / ``state_mask80`` ``[B, 80]``,
                ``action80`` / ``action_mask80`` ``[B, C, 80]`` and ``action_timesteps`` ``[B, C]`` (raw).
            caches: None for the cache-free chunk-0 window, else one ``LayerCache`` per block.

        Returns:
            ``{}`` for a prefill; else ``{"x": [B, C, F, H, W] velocity, "action_pred": [B, C, 80] velocity}``.
        """

        if self.training:
            raise RuntimeError("the causal mirror is inference-only; call eval()")
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, F, H, W], got {tuple(latent.shape)}")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} layer caches, got {len(caches)}")
        batch = latent.shape[0]
        device = latent.device
        x = latent.to(self.dtype)
        y = y.to(self.dtype)
        self.f, self.h, self.w = (
            x.shape[-3] // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )
        view_shapes = ((self.h, self.w),)
        timesteps = torch.as_tensor(video_timesteps, device=device).reshape(batch, -1)
        if timesteps.shape[1] != self.f:
            raise ValueError(f"video_timesteps carries {timesteps.shape[1]} frames for a {self.f}-frame window")

        video = strip_to_view_tokens(self.x_embedder(x), self.f, view_shapes)
        n_video = video.shape[1]
        video_timestep = strip_to_view_tokens(
            self._to_model_t_domain(timesteps).repeat_interleave(self.h * self.w, dim=1), self.f, view_shapes
        )
        if robot is None:
            tokens = video
            token_timestep = video_timestep.unsqueeze(1)
            robot_rows = 0
            action_mask = None
        else:
            action = torch.as_tensor(robot["action80"]).to(device)
            action_mask = torch.as_tensor(robot["action_mask80"]).to(device=device, dtype=torch.bool)
            state = torch.as_tensor(robot["state80"]).to(device).reshape(batch, -1)
            state_mask = torch.as_tensor(robot["state_mask80"]).to(device=device, dtype=torch.bool).reshape(batch, -1)
            action_steps = action.shape[1]
            if action_steps != self.chunk_actions:
                raise ValueError(f"a window carries {self.chunk_actions} action rows, got {action_steps}")
            action_t = torch.as_tensor(robot["action_timesteps"], device=device).reshape(batch, -1)
            if action_t.shape[1] == 1:
                action_t = action_t.expand(-1, action_steps)
            if bool((action_t != action_t[:, :1]).any()):
                raise ValueError("the action rows of one chunk share one timestep")
            robot_tokens = torch.cat(
                (
                    self.state_embed(state.to(dtype=video.dtype), state_mask).unsqueeze(1),
                    self.action_embed(action.to(dtype=video.dtype), action_mask),
                ),
                dim=1,
            )
            tokens = torch.cat((video, robot_tokens), dim=1)
            token_timestep = torch.cat(
                (video_timestep, video_timestep.new_zeros(batch, 1), self._to_model_t_domain(action_t)), dim=1
            ).unsqueeze(1)
            robot_rows = 1 + action_steps

        fps = PhysicalTimeWanRotaryPosEmbed._normalize_fps(model_fps, batch, device)
        rope_linear = self.window_rope(
            self.rope_linear, fps, self.f, self.h, self.w, robot_rows, frame_offset, step_offset, device
        )
        rope_softmax = self.window_rope(
            self.rope_softmax, fps, self.f, self.h, self.w, robot_rows, frame_offset, step_offset, device
        )
        time_embedding = self.t_embedder(token_timestep.flatten()).unflatten(0, token_timestep.shape)
        modulation = self.t_block(time_embedding)
        text, text_mask = self._grouped_text_condition(y, mask)
        if text.shape[1] != 1:
            raise ValueError(f"the one-view canvas carries ONE prompt row (G = 1), got G={text.shape[1]}")
        tokens = self._causal_trunk(
            tokens,
            text,
            modulation,
            text_mask,
            ((0, int(tokens.shape[1])),),
            rope_linear,
            rope_softmax,
            caches,
        )
        if robot is None:
            return {}
        video_tokens = self.final_layer(tokens[:, :n_video], time_embedding[:, :, :n_video])
        video_tokens = view_to_strip_tokens(video_tokens, self.f, view_shapes)
        action_pred = self.action_head(tokens[:, n_video + 1 :], time_embedding[:, :, n_video + 1 :]).masked_fill(
            ~action_mask, 0
        )
        return {"x": self.unpatchify(video_tokens), "action_pred": action_pred}


def build_causal_policy(
    config: PolicyConfig,
    contract: CausalContract,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> CausalPolicyModel:
    """Construct in eval mode, converted to ``dtype`` before any weight is copied in (the bidirectional mirror's order)."""

    model = CausalPolicyModel(config, contract)
    model.set_fp32_attention(config.fp32_attention)
    return model.to(device=device, dtype=dtype).eval()


__all__ = ["CausalPolicyModel", "GDN_RECURRENCE_KEYS", "build_causal_policy"]
