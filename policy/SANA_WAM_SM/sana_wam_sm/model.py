"""Inference-only mirror of the sparse-memory policy (Sana rwm/sparse-memory ``sana_qwennext_policy_causal.py``:
``_forward_multiview`` -> ``_sparse_memory_layers`` -> ``_run_causal_layers`` / ``_layer_step``, and the deploy entries
``forward_memory_prefix`` / ``forward_memory_target``, checked against 3fa3e9689).

The model is the bidirectional policy trunk (``sana_wam_min.policy_model.PolicyModel``: embedders, AdaLN table, Block
AttnRes V2, text cross-attention, final layer, action head) with every block rebuilt around the shells of ``layers.py``.
One call evaluates one forward of ``sparse_memory.form_layout``:

* ``sequence``: the training forward, [S | M_1..M_n | T] + [state | 32 actions]; every block (GDN and softmax) on the
  whole sequence, one chunk per sink / memory latent and the target chunk last; the deploy's ``uncached`` form;
* ``prefix``: a decision's clean prefix [S | M_1..M_n] without robot rows, which fills every layer's cache: a GDN layer
  keeps its state after M_n, a softmax layer the prefix's rotated keys and values (Sana ``forward_memory_prefix``);
* ``target``: one denoising step of [T | state | actions] alone against the prefix's caches, which it reads and never
  writes (Sana ``forward_memory_target``).

The prefix never reads the target (the GDN scan reads its chunks in order, the softmax attention is chunk-causal), its t
is 0 and without AttnRes time conditioning its depth mixing is its own, so prefix + target reproduce the sequence's last
chunk (fp64 round-off: the matmuls run on fewer rows). Positions count latents: latent i at RoPE frame i (the target at
n + 1), one latent frame = 32 rows on the aligned clock, the robot rows at action steps 32n + r.
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

from . import sparse_memory as sm  # noqa: E402
from .contract import ACTION_TEMPORAL_COMPRESSION, SMContract  # noqa: E402
from .layers import GDN_KIND, GDNCache, SMPolicyBlock, SoftmaxCache  # noqa: E402

GDN_RECURRENCE_KEYS = ("gate_proj.weight", "gate_proj.bias", "A_log", "dt_bias", "recall_gate")


class SMPolicyModel(PolicyModel):
    """The sparse-memory policy's forwards on the bidirectional trunk (single view canvas, ``rope: aligned``)."""

    def __init__(self, config: PolicyConfig, contract: SMContract) -> None:
        nn.Module.__init__(self)
        config = config.validate()
        if config.rope not in (None, "aligned"):
            raise NotImplementedError(f"the sparse-memory mirror implements rope aligned only, got {config.rope!r}")
        if config.shared_prompt or config.state_as_cross_attention:
            raise NotImplementedError("the rwm/openwam canvas policy class has no causal variant")
        if config.sana_pixel_pad != "unmasked":
            raise NotImplementedError("sana_pixel_pad masked has no causal variant")
        self.policy_config = config
        self.contract = contract
        self.memory = contract.memory
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
        if self.chunk_actions != sm.CHUNK_ROWS or ACTION_TEMPORAL_COMPRESSION * self.video_frame_stride != sm.CHUNK_ROWS:
            raise ValueError(f"the sparse-memory sequence ends in one target latent of {sm.CHUNK_ROWS} rows")

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
            SMPolicyBlock(
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
        for block in self.blocks:
            if block.attn_type == GDN_KIND:
                block.attn.init_recurrence_parameters()

    def gdn_recurrence_keys(self) -> list[str]:
        keys = []
        for index, block in enumerate(self.blocks):
            if block.attn_type == GDN_KIND:
                keys.extend(f"blocks.{index}.attn.{name}" for name in GDN_RECURRENCE_KEYS)
        return keys

    @property
    def gdn_layers(self) -> list[int]:
        return [i for i, kind in enumerate(self.block_attn_types) if kind == GDN_KIND]

    # -- RoPE ---------------------------------------------------------------------------------------------------------

    def sm_rope(
        self,
        rope: PhysicalTimeWanRotaryPosEmbed,
        fps: torch.Tensor,
        phi: Sequence[int],
        height: int,
        width: int,
        robot_rows: int,
        step_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Rotary table ``[B, 1, F*H*W + robot_rows, D/2]`` of one sparse-memory forward.

        Video token (latent f, y, x): time ``base_fps * phi_f / (fps / s)`` with phi_f the latent's RoPE frame (Sana
        ``_physical_frame_ids`` with a ``video_frame_offset``: latent i at i, the target of a target forward at n + 1)
        on the aligned clock (s = the video frame stride); robot row r: time ``base_fps * (step_offset + r) / (8 * fps)``
        and spatial ids 0 (Sana ``_action_rope`` for one [state | C actions] chunk from ``chunk_step_offset``)."""

        batch = fps.numel()
        frames = len(phi)
        fps_video = fps.to(device=device, dtype=torch.float64) / int(self.video_frame_stride)
        ids = rope._position_ids(frames, height, width, fps_video, device)
        frame_ids = torch.as_tensor([float(p) for p in phi], dtype=torch.float64, device=device)
        frame_of_token = torch.arange(frames, device=device, dtype=torch.long).repeat_interleave(height * width)
        ids = ids.clone()
        ids[..., 0] = rope.base_fps * frame_ids[frame_of_token][None] / fps_video[:, None]
        if robot_rows:
            steps = (int(step_offset) + torch.arange(robot_rows, device=device, dtype=torch.long)).to(torch.float64)
            time_ids = rope.base_fps * steps[None] / (self.action_temporal_compression * fps.to(device=device)[:, None])
            robot = torch.stack((time_ids, torch.zeros_like(time_ids), torch.zeros_like(time_ids)), dim=-1)
            ids = torch.cat((ids, robot.expand(batch, -1, -1)), dim=1)
        return rope.from_position_ids(ids)

    # -- the trunk ----------------------------------------------------------------------------------------------------

    def _sm_trunk(
        self,
        tokens: torch.Tensor,
        text: torch.Tensor,
        modulation: torch.Tensor,
        text_mask,
        rope_linear: torch.Tensor,
        rope_softmax: torch.Tensor,
        bounds: Sequence[int],
        caches: Optional[Sequence[GDNCache | SoftmaxCache]],
    ) -> torch.Tensor:
        """Sana ``_run_causal_layers``: Block AttnRes V2 over depth, every block on the whole forward over ``bounds`` (its
        attention reading / writing its cache), one caption span for every token (G = 1: every chunk reads the same
        window caption)."""

        block_size = self.attn_res_block_size
        num_blocks = math.ceil(len(self.blocks) / block_size)
        value_buffer = torch.empty((num_blocks + 1, *tokens.shape), device=tokens.device, dtype=tokens.dtype)
        key_buffer = torch.empty_like(value_buffer)
        value_buffer[0] = tokens
        key_buffer[0] = self.attn_res.key_norm(tokens.unsqueeze(0)).squeeze(0)
        n_active = 1
        partial = None
        spans = ((0, int(tokens.shape[1])),)
        bounds = [int(b) for b in bounds]
        for block_index in range(num_blocks):
            start = block_index * block_size
            end = min(start + block_size, len(self.blocks))
            for layer_index in range(start, end):
                block = self.blocks[layer_index]
                hidden = self.attn_res._attend_buffer(self.attn_res.attn_proj, value_buffer, key_buffer, n_active, partial)
                attn_delta = block.forward_attn_sublayer(
                    hidden,
                    text,
                    modulation,
                    text_mask=text_mask,
                    rotary_emb=rope_linear if block.attn_type == GDN_KIND else rope_softmax,
                    prompt_group_spans=spans,
                    bounds=bounds,
                    cache=None if caches is None else caches[layer_index],
                )
                partial = attn_delta if partial is None else partial + attn_delta
                hidden = self.attn_res._attend_buffer(self.attn_res.mlp_proj, value_buffer, key_buffer, n_active, partial)
                partial = partial + block.forward_mlp_sublayer(hidden, modulation)
            value_buffer[n_active] = partial
            key_buffer[n_active] = self.attn_res.key_norm(partial.unsqueeze(0)).squeeze(0)
            n_active += 1
            partial = None
        return self.attn_res._attend_buffer(self.attn_res.final_proj, value_buffer, key_buffer, n_active, None)

    def _check_caches(self, form: str, caches, memory_latents: int) -> None:
        if (caches is None) != (form == sm.FORM_SEQUENCE):
            raise ValueError(f"a {form} forward {'takes no' if form == sm.FORM_SEQUENCE else 'needs'} layer caches")
        if caches is None:
            return
        if len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} layer caches, got {len(caches)}")
        prefix_tokens = sm.TOKENS_PER_FRAME * (memory_latents + 1)
        for index, (block, cache) in enumerate(zip(self.blocks, caches)):
            expected = GDNCache if block.attn_type == GDN_KIND else SoftmaxCache
            if not isinstance(cache, expected):
                raise TypeError(f"layer {index} ({block.attn_type}) takes a {expected.__name__}, got {type(cache).__name__}")
            held = cache.seed if isinstance(cache, GDNCache) else cache.prefix
            if form == sm.FORM_PREFIX and (held is not None or not cache.commit):
                raise ValueError(f"layer {index}: the prefix forward starts its cache empty and fills it (commit)")
            if form == sm.FORM_TARGET:
                if held is None or cache.commit:
                    raise ValueError(f"layer {index}: the target forward reads the prefix's cache and writes nothing")
                if isinstance(cache, SoftmaxCache) and int(cache.prefix[0].shape[1]) != prefix_tokens:
                    raise ValueError(
                        f"layer {index}: the prefix holds {int(cache.prefix[0].shape[1])} tokens, a target after "
                        f"{memory_latents} memory latents reads {prefix_tokens}"
                    )

    # -- one forward ----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def forward_sm(
        self,
        latent: torch.Tensor,
        video_timesteps: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        form: str,
        model_fps: Any,
        robot: Optional[Mapping[str, torch.Tensor]] = None,
        caches: Optional[Sequence[GDNCache | SoftmaxCache]] = None,
        memory_latents: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate one sparse-memory forward of ``form`` (``sparse_memory.form_layout``).

        Args:
            latent: ``[B, C, F, H, W]`` canvas latents: [S | M_1..M_n | T] (sequence), [S | M_1..M_n] (prefix) or [T]
                (target).
            video_timesteps: ``[B, F]`` raw per-frame timesteps: 0 on the sink and every memory latent, the step's t on
                the target.
            y / mask: ``[B, 1, 1, L, C_text]`` caption embeddings and ``[B, 1, L]`` key mask (G = 1).
            form: ``sequence`` | ``prefix`` | ``target``.
            model_fps: the batch fps (25 for RoboDojo).
            robot: ``state80`` / ``state_mask80`` ``[B, 80]``, ``action80`` / ``action_mask80`` ``[B, C, 80]`` and
                ``action_timesteps`` ``[B, C]`` for a sequence / target forward; None for the prefix.
            caches: one cache per block for the prefix (empty, filled here: ``GDNCache(commit=True)`` /
                ``SoftmaxCache(commit=True)``) and the target (the prefix's: ``GDNCache(seed=...)`` /
                ``SoftmaxCache(prefix=...)``); None for the sequence.
            memory_latents: n of a target forward (its frame n + 1, its rows from step 32n).

        Returns:
            ``{}`` for the prefix; else ``{"x": [B, C, F, H, W] velocity, "action_pred": [B, C, 80] velocity}``.
        """

        if self.training:
            raise RuntimeError("the sparse-memory mirror is inference-only; call eval()")
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, F, H, W], got {tuple(latent.shape)}")
        if form != sm.FORM_TARGET and memory_latents is not None:
            raise ValueError("memory_latents places a target forward; the sequence and the prefix count their own latents")
        layout = sm.form_layout(form, latent.shape[2] // self.patch_size[0], memory_latents)
        batch = latent.shape[0]
        device = latent.device
        x = latent.to(self.dtype)
        y = y.to(self.dtype)
        self.f, self.h, self.w = (
            x.shape[-3] // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )
        if self.h * self.w != sm.TOKENS_PER_FRAME:
            raise ValueError(
                f"a sparse-memory latent frame holds {sm.TOKENS_PER_FRAME} tokens (the chunk bounds), got {self.h} x {self.w}"
            )
        if (robot is None) != (layout["rows"] == 0):
            raise ValueError(f"a {form} forward carries {layout['rows']} robot rows; robot={'given' if robot else None}")
        self._check_caches(form, caches, layout["memory_latents"])
        view_shapes = ((self.h, self.w),)
        timesteps = torch.as_tensor(video_timesteps, device=device).reshape(batch, -1)
        if timesteps.shape[1] != self.f:
            raise ValueError(f"video_timesteps carries {timesteps.shape[1]} frames for a {self.f}-frame forward")
        if form == sm.FORM_PREFIX and bool((timesteps != 0).any()):
            raise ValueError("the prefix runs clean latents (t = 0)")
        if form == sm.FORM_SEQUENCE and bool((timesteps[:, :-1] != 0).any()):
            raise ValueError("only the target (the last latent) is noised; the sink and the memory latents are clean")

        video = strip_to_view_tokens(self.x_embedder(x), self.f, view_shapes)
        n_video = video.shape[1]
        video_timestep = strip_to_view_tokens(
            self._to_model_t_domain(timesteps).repeat_interleave(self.h * self.w, dim=1), self.f, view_shapes
        )
        if robot is None:
            tokens = video
            token_timestep = video_timestep.unsqueeze(1)
            action_mask = None
        else:
            action = torch.as_tensor(robot["action80"]).to(device)
            action_mask = torch.as_tensor(robot["action_mask80"]).to(device=device, dtype=torch.bool)
            state = torch.as_tensor(robot["state80"]).to(device).reshape(batch, -1)
            state_mask = torch.as_tensor(robot["state_mask80"]).to(device=device, dtype=torch.bool).reshape(batch, -1)
            if action.shape[1] != self.chunk_actions:
                raise ValueError(f"the target chunk carries {self.chunk_actions} action rows, got {action.shape[1]}")
            action_t = torch.as_tensor(robot["action_timesteps"], device=device).reshape(batch, -1)
            if action_t.shape[1] == 1:
                action_t = action_t.expand(-1, action.shape[1])
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
        if tokens.shape[1] != layout["bounds"][-1]:
            raise ValueError(f"a {form} forward holds {layout['bounds'][-1]} tokens, got {tokens.shape[1]}")

        fps = PhysicalTimeWanRotaryPosEmbed._normalize_fps(model_fps, batch, device)
        rope_linear = self.sm_rope(
            self.rope_linear, fps, layout["phi"], self.h, self.w, layout["rows"], layout["step_offset"], device
        )
        rope_softmax = self.sm_rope(
            self.rope_softmax, fps, layout["phi"], self.h, self.w, layout["rows"], layout["step_offset"], device
        )
        time_embedding = self.t_embedder(token_timestep.flatten()).unflatten(0, token_timestep.shape)
        modulation = self.t_block(time_embedding)
        text, text_mask = self._grouped_text_condition(y, mask)
        if text.shape[1] != 1:
            raise ValueError(f"the one-view canvas carries ONE prompt row (G = 1), got G={text.shape[1]}")
        hidden = self._sm_trunk(tokens, text, modulation, text_mask, rope_linear, rope_softmax, layout["bounds"], caches)
        if robot is None:
            return {}
        video_tokens = self.final_layer(hidden[:, :n_video], time_embedding[:, :, :n_video])
        video_tokens = view_to_strip_tokens(video_tokens, self.f, view_shapes)
        action_pred = self.action_head(hidden[:, n_video + 1 :], time_embedding[:, :, n_video + 1 :]).masked_fill(
            ~action_mask, 0
        )
        return {"x": self.unpatchify(video_tokens), "action_pred": action_pred}


def build_sm_policy(
    config: PolicyConfig,
    contract: SMContract,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> SMPolicyModel:
    """Construct in eval mode, converted to ``dtype`` before any weight is copied in."""

    model = SMPolicyModel(config, contract)
    model.set_fp32_attention(config.fp32_attention)
    return model.to(device=device, dtype=dtype).eval()


__all__ = ["GDN_RECURRENCE_KEYS", "SMPolicyModel", "build_sm_policy"]
