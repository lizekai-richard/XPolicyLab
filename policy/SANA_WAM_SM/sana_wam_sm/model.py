"""Inference-only mirror of the sparse-memory policy (Sana rwm/sparse-memory ``sana_qwennext_policy_causal.py``:
``_forward_multiview`` -> ``_sparse_memory_layers`` -> ``_run_causal_layers`` / ``_layer_step``, and the deploy entries
``forward_memory_prefix`` / ``forward_memory_target``, checked against cc18ad78f).

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

Two layouts: the one-view sana_pixel canvas (one 10 x 16 grid per latent, one caption group) and the sana_latent strip
(Sana a881daf7a / ebc25b98e: three 8 x 10 views per latent packed as [1, sum(h_v * w_v)], the semantic 2x2 RoPE tiles,
G = V + 1 caption groups, one per view and the robot group). A chunk holds one latent of every view, so the strip runs
its trunk in strip order (Sana ``_strip_order``) and the views' prompt groups in view-major order.
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
from sana_wam_min.policy_model.rope import (  # noqa: E402
    PhysicalTimeWanRotaryPosEmbed,
    WanRotaryPosEmbed,
    semantic_2x2_position_ids,
)

from . import sparse_memory as sm  # noqa: E402
from .contract import ACTION_TEMPORAL_COMPRESSION, SMContract  # noqa: E402
from .layers import GDN_KIND, GDNCache, SMPolicyBlock, SoftmaxCache  # noqa: E402

GDN_RECURRENCE_KEYS = ("gate_proj.weight", "gate_proj.bias", "A_log", "dt_bias", "recall_gate")
ViewShapes = tuple[tuple[int, int], ...]


def view_token_counts(frames: int, view_shapes: ViewShapes) -> list[int]:
    """Tokens of every view's run in view-major order: ``frames * h_v * w_v``."""

    return [int(frames) * int(height) * int(width) for height, width in view_shapes]


class SMPolicyModel(PolicyModel):
    """The sparse-memory policy's forwards on the bidirectional trunk (the one-view canvas or the sana_latent strip,
    ``rope: aligned``)."""

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
        view_shapes: ViewShapes,
        view_slot_ids: Sequence[int],
        robot_rows: int,
        step_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Rotary table ``[B, 1, sum_v F*H_v*W_v + robot_rows, D/2]`` of one sparse-memory forward, video tokens in
        view-major order (Sana ``_causal_multiview_rope``).

        Video token (latent f, view v, y, x): the bidirectional policy's spatial ids (the plain grid for one view, the
        semantic 2x2 tiles at ``view_slot_ids`` for several) and time ``base_fps * phi_f / (fps / s)`` with phi_f the
        latent's RoPE frame (Sana ``_physical_frame_ids`` with a ``video_frame_offset``: latent i at i, the target of a
        target forward at n + 1) on the aligned clock (s = the video frame stride); robot row r: time
        ``base_fps * (step_offset + r) / (8 * fps)`` and spatial ids 0 (Sana ``_action_rope`` for one [state | C actions]
        chunk from ``chunk_step_offset``)."""

        batch = fps.numel()
        frames = len(phi)
        fps_video = fps.to(device=device, dtype=torch.float64) / int(self.video_frame_stride)
        if len(view_shapes) == 1:
            (height, width), = view_shapes
            ids = rope._position_ids(frames, int(height), int(width), fps_video, device)
        else:
            ids = semantic_2x2_position_ids(
                frames, view_shapes, tuple(view_slot_ids), fps_video, rope.base_fps,
                self.multiview_spatial_rope_tile_shape, device,
            )
        frame_ids = torch.as_tensor([float(p) for p in phi], dtype=torch.float64, device=device)
        frame_of_token = torch.cat(
            [
                torch.arange(frames, device=device, dtype=torch.long).repeat_interleave(int(height) * int(width))
                for height, width in view_shapes
            ]
        )
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
        prompt_group_spans: Sequence[tuple[int, int]],
        text_order: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Sana ``_run_causal_layers``: Block AttnRes V2 over depth, every block on the whole forward over ``bounds`` (its
        attention reading / writing its cache), every token reading its caption group (``prompt_group_spans`` in the
        order ``text_order`` gathers; every chunk reads the same window caption)."""

        block_size = self.attn_res_block_size
        num_blocks = math.ceil(len(self.blocks) / block_size)
        value_buffer = torch.empty((num_blocks + 1, *tokens.shape), device=tokens.device, dtype=tokens.dtype)
        key_buffer = torch.empty_like(value_buffer)
        value_buffer[0] = tokens
        key_buffer[0] = self.attn_res.key_norm(tokens.unsqueeze(0)).squeeze(0)
        n_active = 1
        partial = None
        spans = tuple((int(offset), int(length)) for offset, length in prompt_group_spans)
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
                    text_order=text_order,
                )
                partial = attn_delta if partial is None else partial + attn_delta
                hidden = self.attn_res._attend_buffer(self.attn_res.mlp_proj, value_buffer, key_buffer, n_active, partial)
                partial = partial + block.forward_mlp_sublayer(hidden, modulation)
            value_buffer[n_active] = partial
            key_buffer[n_active] = self.attn_res.key_norm(partial.unsqueeze(0)).squeeze(0)
            n_active += 1
            partial = None
        return self.attn_res._attend_buffer(self.attn_res.final_proj, value_buffer, key_buffer, n_active, None)

    def _check_caches(self, form: str, caches, memory_latents: int, frame_tokens: int) -> None:
        if (caches is None) != (form == sm.FORM_SEQUENCE):
            raise ValueError(f"a {form} forward {'takes no' if form == sm.FORM_SEQUENCE else 'needs'} layer caches")
        if caches is None:
            return
        if len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} layer caches, got {len(caches)}")
        prefix_tokens = int(frame_tokens) * (memory_latents + 1)
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

    @staticmethod
    def _text_spans(text, text_mask, view_shapes: ViewShapes, frames: int, robot_rows: int):
        """(text, mask, query spans in view-major order) of one forward: G = 1 is one span over every token; G = V + 1 is
        one span per view run, then the robot rows, a forward without robot rows reading the V view groups alone (Sana
        ``_grouped_query_sublens``: a video-only segment carries no robot entry)."""

        groups = int(text.shape[1])
        counts = view_token_counts(frames, view_shapes)
        tokens = sum(counts) + int(robot_rows)
        if groups == 1:
            return text, text_mask, ((0, tokens),)
        views = len(view_shapes)
        if groups != views + 1:
            raise ValueError(f"a {views}-view forward reads G = 1 or G = V + 1 = {views + 1} prompt rows, got G={groups}")
        spans, offset = [], 0
        for count in counts:
            spans.append((offset, count))
            offset += count
        if robot_rows:
            spans.append((offset, int(robot_rows)))
        else:
            text = text[:, :views]
            text_mask = None if text_mask is None else text_mask[:, :views]
        return text, text_mask, tuple(spans)

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
        view_shapes: Optional[ViewShapes] = None,
        view_slot_ids: Sequence[int] = (0,),
    ) -> dict[str, torch.Tensor]:
        """Evaluate one sparse-memory forward of ``form`` (``sparse_memory.form_layout``).

        Args:
            latent: ``[B, C, F, H, W]`` canvas latents or the ``[B, C, F, 1, sum(h_v * w_v)]`` strip of several views:
                [S | M_1..M_n | T] (sequence), [S | M_1..M_n] (prefix) or [T] (target).
            video_timesteps: ``[B, F]`` raw per-frame timesteps: 0 on the sink and every memory latent, the step's t on
                the target.
            y / mask: ``[B, G, 1, L, C_text]`` caption embeddings and ``[B, G, L]`` key mask: G = 1 on the canvas,
                G = V + 1 on the strip (one prompt per view, then the robot group).
            form: ``sequence`` | ``prefix`` | ``target``.
            model_fps: the batch fps (25 for RoboDojo).
            robot: ``state80`` / ``state_mask80`` ``[B, 80]``, ``action80`` / ``action_mask80`` ``[B, C, 80]`` and
                ``action_timesteps`` ``[B, C]`` for a sequence / target forward; None for the prefix.
            caches: one cache per block for the prefix (empty, filled here: ``GDNCache(commit=True)`` /
                ``SoftmaxCache(commit=True)``) and the target (the prefix's: ``GDNCache(seed=...)`` /
                ``SoftmaxCache(prefix=...)``); None for the sequence.
            memory_latents: n of a target forward (its frame n + 1, its rows from step 32n).
            view_shapes / view_slot_ids: the strip's per-view latent (h, w) and semantic 2x2 slots (Sana
                ``view_latent_shape`` / ``view_slot_ids``); None (or one view) reads the latent's own grid.

        Returns:
            ``{}`` for the prefix; else ``{"x": velocity in the latent's layout, "action_pred": [B, C, 80] velocity}``.

        Several views run as Sana's ``_sparse_memory_layers`` runs them: embedded view-major, gathered into strip order
        (latent by latent, its views in order, the robot rows last) so that one chunk holds one latent of every view,
        and gathered back after the trunk; the per-token cross-attention reads its prompt group in view-major order.
        """

        if self.training:
            raise RuntimeError("the sparse-memory mirror is inference-only; call eval()")
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, F, H, W] or the strip [B, C, F, 1, S], got {tuple(latent.shape)}")
        if form != sm.FORM_TARGET and memory_latents is not None:
            raise ValueError("memory_latents places a target forward; the sequence and the prefix count their own latents")
        batch = latent.shape[0]
        device = latent.device
        x = latent.to(self.dtype)
        y = y.to(self.dtype)
        self.f, self.h, self.w = (
            x.shape[-3] // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )
        if view_shapes is None or len(view_shapes) == 1:
            if view_shapes is not None and tuple(int(v) for v in view_shapes[0]) != (self.h, self.w):
                raise ValueError(f"one view of latent shape {tuple(view_shapes[0])} on a {self.h} x {self.w} grid")
            view_shapes, view_slot_ids, strip = ((self.h, self.w),), (0,), False
        else:
            view_shapes = tuple((int(height), int(width)) for height, width in view_shapes)
            view_slot_ids = tuple(int(slot) for slot in view_slot_ids)
            if len(view_slot_ids) != len(view_shapes):
                raise ValueError(f"{len(view_shapes)} views need as many slot ids, got {view_slot_ids}")
            if tuple(self.patch_size) != (1, 1, 1) or self.h != 1 or self.w != sum(h * w for h, w in view_shapes):
                raise ValueError(
                    f"a strip of the views {view_shapes} is [B, C, F, 1, {sum(h * w for h, w in view_shapes)}] at patch "
                    f"size 1, got {tuple(latent.shape)} at patch size {tuple(self.patch_size)}"
                )
            strip = True
        frame_tokens = self.h * self.w
        layout = sm.form_layout(form, self.f, memory_latents, frame_tokens)
        if (robot is None) != (layout["rows"] == 0):
            raise ValueError(f"a {form} forward carries {layout['rows']} robot rows; robot={'given' if robot else None}")
        self._check_caches(form, caches, layout["memory_latents"], frame_tokens)
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
            self._to_model_t_domain(timesteps).repeat_interleave(frame_tokens, dim=1), self.f, view_shapes
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
        rope_args = (fps, layout["phi"], view_shapes, view_slot_ids, layout["rows"], layout["step_offset"], device)
        rope_linear = self.sm_rope(self.rope_linear, *rope_args)
        rope_softmax = self.sm_rope(self.rope_softmax, *rope_args)
        time_embedding = self.t_embedder(token_timestep.flatten()).unflatten(0, token_timestep.shape)
        modulation = self.t_block(time_embedding)
        text, text_mask = self._grouped_text_condition(y, mask)
        if not strip and text.shape[1] != 1:
            raise ValueError(f"the one-view canvas carries ONE prompt row (G = 1), got G={text.shape[1]}")
        text, text_mask, spans = self._text_spans(text, text_mask, view_shapes, self.f, layout["rows"])
        text_order = None
        if strip:
            # Sana ``_strip_order``: the view-major tokens in strip order, the robot rows last
            order = torch.cat(
                (
                    view_to_strip_tokens(torch.arange(n_video, device=device)[None, :, None], self.f, view_shapes)[0, :, 0],
                    torch.arange(n_video, tokens.shape[1], device=device),
                )
            )
            text_order = (torch.argsort(order), order)
            tokens = tokens.index_select(1, order)
            modulation = modulation.index_select(2, order)
            rope_linear = rope_linear.index_select(2, order)
            rope_softmax = rope_softmax.index_select(2, order)
        hidden = self._sm_trunk(
            tokens, text, modulation, text_mask, rope_linear, rope_softmax, layout["bounds"], caches, spans, text_order
        )
        if robot is None:
            return {}
        if text_order is not None:
            hidden = hidden.index_select(1, text_order[0])
        video_tokens = self.final_layer(hidden[:, :n_video], time_embedding[:, :, :n_video])
        video_tokens = view_to_strip_tokens(video_tokens, self.f, view_shapes)
        action_pred = self.action_head(hidden[:, n_video + 1 :], time_embedding[:, :, n_video + 1 :]).masked_fill(
            ~action_mask, 0
        )
        if strip:
            velocity = video_tokens.unflatten(1, (self.f, frame_tokens)).permute(0, 3, 1, 2).unsqueeze(3)
        else:
            velocity = self.unpatchify(video_tokens)
        return {"x": velocity, "action_pred": action_pred}


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


__all__ = ["GDN_RECURRENCE_KEYS", "SMPolicyModel", "ViewShapes", "build_sm_policy", "view_token_counts"]
