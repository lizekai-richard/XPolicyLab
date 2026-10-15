"""Inference-only mirror of the hierarchical-memory policy (Sana rwm/sparse-memory ``sana_qwennext_policy_causal.py``:
``_forward_multiview`` -> ``_memory_core`` -> ``_run_causal_layers`` / ``_layer_step``, checked against e8dde6b56).

The model is the bidirectional policy trunk (``sana_wam_min.policy_model.PolicyModel``: embedders, AdaLN table, Block
AttnRes V2, text cross-attention, final layer, action head) with every block rebuilt around the shells of ``layers.py``.
One call evaluates one forward of Sana's three forms (``memory.memory_layout``) or of the adapter's two commit-once
forms (``memory.stream_layout``: ``memory`` commits one memory's latents through the blocks that read it, ``noisy`` runs
the noisy chunk alone against every block's committed memory -- the deploy's default). Sana's forms:

* ``train``: [dense latents | sparse latents | obs | target] + 33 robot rows (no obs under ``chunk_observation: false``),
  each block kind computing on (the memory it reads + the noisy chunk) and every other token passing it unchanged; the
  deploy's ``full`` form (and the parity reference of the streaming one);
* ``commit`` ({gdn: dense, softmax: sparse}): one dense latent without robot rows through the GDN blocks alone, which
  read the carried state and keep the new one;
* ``generate`` ({gdn: dense, softmax: sparse}): [A_0.. | obs | target] + rows (no obs without ``chunk_observation``);
  the GDN blocks run the noisy chunk alone against the carried state (nothing kept), the softmax blocks the whole window
  prefix-causally.

Routing (Sana ``_layer_step``): the AttnRes depth attention runs per token over the whole sequence; a block computes on
its subset's tokens (``index_select``) with the subset's modulation, RoPE rows, chunk bounds and one text span, and its
attention / MLP deltas scatter back with zeros elsewhere; a kind without a subset (the commit's softmax blocks) is
skipped with a zero delta (it still opens the AttnRes group's partial, as training's scatter does). Positions: every latent frame sits at its row offset (``frame_offsets_x8 / 8`` rows, one latent frame = 32 rows
on the aligned clock), the robot rows at ``step_offset + r``.
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

from . import memory as hm  # noqa: E402
from .contract import ACTION_TEMPORAL_COMPRESSION, HMContract  # noqa: E402
from .layers import GDN_KIND, GDNCache, HMPolicyBlock, SoftmaxCache  # noqa: E402

GDN_RECURRENCE_KEYS = ("gate_proj.weight", "gate_proj.bias", "A_log", "dt_bias", "recall_gate")


class HMPolicyModel(PolicyModel):
    """The hierarchical-memory policy's forwards on the bidirectional trunk (single view canvas, ``rope: aligned``)."""

    def __init__(self, config: PolicyConfig, contract: HMContract) -> None:
        nn.Module.__init__(self)
        config = config.validate()
        if config.rope not in (None, "aligned"):
            raise NotImplementedError(f"the hierarchical-memory mirror implements rope aligned only, got {config.rope!r}")
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
            HMPolicyBlock(
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
        self._subset_cache: dict = {}
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

    def memory_rope(
        self,
        rope: PhysicalTimeWanRotaryPosEmbed,
        fps: torch.Tensor,
        frame_offsets_x8: torch.Tensor,
        height: int,
        width: int,
        robot_rows: int,
        step_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Rotary table ``[B, 1, F*H*W + robot_rows, D/2]`` of one hierarchical-memory forward.

        Video token (frame f, y, x): time ``base_fps * phi_f / (fps / s)`` with phi_f = ``frame_offsets_x8 / 8`` rows
        over 32 rows per latent frame (Sana ``_physical_frame_ids`` with ``frame_rows``); robot row r: time
        ``base_fps * (step_offset + r) / (8 * fps)`` and spatial ids 0 (Sana ``_action_rope`` for one chunk)."""

        batch = fps.numel()
        if frame_offsets_x8.dtype != torch.int64 or frame_offsets_x8.ndim != 2 or frame_offsets_x8.shape[0] != batch:
            raise ValueError(f"frame_offsets_x8 must be int64 [B={batch}, F], got {tuple(frame_offsets_x8.shape)} {frame_offsets_x8.dtype}")
        frames = int(frame_offsets_x8.shape[1])
        fps_video = fps.to(device=device, dtype=torch.float64) / int(self.video_frame_stride)
        ids = rope._position_ids(frames, height, width, fps_video, device)
        rows = frame_offsets_x8.to(device=device, dtype=torch.float64) / hm.OFFSET_SCALE
        phi = rows / (int(self.action_temporal_compression) * int(self.video_frame_stride))
        frame_of_token = torch.arange(frames, device=device, dtype=torch.long).repeat_interleave(height * width)
        ids = ids.clone()
        ids[..., 0] = rope.base_fps * phi[..., frame_of_token] / fps_video[:, None]
        if robot_rows:
            steps = (int(step_offset) + torch.arange(robot_rows, device=device, dtype=torch.long)).to(torch.float64)
            time_ids = rope.base_fps * steps[None] / (self.action_temporal_compression * fps.to(device=device)[:, None])
            robot = torch.stack((time_ids, torch.zeros_like(time_ids), torch.zeros_like(time_ids)), dim=-1)
            ids = torch.cat((ids, robot.expand(batch, -1, -1)), dim=1)
        return rope.from_position_ids(ids)

    # -- routing ------------------------------------------------------------------------------------------------------

    def _subset_index(self, key: tuple, index: torch.Tensor, device: torch.device) -> torch.Tensor:
        key = (*key, int(self.memory["sparse_frames"]), str(device))
        cached = self._subset_cache.get(key)
        if cached is None:
            cached = self._subset_cache[key] = index.to(device)
        return cached

    def _hm_trunk(
        self,
        tokens: torch.Tensor,
        text: torch.Tensor,
        modulation: torch.Tensor,
        text_mask,
        plans: Mapping[str, Optional[dict]],
        caches: Optional[Sequence[Optional[GDNCache | SoftmaxCache]]],
    ) -> torch.Tensor:
        """Sana ``_run_causal_layers`` with ``memory`` plans: Block AttnRes V2 over depth on the whole sequence, each block
        on its kind's subset with deltas scattered back; a kind without a plan is skipped."""

        block_size = self.attn_res_block_size
        num_blocks = math.ceil(len(self.blocks) / block_size)
        value_buffer = torch.empty((num_blocks + 1, *tokens.shape), device=tokens.device, dtype=tokens.dtype)
        key_buffer = torch.empty_like(value_buffer)
        value_buffer[0] = tokens
        key_buffer[0] = self.attn_res.key_norm(tokens.unsqueeze(0)).squeeze(0)
        n_active = 1
        partial = None

        def scatter(delta: torch.Tensor, plan: dict) -> torch.Tensor:
            if plan["full"]:
                return delta
            return torch.zeros_like(tokens, dtype=delta.dtype).index_copy(1, plan["index"], delta)

        def gather(hidden: torch.Tensor, plan: dict) -> torch.Tensor:
            return hidden if plan["full"] else hidden.index_select(1, plan["index"])

        for block_index in range(num_blocks):
            start = block_index * block_size
            end = min(start + block_size, len(self.blocks))
            for layer_index in range(start, end):
                block = self.blocks[layer_index]
                kind = hm.GDN if block.attn_type == GDN_KIND else hm.SOFTMAX
                plan = plans[kind]
                if plan is None:
                    # a block the forward's tokens pass unchanged contributes a ZERO delta, as training's scatter does: in
                    # Block AttnRes a zero partial is one more depth source (key_norm(0) = 0, logit 0), so "no partial"
                    # and "a zero partial" mix differently. Sana's commit form skips (``continue``), which agrees only
                    # because every AttnRes group of the real trunk starts with a GDN block; a commit of the sparse
                    # memory skips the GDN blocks, which lead every group.
                    if partial is None:
                        partial = torch.zeros_like(tokens)
                    continue
                cache = None if caches is None else caches[layer_index]
                hidden = self.attn_res._attend_buffer(self.attn_res.attn_proj, value_buffer, key_buffer, n_active, partial)
                attn_delta = scatter(
                    block.forward_attn_sublayer(
                        gather(hidden, plan),
                        text,
                        plan["modulation"],
                        text_mask=text_mask,
                        rotary_emb=plan["rope"],
                        prompt_group_spans=((0, plan["tokens"]),),
                        bounds=plan["bounds"],
                        cache=cache,
                    ),
                    plan,
                )
                partial = attn_delta if partial is None else partial + attn_delta
                hidden = self.attn_res._attend_buffer(self.attn_res.mlp_proj, value_buffer, key_buffer, n_active, partial)
                partial = partial + scatter(block.forward_mlp_sublayer(gather(hidden, plan), plan["modulation"]), plan)
            if partial is None:
                partial = torch.zeros_like(tokens)
            value_buffer[n_active] = partial
            key_buffer[n_active] = self.attn_res.key_norm(partial.unsqueeze(0)).squeeze(0)
            n_active += 1
            partial = None
        return self.attn_res._attend_buffer(self.attn_res.final_proj, value_buffer, key_buffer, n_active, None)

    # -- one forward ----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def forward_hm(
        self,
        latent: torch.Tensor,
        video_timesteps: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        form: str,
        k: int,
        frame_offsets_x8: torch.Tensor,
        model_fps: Any,
        step_offset: int = 0,
        robot: Optional[Mapping[str, torch.Tensor]] = None,
        caches: Optional[Sequence[Optional[GDNCache | SoftmaxCache]]] = None,
        memory_kind: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate one hierarchical-memory forward of ``form`` (``memory.memory_layout`` / ``memory.stream_layout``).

        Args:
            latent: ``[B, C, F, H, W]`` canvas latent frames in the form's layout.
            video_timesteps: ``[B, F]`` raw per-frame timesteps: 0 on every memory latent and the observation (when the
                noisy chunk has one), the step's t on the target (the last frame of a train / generate / noisy forward);
                all 0 for a commit.
            y / mask: ``[B, 1, 1, L, C_text]`` caption embeddings and ``[B, 1, L]`` key mask (G = 1).
            form: ``train`` | ``commit`` | ``generate`` (Sana) | ``memory`` | ``noisy`` (the adapter's commit-once forms).
            k: K of a train forward with a dense memory (the dense latents are L_0..L_K); ignored otherwise.
            frame_offsets_x8: int64 ``[B, F]`` row offsets in eighths of every latent frame (``memory.offsets_x8``).
            model_fps: the batch fps (25 for RoboDojo).
            step_offset: the robot rows' first step on the same clock as the frame rows (0 on the training clock, 32c
                on the episode clock of decision c).
            robot: ``state80`` / ``state_mask80`` ``[B, 80]``, ``action80`` / ``action_mask80`` ``[B, C, 80]`` and
                ``action_timesteps`` ``[B, C]`` for a train / generate forward; None for a commit.
            caches: one cache (or None) per block for every form but train: a GDN block's ``GDNCache`` (seed / commit),
                a softmax block's ``SoftmaxCache`` (the commit-once forms only; Sana's forms keep no softmax K / V).
            memory_kind: the memory a ``memory`` commit carries (``dense`` / ``sparse``).

        Returns:
            ``{}`` for a commit; else ``{"x": [B, C, F, H, W] velocity, "action_pred": [B, C, 80] velocity}``.
        """

        if self.training:
            raise RuntimeError("the hierarchical-memory mirror is inference-only; call eval()")
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, F, H, W], got {tuple(latent.shape)}")
        if form in hm.STREAM_FORMS:
            layout = hm.stream_layout(form, self.memory, kind=memory_kind, latents=latent.shape[2] // self.patch_size[0])
            subset_key = (form, memory_kind, int(layout["frames"]))
        else:
            if memory_kind is not None:
                raise ValueError(f"memory_kind names the memory of a commit-once 'memory' forward, not of {form!r}")
            layout = hm.memory_layout(form, k, self.memory)
            subset_key = (form, int(k))
        batch = latent.shape[0]
        device = latent.device
        x = latent.to(self.dtype)
        y = y.to(self.dtype)
        self.f, self.h, self.w = (
            x.shape[-3] // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )
        if self.f != layout["frames"] or self.h * self.w != hm.LATENT_TOKENS:
            raise ValueError(
                f"a hierarchical-memory {form} forward holds {layout['frames']} latent frames of {hm.LATENT_TOKENS} tokens "
                f"(K={k}); got {self.f} frames of {self.h} x {self.w}"
            )
        if (robot is None) != (layout["rows"] == 0):
            raise ValueError(f"a {form} forward carries {layout['rows']} robot rows; robot={'given' if robot else None}")
        if (caches is None) != (form == hm.FORM_TRAIN):
            raise ValueError(f"a {form} forward {'takes no' if form == hm.FORM_TRAIN else 'needs'} GDN caches")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} layer caches, got {len(caches)}")
        if caches is not None and form in (hm.FORM_COMMIT, hm.FORM_GENERATE):
            for block, cache in zip(self.blocks, caches):
                if block.attn_type != GDN_KIND and cache is not None:
                    raise ValueError(f"Sana's {form} form keeps no softmax K/V (its sparse memory rides the generate window)")
        if tuple(frame_offsets_x8.shape) != (batch, self.f):
            raise ValueError(f"frame_offsets_x8 must be [B={batch}, F={self.f}], got {tuple(frame_offsets_x8.shape)}")
        view_shapes = ((self.h, self.w),)
        timesteps = torch.as_tensor(video_timesteps, device=device).reshape(batch, -1)
        if timesteps.shape[1] != self.f:
            raise ValueError(f"video_timesteps carries {timesteps.shape[1]} frames for a {self.f}-frame forward")
        if form in (hm.FORM_COMMIT, hm.FORM_MEMORY) and bool((timesteps != 0).any()):
            raise ValueError("a commit forward runs clean latents (t = 0)")
        if form not in (hm.FORM_COMMIT, hm.FORM_MEMORY) and bool((timesteps[:, :-1] != 0).any()):
            raise ValueError("only the target (the last frame) is noised; every memory latent and the observation are clean")

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
                raise ValueError(f"a noisy chunk carries {self.chunk_actions} action rows, got {action.shape[1]}")
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
        if tokens.shape[1] != hm.LATENT_TOKENS * layout["frames"] + layout["rows"]:
            raise ValueError(f"a {form} forward holds {hm.LATENT_TOKENS * layout['frames'] + layout['rows']} tokens, got {tokens.shape[1]}")

        fps = PhysicalTimeWanRotaryPosEmbed._normalize_fps(model_fps, batch, device)
        offsets = frame_offsets_x8.to(device)
        rope_linear = self.memory_rope(
            self.rope_linear, fps, offsets, self.h, self.w, layout["rows"], step_offset, device
        )
        rope_softmax = self.memory_rope(
            self.rope_softmax, fps, offsets, self.h, self.w, layout["rows"], step_offset, device
        )
        time_embedding = self.t_embedder(token_timestep.flatten()).unflatten(0, token_timestep.shape)
        modulation = self.t_block(time_embedding)
        text, text_mask = self._grouped_text_condition(y, mask)
        if text.shape[1] != 1:
            raise ValueError(f"the one-view canvas carries ONE prompt row (G = 1), got G={text.shape[1]}")

        plans: dict[str, Optional[dict]] = {}
        for kind, subset in layout["subsets"].items():
            if subset is None:
                plans[kind] = None
                continue
            index = self._subset_index((*subset_key, kind), subset[0], device)
            full = int(index.numel()) == int(tokens.shape[1])
            rope = rope_softmax if kind == hm.SOFTMAX else rope_linear
            plans[kind] = {
                "index": index,
                "bounds": [int(b) for b in subset[1]],
                "tokens": int(index.numel()),
                "full": full,
                "modulation": modulation if full else modulation.index_select(2, index),
                "rope": rope if full else rope.index_select(2, index),
            }
        hidden = self._hm_trunk(tokens, text, modulation, text_mask, plans, caches)
        if robot is None:
            return {}
        video_tokens = self.final_layer(hidden[:, :n_video], time_embedding[:, :, :n_video])
        video_tokens = view_to_strip_tokens(video_tokens, self.f, view_shapes)
        action_pred = self.action_head(hidden[:, n_video + 1 :], time_embedding[:, :, n_video + 1 :]).masked_fill(
            ~action_mask, 0
        )
        return {"x": self.unpatchify(video_tokens), "action_pred": action_pred}


def build_hm_policy(
    config: PolicyConfig,
    contract: HMContract,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> HMPolicyModel:
    """Construct in eval mode, converted to ``dtype`` before any weight is copied in."""

    model = HMPolicyModel(config, contract)
    model.set_fp32_attention(config.fp32_attention)
    return model.to(device=device, dtype=dtype).eval()


__all__ = ["GDN_RECURRENCE_KEYS", "HMPolicyModel", "build_hm_policy"]
