"""Self-contained, inference-only mirror of the unified policy branch."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .attnres import BlockAttnResV2
from .block import PolicyBlock
from .checkpoint import load_policy_state_dict
from .config import PolicyConfig
from .embeddings import (
    ActionFinalLayer,
    CaptionEmbedder,
    MaskedProjector,
    PatchEmbedMS3D,
    RMSNorm,
    T2IFinalLayer,
    TimestepEmbedder,
)
from .geometry import strip_to_view_tokens, view_to_strip_tokens
from .rope import (
    PhysicalTimeWanRotaryPosEmbed,
    WanRotaryPosEmbed,
    semantic_2x2_position_ids,
)


def camera_conditioning_enabled(data_info: dict) -> bool:
    """The live batch-uniform camera-geometry gate
    (sana_qwennext_camera_condition.py camera_conditioning_enabled)."""

    value = data_info.get("camera_conditioning_enabled")
    if value is None:
        return "first_frame_plucker" in data_info
    values = torch.as_tensor(value).reshape(-1)
    if bool((values != values[0]).any()):
        raise ValueError("camera_conditioning_enabled must be uniform within a batch")
    return bool(values[0].item())


def _view_count(data_info: dict) -> int:
    counts = torch.as_tensor(data_info["view_count"]).reshape(-1)
    if bool((counts != counts[0]).any()):
        raise ValueError("all scenes in a multiview batch must have the same V")
    return int(counts[0].item())


def _view_shapes(data_info: dict) -> tuple[tuple[int, int], ...]:
    value = data_info["view_latent_shape"]
    if isinstance(value, torch.Tensor):
        if value.ndim == 3:
            value = value[0]
        value = value.tolist()
    return tuple((int(height), int(width)) for height, width in value)


def _view_slot_ids(data_info: dict) -> tuple[int, ...]:
    value = torch.as_tensor(data_info.get("view_slot_ids"))
    if value.ndim == 2:
        if bool((value != value[0]).any()):
            raise ValueError("all scenes in a batch must use the same view_slot_ids")
        value = value[0]
    slots = tuple(int(slot) for slot in value.tolist())
    # A duplicate or out-of-range slot would place wrong RoPE tiles silently.
    if len(set(slots)) != len(slots) or any(slot < 0 or slot > 3 for slot in slots):
        raise ValueError(f"view_slot_ids must be unique IDs in [0,3], got {slots}")
    return slots


class PolicyModel(nn.Module):
    """The unified world model's policy branch as one flat inference graph.

    Token sequence: per-view video blocks in view order, one state token, one
    token per 80D action row.  Text arrives as ``V+1`` token groups, one per
    view plus one for the robot tail, each cross-attended by its own query
    span.  The camera channel is not modeled: ``plucker_embed`` is absent and
    a batch with the camera gate on is refused.  Checkpoint tensors the
    mirror does not model are removed by :func:`load_policy_state_dict`.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        super().__init__()
        config = (config or PolicyConfig()).validate()
        self.policy_config = config
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
        self.multiview_spatial_rope_layout = config.multiview_spatial_rope_layout
        self.multiview_spatial_rope_tile_shape = tuple(
            config.multiview_spatial_rope_tile_shape
        )
        self.use_xformers_cross_attention = False

        self.x_embedder = PatchEmbedMS3D(
            config.patch_size,
            config.in_channels,
            config.hidden_size,
        )
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )
        self.y_embedder = CaptionEmbedder(
            config.caption_channels,
            config.hidden_size,
        )
        self.attention_y_norm = RMSNorm(
            config.hidden_size,
            scale_factor=config.y_norm_scale_factor,
            eps=config.norm_eps,
        )

        self.rope_linear = PhysicalTimeWanRotaryPosEmbed(
            WanRotaryPosEmbed(
                config.linear_head_dim,
                max_seq_len=1024,
            ),
            attention_head_dim=config.linear_head_dim,
        )
        self.rope_softmax = PhysicalTimeWanRotaryPosEmbed(
            WanRotaryPosEmbed(
                config.softmax_head_dim,
                max_seq_len=1024,
            ),
            attention_head_dim=config.softmax_head_dim,
        )

        self.blocks = nn.ModuleList(
            PolicyBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                attention_kind=kind,
                linear_head_dim=config.linear_head_dim,
                softmax_head_dim=config.softmax_head_dim,
                fp32_attention=config.fp32_attention,
            )
            for kind in self.block_attn_types
        )
        self.set_cross_attention_xformers(True)
        self.attn_res = BlockAttnResV2(
            config.hidden_size,
            config.depth,
            config.attn_res_block_size,
        )
        self.final_layer = T2IFinalLayer(
            config.hidden_size,
            config.patch_size,
            config.out_channels,
        )
        self.state_embed = MaskedProjector(config.state_dim, config.hidden_size)
        self.action_embed = MaskedProjector(
            config.action_dim, config.hidden_size
        )
        self.action_head = ActionFinalLayer(config.hidden_size, config.action_dim)

        self.f = self.h = self.w = 0
        self._initialize_weights()
        self.eval()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        nn.init.zeros_(self.attn_res.attn_proj.weight)
        nn.init.zeros_(self.attn_res.mlp_proj.weight)
        nn.init.zeros_(self.attn_res.final_proj.weight)
        nn.init.zeros_(self.state_embed.proj.weight)
        nn.init.zeros_(self.state_embed.proj.bias)
        nn.init.zeros_(self.action_embed.proj.weight)
        nn.init.zeros_(self.action_embed.proj.bias)
        nn.init.zeros_(self.action_head.linear.weight)
        nn.init.zeros_(self.action_head.linear.bias)

    def set_cross_attention_xformers(self, enabled: bool) -> None:
        for block in self.blocks:
            block.cross_attn.set_use_xformers(enabled)
        self.use_xformers_cross_attention = all(
            block.cross_attn.use_xformers for block in self.blocks
        )

    def set_fp32_attention(self, enabled: bool) -> None:
        for block in self.blocks:
            block.attn.fp32_attention = bool(enabled)

    def load_checkpoint_state(self, state_dict) -> None:
        load_policy_state_dict(self, state_dict)

    def _to_model_t_domain(self, value: torch.Tensor) -> torch.Tensor:
        if self.timestep_norm_scale_factor != 1.0:
            return value.float() / self.timestep_norm_scale_factor
        return value.long().float()

    @staticmethod
    def _assert_lockstep(
        timestep: torch.Tensor, data_info: dict, latent_frames: int
    ) -> None:
        """The task=ltx lockstep diagonal in the raw domain: frame 0 clean, one
        shared scalar t over the noisy frames, action schedule equal to it
        (sana_qwennext_action_policy.py _assert_policy_lockstep)."""

        t = torch.as_tensor(timestep).reshape(timestep.shape[0], -1)
        if latent_frames < 2 or t.shape[1] != latent_frames:
            raise ValueError(
                "the policy branch requires the frame-indexed timestep format "
                f"([B, F] or [B, 1, F] with F={latent_frames} > 1, frame 0 "
                f"clean); got {tuple(timestep.shape)}."
            )
        if not bool((t[:, 0] == 0).all()):
            raise ValueError(
                "policy requires a clean first visual frame (frame-0 timestep "
                "must be 0)."
            )
        if not bool((t[:, 1:] == t[:, -1:]).all()):
            raise ValueError(
                "lockstep violated: noisy frames carry heterogeneous timesteps."
            )
        action_timestep = data_info.get("action_timestep")
        if not isinstance(action_timestep, torch.Tensor):
            raise KeyError(
                "the policy branch requires data_info['action_timestep'] "
                "([B, T] or per-batch scalar, raw domain)."
            )
        action_rows = action_timestep.reshape(t.shape[0], -1)
        if not bool((action_rows == t[:, -1:]).all()):
            raise ValueError(
                "lockstep violated: supplied action_timestep differs from the "
                "noisy video t."
            )

    def _robot_tokens(
        self, data_info: dict, video: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """State token followed by one token per 80D action row
        (sana_qwennext_pretrain.py _robot_tokens)."""

        state = data_info["initial_state80"].to(video.device)
        state_mask = data_info["initial_state_condition_mask80"].to(video.device)
        action = data_info["action80"].to(video.device)
        action_mask = data_info["action_mask80"].to(video.device)

        action_steps = action.shape[1]
        expected_steps = (self.f - 1) * self.action_temporal_compression
        if action_steps != expected_steps:
            raise ValueError(
                "one motion80 row per source-frame transition is required: "
                f"got {action_steps}, expected {expected_steps} for latent F={self.f}"
            )

        state = state.to(dtype=video.dtype)
        action = action.to(dtype=video.dtype)
        return torch.cat(
            (
                self.state_embed(state, state_mask).unsqueeze(1),
                self.action_embed(action, action_mask),
            ),
            dim=1,
        ), action_steps

    def _action_timesteps(
        self, data_info: dict, batch: int, action_steps: int, device
    ) -> torch.Tensor:
        value = torch.as_tensor(data_info["action_timestep"])
        value = value.to(device=device).reshape(batch, -1)
        if value.shape[1] == 1:
            value = value.expand(-1, action_steps)
        return value

    def _robot_rope(
        self,
        rope: PhysicalTimeWanRotaryPosEmbed,
        data_info: dict,
        fps: torch.Tensor,
        view_shapes: tuple[tuple[int, int], ...],
        action_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Video frequencies followed by the robot clock (state at t=0, action
        row k at ``k / (compression * fps)`` in base-fps frame units).

        V=1 is the regular grid (sana_qwennext_pretrain.py _unified_rope); V>1
        follows the configured multiview spatial layout
        (sana_qwennext_multiview.py _multiview_robot_rope).
        """

        batch = fps.numel()
        if len(view_shapes) == 1:
            video = rope((self.f, self.h, self.w), device).expand(batch, -1, -1, -1)
        elif self.multiview_spatial_rope_layout == "semantic_2x2":
            video = rope.from_position_ids(
                semantic_2x2_position_ids(
                    self.f,
                    view_shapes,
                    _view_slot_ids(data_info),
                    fps,
                    rope.base_fps,
                    self.multiview_spatial_rope_tile_shape,
                    device,
                )
            )
        else:
            video = torch.cat(
                [
                    rope((self.f, height, width), device)
                    for height, width in view_shapes
                ],
                dim=2,
            ).expand(batch, -1, -1, -1)
        action_ids = torch.arange(
            1, action_steps + 1, device=device, dtype=torch.float64
        )
        action_time = (
            rope.base_fps
            * action_ids[None]
            / (self.action_temporal_compression * fps[:, None])
        )
        condition_time = torch.cat(
            (torch.zeros(batch, 1, device=device), action_time), dim=1
        )
        condition_ids = torch.stack(
            (
                condition_time,
                torch.zeros_like(condition_time),
                torch.zeros_like(condition_time),
            ),
            dim=-1,
        )
        return torch.cat((video, rope.from_position_ids(condition_ids)), dim=2)

    def _grouped_text_condition(
        self, y: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-group projected text ``[B,G,L,C]`` and its ``[B,G,L]`` int16 mask
        (sana_qwennext_camera_condition.py _grouped_text_condition)."""

        batch, groups, _, length, channels = y.shape
        y = self.y_embedder(
            y.reshape(batch * groups, 1, length, channels)
        ).reshape(batch, groups, 1, length, self.hidden_size).squeeze(2)
        y = self.attention_y_norm(y)
        if mask is None:
            return y, None
        if mask.shape[:2] != y.shape[:2]:
            raise ValueError("token-group text mask must start with [B,G]")
        if mask.shape[-1] != y.shape[-2]:
            raise ValueError("token-group text mask length differs from embeddings")
        return y, mask.to(torch.int16).reshape(batch, groups, -1)

    @staticmethod
    def _prompt_group_spans(
        view_token_counts, action_steps: int
    ) -> tuple[tuple[int, int], ...]:
        """Static (offset, length) query span per text group: one per view
        block in view order, then the robot tail (state row + action rows)."""

        spans = []
        offset = 0
        for count in view_token_counts:
            spans.append((offset, int(count)))
            offset += int(count)
        spans.append((offset, 1 + int(action_steps)))
        return tuple(spans)

    @torch.no_grad()
    def _forward_trunk(
        self,
        tokens: torch.Tensor,
        text: torch.Tensor,
        modulation: torch.Tensor,
        text_mask,
        prompt_group_spans,
        rope_linear,
        rope_softmax,
    ) -> torch.Tensor:
        block_size = self.attn_res_block_size
        num_blocks = math.ceil(len(self.blocks) / block_size)
        value_buffer = torch.empty(
            (num_blocks + 1, *tokens.shape),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        key_buffer = torch.empty_like(value_buffer)
        value_buffer[0] = tokens
        key_buffer[0] = self.attn_res.key_norm(tokens.unsqueeze(0)).squeeze(0)
        n_active = 1
        partial = None

        for block_index in range(num_blocks):
            start = block_index * block_size
            end = min(start + block_size, len(self.blocks))
            for layer_index in range(start, end):
                rope = (
                    rope_softmax
                    if self.block_attn_types[layer_index]
                    == self.softmax_attn_type
                    else rope_linear
                )
                hidden = self.attn_res._attend_buffer(
                    self.attn_res.attn_proj,
                    value_buffer,
                    key_buffer,
                    n_active,
                    partial,
                )
                attn_delta = self.blocks[layer_index].forward_attn_sublayer(
                    hidden,
                    text,
                    modulation,
                    text_mask=text_mask,
                    rotary_emb=rope,
                    prompt_group_spans=prompt_group_spans,
                )
                partial = attn_delta if partial is None else partial + attn_delta
                hidden = self.attn_res._attend_buffer(
                    self.attn_res.mlp_proj,
                    value_buffer,
                    key_buffer,
                    n_active,
                    partial,
                )
                partial = partial + self.blocks[
                    layer_index
                ].forward_mlp_sublayer(hidden, modulation)

            value_buffer[n_active] = partial
            key_buffer[n_active] = self.attn_res.key_norm(
                partial.unsqueeze(0)
            ).squeeze(0)
            n_active += 1
            partial = None

        return self.attn_res._attend_buffer(
            self.attn_res.final_proj,
            value_buffer,
            key_buffer,
            n_active,
            None,
        )

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_f, patch_h, patch_w = self.x_embedder.patch_size
        tokens = tokens.reshape(
            tokens.shape[0],
            self.f,
            self.h,
            self.w,
            patch_f,
            patch_h,
            patch_w,
            self.out_channels,
        )
        tokens = torch.einsum("nfhwopqc->ncfohpwq", tokens)
        return tokens.reshape(
            tokens.shape[0],
            self.out_channels,
            self.f * patch_f,
            self.h * patch_h,
            self.w * patch_w,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Unified dialect: the robot stream rides ``data_info``.

        ``x`` is ``[B,C,F,H,W]`` for V=1 or the strip ``[B,C,F,1,sum(H_v*W_v)]``
        for V>1; ``y`` is ``[B,G=V+1,1,L,C]`` with ``mask`` ``[B,G,L]``;
        ``timestep`` is ``[B,1,F]`` in the raw domain.
        """

        for retired in ("action", "action_mask"):
            if retired in kwargs:
                raise TypeError(
                    f"the unified policy branch takes no explicit {retired!r} "
                    "argument — the robot stream rides "
                    "data_info['action80'/'action_mask80'] (unified dialect)."
                )
        if self.training:
            raise RuntimeError("kernel policy is inference-only; call eval()")
        data_info = kwargs.get("data_info") or {}
        task = data_info.get("rwm_task")
        if task != "policy":
            raise ValueError(
                "the unified policy branch serves rwm_task='policy' only; got "
                f"{task!r}."
            )
        if "view_count" not in data_info:
            raise KeyError(
                "the unified policy branch requires data_info['view_count']."
            )
        if camera_conditioning_enabled(data_info):
            raise NotImplementedError(
                "the kernel policy mirror does not model the camera channel "
                "(first_frame_plucker / camera_conditioning_enabled)."
            )

        batch = x.shape[0]
        num_views = _view_count(data_info)
        x = x.to(self.dtype)
        y = y.to(self.dtype)
        self.f, self.h, self.w = (
            x.shape[-3] // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )
        self._assert_lockstep(timestep, data_info, self.f)
        timestep = self._to_model_t_domain(timestep)
        view_shapes = (
            ((self.h, self.w),) if num_views == 1 else _view_shapes(data_info)
        )

        video = strip_to_view_tokens(self.x_embedder(x), self.f, view_shapes)
        view_token_counts = [
            self.f * height * width for height, width in view_shapes
        ]
        n_video = video.shape[1]
        robot, action_steps = self._robot_tokens(data_info, video)
        tokens = torch.cat((video, robot), dim=1)

        video_timestep = strip_to_view_tokens(
            timestep.reshape(batch, -1).repeat_interleave(self.h * self.w, dim=1),
            self.f,
            view_shapes,
        )
        action_timestep = self._to_model_t_domain(
            self._action_timesteps(data_info, batch, action_steps, x.device)
        )
        token_timestep = torch.cat(
            (
                video_timestep,
                video_timestep.new_zeros(batch, 1),
                action_timestep,
            ),
            dim=1,
        ).unsqueeze(1)

        fps = PhysicalTimeWanRotaryPosEmbed._normalize_fps(
            data_info["model_fps"], batch, x.device
        )
        with self.rope_linear.use_model_fps(
            fps, batch_size=batch, device=x.device
        ), self.rope_softmax.use_model_fps(
            fps, batch_size=batch, device=x.device
        ):
            rope_linear = self._robot_rope(
                self.rope_linear, data_info, fps, view_shapes, action_steps, x.device
            )
            rope_softmax = self._robot_rope(
                self.rope_softmax, data_info, fps, view_shapes, action_steps, x.device
            )

        time_embedding = self.t_embedder(token_timestep.flatten()).unflatten(
            0, token_timestep.shape
        )
        modulation = self.t_block(time_embedding)
        text, text_mask = self._grouped_text_condition(y, mask)
        tokens = self._forward_trunk(
            tokens,
            text,
            modulation,
            text_mask,
            self._prompt_group_spans(view_token_counts, action_steps),
            rope_linear,
            rope_softmax,
        )

        video_tokens = self.final_layer(
            tokens[:, :n_video], time_embedding[:, :, :n_video]
        )
        video_tokens = view_to_strip_tokens(video_tokens, self.f, view_shapes)
        action_mask = data_info["action_mask80"].to(tokens.device)
        action_pred = self.action_head(
            tokens[:, n_video + 1 :], time_embedding[:, :, n_video + 1 :]
        ).masked_fill(~action_mask, 0)
        return {
            "x": self.unpatchify(video_tokens),
            "action_pred": action_pred,
        }


def build_policy(
    config: PolicyConfig | None = None,
    **builder_kwargs: Any,
) -> PolicyModel:
    if isinstance(config, PolicyConfig):
        if builder_kwargs:
            raise ValueError("pass PolicyConfig or the live builder kwargs, not both")
        resolved = config
    else:
        if config is not None:
            builder_kwargs["config"] = config
        resolved = PolicyConfig.from_sana_kwargs(**builder_kwargs)
    model = PolicyModel(resolved)
    model.set_fp32_attention(resolved.fp32_attention)
    return model


def SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy_5B_P1_D36(
    **kwargs: Any,
) -> PolicyModel:
    kwargs = dict(kwargs)
    kwargs.update(
        depth=32,
        hidden_size=2560,
        patch_size=(1, 1, 1),
        num_heads=20,
    )
    return build_policy(**kwargs)


__all__ = [
    "PolicyModel",
    "build_policy",
    "camera_conditioning_enabled",
    "SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy_5B_P1_D36",
]
