"""Inference-only mirror of the chunk-causal policy (Sana ``sana_qwennext_policy_causal.py``, episode protocol:
``d37817788`` / ``fcccb4c88`` / ``3cbe3d85e`` and the deploy forms of ``4bd1b674c``, single form since ``4bd6eca83``:
no chunk observations; checked against rwm/zekai-merge ``7de7711c2``).

The model is the bidirectional policy trunk (``sana_wam_min.policy_model.PolicyModel``: embedders, AdaLN table,
Block AttnRes V2, text cross-attention, final layer, action head) with every block rebuilt around the chunk-causal
shells of ``layers.py``. It evaluates ONE deploy window per call, always a single segment, which is every form the
streaming session needs (Sana ``CausalPolicyChunkSession`` with ``forward_sink`` / ``forward_chunk``):

* the sink commit: the episode's first frame (one latent frame, no robot rows, t = 0) on video frame 0 -- the
  two-stream sink segment, which reads only itself and which every later forward reads;
* the denoising window of chunk c: [chunk-c targets] + [state | C actions] -- the two-stream noisy chunk c, which reads
  the session memory and takes its observation from the last frame of the most recent latent in it;
* the commit of chunk c at t = 0: [chunk-c targets] + [state | C actions] under ``memory_content`` video_and_action,
  the targets alone (no robot rows) under video_only -- the two-stream clean chunk c (chunk 0 the frozen first chunk),
  which reads like the generation of chunk c, then writes its GDN state and softmax K/V;
* under ``causal_observation`` (Sana 764ee2113): the observation prefill of chunk c, its own observation latent (one
  frame at ``c * f_chunk + 1``, t = 0; under ``causal_state`` with the chunk's state row alone, ``robot_form`` state),
  which reads like a denoising window and writes nothing to the memory; the denoising window then sits one frame later
  and its robot rows ``C / f_chunk`` steps later, the C action rows alone under ``causal_state`` (``robot_form``
  actions, at rows 1..C of the chunk grid).

Front-ends: the one-view canvas ``[B, C, F, H, W]`` (``sana_pixel``) and the three-view strip ``[B, C, F, 1,
sum(H_v * W_v)]`` (``sana_latent``), restored to view-major tokens (every view's frames in turn) before the trunk, the
robot rows after them (Sana ``_chunk_permutation`` of a single segment).

RoPE (``rope: aligned``): the bidirectional policy's video ids (the plain (t, y, x) grid for one view, the semantic 2x2
tiles for three, Sana ``_video_position_ids``) with the time column on the physical frame clock: latent frame phi at
``base_fps * phi / (fps / s)`` (s = the video frame stride), robot step n at ``base_fps * n / (8 * fps)`` with spatial
ids 0. The sink sits at frame 0, chunk c's targets at ``c * f_chunk + 1`` onwards (its observation instant
``c * f_chunk`` is the frame of chunk c - 1's last target, or of the sink), its state row at step ``c * C`` and its actions at steps
``c * C + 1 .. c * C + C`` (Sana ``_causal_multiview_rope``, ``forward_sink``, ``forward_chunk``). Text: one prompt row
(G = 1) for the canvas, G = V + 1 rows for the strip (one per view, then the robot tail; a robotless window reads the V
view rows only, Sana ``_grouped_query_sublens``).
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

from .contract import ACTION_TEMPORAL_COMPRESSION, MEMORY_VIDEO_AND_ACTION, CausalContract  # noqa: E402
from .layers import GDN_KIND, CausalPolicyBlock, LayerCache  # noqa: E402

GDN_RECURRENCE_KEYS = ("gate_proj.weight", "gate_proj.bias", "A_log", "dt_bias", "recall_gate")
# the robot rows of a window (Sana ``ROBOT_ROWS_STATE`` / ``ROBOT_ROWS_ACTIONS`` of a causal_state policy)
ROBOT_FORM_CHUNK = "chunk"        # the state row and the C action rows (every window without causal_state)
ROBOT_FORM_STATE = "state"        # the state row alone: a causal_state observation prefill
ROBOT_FORM_ACTIONS = "actions"    # the C action rows alone: a causal_state denoise window
ROBOT_FORMS = (ROBOT_FORM_CHUNK, ROBOT_FORM_STATE, ROBOT_FORM_ACTIONS)


ViewShapes = tuple[tuple[int, int], ...]


def view_token_counts(frames: int, view_shapes: ViewShapes) -> list[int]:
    """Tokens of every view's run in a window of ``frames`` latent frames (view-major order)."""

    return [int(frames) * int(height) * int(width) for height, width in view_shapes]


class CausalPolicyModel(PolicyModel):
    """The causal policy's deploy windows on the bidirectional trunk (one-view canvas or three-view strip, ``rope:
    aligned``)."""

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
        self.memory_content = str(contract.memory_content)
        self.aligned_phase_window = str(contract.aligned_phase_window)
        self.first_chunk = int(contract.first_chunk)   # the chunk an episode emits first (0: first-frame window)
        self.causal_observation = bool(contract.causal_observation)
        self.causal_state = bool(contract.causal_state)
        self.physical_rope = bool(contract.physical_rope)

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
                gdn_output_normalizer=contract.gdn_output_normalizer,
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
        view_shapes: ViewShapes,
        view_slot_ids: tuple[int, ...],
        robot_rows: int,
        frame_offset: int,
        step_offset: int,
        device: torch.device,
        robot_step_start: int = 0,
    ) -> torch.Tensor:
        """Rotary table ``[B, 1, sum_v F*H_v*W_v + robot_rows, D/2]`` of one window on the physical clock.

        Video tokens in view-major order with the bidirectional policy's spatial ids (the plain grid for one view, the
        semantic 2x2 tiles for several) and time ``base_fps * phi / (fps / s)``, phi = ``frame_offset + j`` for the
        window's frame j; robot row r: time ``base_fps * (step_offset + robot_step_start + r) / (8 * fps)`` and spatial
        ids 0 (Sana ``_causal_multiview_rope`` + ``_action_rope`` for one segment: row 0 is the state at the chunk's
        first step, row j + 1 action j; a window of the action rows alone starts at row 1, Sana ``_robot_grid``)."""

        batch = fps.numel()
        fps_video = fps.to(device=device, dtype=torch.float64) / int(self.video_frame_stride)
        if len(view_shapes) == 1:
            (height, width), = view_shapes
            ids = rope._position_ids(frames, int(height), int(width), fps_video, device)
        else:
            ids = semantic_2x2_position_ids(
                frames, view_shapes, view_slot_ids, fps_video, rope.base_fps, self.multiview_spatial_rope_tile_shape,
                device,
            )
        # physical_rope: the frame offset may be fractional (a target latent at the middle of its frames)
        phi = float(frame_offset) + torch.arange(frames, dtype=torch.float64, device=device)
        frame_of_token = torch.cat(
            [
                torch.arange(frames, device=device, dtype=torch.long).repeat_interleave(int(height) * int(width))
                for height, width in view_shapes
            ]
        )
        ids = ids.clone()
        ids[..., 0] = rope.base_fps * phi[frame_of_token][None] / fps_video[:, None]
        if robot_rows:
            first = int(step_offset) + int(robot_step_start)
            steps = (first + torch.arange(robot_rows, device=device, dtype=torch.long)).to(torch.float64)
            time_ids = rope.base_fps * steps[None] / (self.action_temporal_compression * fps.to(device=device)[:, None])
            robot = torch.stack((time_ids, torch.zeros_like(time_ids), torch.zeros_like(time_ids)), dim=-1)
            ids = torch.cat((ids, robot.expand(batch, -1, -1)), dim=1)
        return rope.from_position_ids(ids)

    def softmax_entry_rope(
        self,
        *,
        frames: int,
        frame_offset: float,
        robot_rows: int,
        step_offset: int,
        model_fps: Any,
        view_shapes: ViewShapes,
        view_slot_ids: tuple[int, ...],
        device,
    ) -> torch.Tensor:
        """Softmax rotary table of one committed entry, in its window's token order, at the read's positions: the sink
        (one frame, no rows) or a committed chunk (``f_c`` target frames from ``slot * f_c + 1``, then its state row
        at step ``slot * C`` and its C action rows under ``memory_content`` video_and_action)."""

        fps = PhysicalTimeWanRotaryPosEmbed._normalize_fps(model_fps, 1, device)
        return self.window_rope(
            self.rope_softmax, fps, int(frames), view_shapes, view_slot_ids, int(robot_rows), float(frame_offset),
            int(step_offset), device,
        )

    def target_frame_offset(self) -> float:
        """Where chunk c's target latents start on the video clock, after its start instant c * f_c: one frame later on
        the latent-slot clock (the latent's last frame), ``latent_centre_offset`` = (8 + 1) / (2 * 8) = 0.5625 later under
        ``physical_rope`` (the middle of the frames it covers; Sana ``causal_rwm_train.latent_centre_offset``)."""

        if not self.physical_rope:
            return 1.0
        stride = int(self.action_temporal_compression)
        return (stride + 1) / (2 * stride)

    def chunk_robot_rows(self) -> int:
        """Robot rows a committed chunk keeps: its state and C actions under video_and_action, none under video_only."""

        return 1 + self.chunk_actions if self.memory_content == MEMORY_VIDEO_AND_ACTION else 0

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

    def _window_text_spans(self, text, text_mask, view_shapes: ViewShapes, robot_rows: int, tokens: int):
        """(text, mask, query spans) of one window: G = 1 is one span over every token; G = V + 1 is one span per view
        run, then the robot tail, a robotless window reading the V view rows alone (Sana ``_grouped_query_sublens``:
        a robotless segment has no robot entry, so the robot group receives no query)."""

        groups = int(text.shape[1])
        if groups == 1:
            return text, text_mask, ((0, tokens),)
        views = len(view_shapes)
        if groups != views + 1:
            raise ValueError(f"a {views}-view window reads G = 1 or G = V + 1 = {views + 1} prompt rows, got G={groups}")
        spans, offset = [], 0
        for count in view_token_counts(self.f, view_shapes):
            spans.append((offset, count))
            offset += count
        if robot_rows:
            spans.append((offset, int(robot_rows)))
            offset += int(robot_rows)
        else:
            text = text[:, :views]
            text_mask = None if text_mask is None else text_mask[:, :views]
        if offset != tokens:
            raise ValueError(f"the text spans cover {offset} of {tokens} window tokens")
        return text, text_mask, tuple(spans)

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
        view_shapes: Optional[ViewShapes] = None,
        view_slot_ids: tuple[int, ...] = (0,),
        robot: Optional[Mapping[str, torch.Tensor]] = None,
        caches: Optional[Sequence[LayerCache]] = None,
        softmax_frame_offset: Optional[int] = None,
        softmax_step_offset: Optional[int] = None,
        robot_form: str = ROBOT_FORM_CHUNK,
    ) -> dict[str, torch.Tensor]:
        """Evaluate one single-segment window: a chunk's denoising window [targets] + robot rows, the commit of an
        executed chunk [targets] (+ robot rows under ``memory_content`` video_and_action), or the sink commit (one
        frame, no rows).

        Args:
            latent: the window's latent frames: ``[B, C, F, H, W]`` for one view (``view_shapes`` None or one entry),
                the view-major strip ``[B, C, F, 1, sum(H_v * W_v)]`` for several.
            view_shapes / view_slot_ids: per-view latent (H, W) and semantic 2x2 slot of a strip (None: the one-view
                grid of ``latent``).
            video_timesteps: ``[B, F]`` raw per-frame timesteps (0 for clean frames).
            y / mask: ``[B, G, 1, L, C_text]`` caption embeddings and ``[B, G, L]`` key mask (G = 1 for the canvas,
                G = V + 1 for the strip).
            model_fps: the batch fps (25 for RoboDojo).
            frame_offset / step_offset: physical index of the window's first latent frame and first robot step (the
                GDN layers' clock: their state keeps keys rotated at the physical positions of the whole episode):
                0 / 0 for the sink, ``c * f_c + 1`` / ``c * C`` for chunk c's denoising window and for its commit.
            softmax_frame_offset / softmax_step_offset: the same window on the softmax layers' clock, numbered from 0
                at the read's origin (the session's ``read_plan``; the cached entries are rotated on that clock too);
                None = the physical offsets (nothing slid out of the window yet).
            robot: None for a robotless window (the sink, the commit under ``memory_content`` video_only), else ``state80`` /
                ``state_mask80`` ``[B, 80]``, ``action80`` / ``action_mask80`` ``[B, C, 80]`` and ``action_timesteps``
                ``[B, C]`` (raw).
            caches: None for a window that reads and writes nothing, else one ``LayerCache`` per block.
            robot_form: which robot rows ``robot`` carries (``causal_state``, Sana ``robot_rows``): ``chunk`` the state
                row and the C action rows; ``state`` the state row alone (the observation prefill's group, t = 0);
                ``actions`` the C action rows alone (the denoise window, at rows 1..C of the chunk grid).

        Returns:
            ``{}`` for a robotless window or the state row alone (nothing reads its prediction); else ``{"x": velocity
            in ``latent``'s layout, "action_pred": [B, C, 80] velocity}``.
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
        if view_shapes is None or len(view_shapes) == 1:
            view_shapes = ((self.h, self.w),) if view_shapes is None else tuple((int(h), int(w)) for h, w in view_shapes)
            if view_shapes != ((self.h, self.w),):
                raise ValueError(f"a one-view window is the {view_shapes[0]} grid, got {(self.h, self.w)}")
            view_slot_ids = (0,)
        else:
            view_shapes = tuple((int(h), int(w)) for h, w in view_shapes)
            if self.h != 1 or self.w != sum(h * w for h, w in view_shapes):
                raise ValueError(f"a {len(view_shapes)}-view strip is [B, C, F, 1, {sum(h * w for h, w in view_shapes)}]")
            view_slot_ids = tuple(int(slot) for slot in view_slot_ids)
            if len(view_slot_ids) != len(view_shapes):
                raise ValueError(f"{len(view_shapes)} views need as many slot ids, got {view_slot_ids}")
        timesteps = torch.as_tensor(video_timesteps, device=device).reshape(batch, -1)
        if timesteps.shape[1] != self.f:
            raise ValueError(f"video_timesteps carries {timesteps.shape[1]} frames for a {self.f}-frame window")

        video = strip_to_view_tokens(self.x_embedder(x), self.f, view_shapes)
        n_video = video.shape[1]
        video_timestep = strip_to_view_tokens(
            self._to_model_t_domain(timesteps).repeat_interleave(self.h * self.w, dim=1), self.f, view_shapes
        )
        if robot_form not in ROBOT_FORMS:
            raise ValueError(f"robot_form must be one of {ROBOT_FORMS}, got {robot_form!r}")
        if robot_form != ROBOT_FORM_CHUNK and (robot is None or not self.causal_state):
            raise ValueError(f"robot_form {robot_form!r} is a causal_state window with robot rows")
        robot_step_start = 1 if robot_form == ROBOT_FORM_ACTIONS else 0
        action_mask = None
        if robot is None:
            tokens = video
            token_timestep = video_timestep.unsqueeze(1)
            robot_rows = 0
        else:
            parts, row_timesteps = [], []
            if robot_form != ROBOT_FORM_ACTIONS:
                state = torch.as_tensor(robot["state80"]).to(device).reshape(batch, -1)
                state_mask = torch.as_tensor(robot["state_mask80"]).to(device=device, dtype=torch.bool).reshape(batch, -1)
                parts.append(self.state_embed(state.to(dtype=video.dtype), state_mask).unsqueeze(1))
                row_timesteps.append(video_timestep.new_zeros(batch, 1))   # the state row is always clean
            if robot_form != ROBOT_FORM_STATE:
                action = torch.as_tensor(robot["action80"]).to(device)
                action_mask = torch.as_tensor(robot["action_mask80"]).to(device=device, dtype=torch.bool)
                action_steps = action.shape[1]
                if action_steps != self.chunk_actions:
                    raise ValueError(f"a window carries {self.chunk_actions} action rows, got {action_steps}")
                action_t = torch.as_tensor(robot["action_timesteps"], device=device).reshape(batch, -1)
                if action_t.shape[1] == 1:
                    action_t = action_t.expand(-1, action_steps)
                if bool((action_t != action_t[:, :1]).any()):
                    raise ValueError("the action rows of one chunk share one timestep")
                parts.append(self.action_embed(action.to(dtype=video.dtype), action_mask))
                row_timesteps.append(self._to_model_t_domain(action_t))
            robot_tokens = torch.cat(parts, dim=1)
            tokens = torch.cat((video, robot_tokens), dim=1)
            token_timestep = torch.cat([video_timestep] + row_timesteps, dim=1).unsqueeze(1)
            robot_rows = int(robot_tokens.shape[1])

        fps = PhysicalTimeWanRotaryPosEmbed._normalize_fps(model_fps, batch, device)
        rope_linear = self.window_rope(
            self.rope_linear, fps, self.f, view_shapes, view_slot_ids, robot_rows, frame_offset, step_offset, device,
            robot_step_start,
        )
        rope_softmax = self.window_rope(
            self.rope_softmax, fps, self.f, view_shapes, view_slot_ids, robot_rows,
            frame_offset if softmax_frame_offset is None else softmax_frame_offset,
            step_offset if softmax_step_offset is None else softmax_step_offset,
            device, robot_step_start,
        )
        time_embedding = self.t_embedder(token_timestep.flatten()).unflatten(0, token_timestep.shape)
        modulation = self.t_block(time_embedding)
        text, text_mask = self._grouped_text_condition(y, mask)
        text, text_mask, spans = self._window_text_spans(text, text_mask, view_shapes, robot_rows, int(tokens.shape[1]))
        tokens = self._causal_trunk(
            tokens,
            text,
            modulation,
            text_mask,
            spans,
            rope_linear,
            rope_softmax,
            caches,
        )
        if robot is None or robot_form == ROBOT_FORM_STATE:
            return {}
        video_tokens = self.final_layer(tokens[:, :n_video], time_embedding[:, :, :n_video])
        video_tokens = view_to_strip_tokens(video_tokens, self.f, view_shapes)
        first_action = n_video + (0 if robot_form == ROBOT_FORM_ACTIONS else 1)
        action_pred = self.action_head(tokens[:, first_action:], time_embedding[:, :, first_action:]).masked_fill(
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


__all__ = [
    "CausalPolicyModel",
    "GDN_RECURRENCE_KEYS",
    "ROBOT_FORMS",
    "ROBOT_FORM_ACTIONS",
    "ROBOT_FORM_CHUNK",
    "ROBOT_FORM_STATE",
    "ViewShapes",
    "build_causal_policy",
    "view_token_counts",
]
