"""Inference-only mirror of Sana's bidirectional ``SanaRWMMoTAttnResPolicy``.

Two independently parameterized experts -- the Sana-Video AttnRes trunk (``video_dit``) and the fresh action expert
(``action_dit``) -- coupled at every layer by ONE joint self-attention over the concatenated video and robot tokens.
Two checkpoint layouts are served (``MoTConfig.context_layout``, detected from the state dict at load):

* ``context_embedder`` / ``shared_caption_embedder`` -- rwm/mot @ b3b9e0e9e .. 71ac93f43 (``sana_qwennext_mot_policy.py`` +
  ``mot_context_embedder.py``): the text of both experts comes from ``ContextEmbedder`` -- one caption projection +
  y-norm per expert width, or (the canvas modes since 4e67e1e1d) ONE projection whose 2560-wide output both experts
  read. The video stream is the 2x2 multiview strip (``video_layout='multiview'`` = ``data.extra.multiview:
  sana_latent``: V views packed as ``[B, C, F, 1, sum(h*w)]``, view-major tokens, ``semantic_2x2`` RoPE tiles, text
  group g routed to view g, the robot group to the action expert) or ONE composited canvas on its native grid
  (``openwam_canvas`` / ``sana_pixel_canvas``: V = 1, one or -- 2026-09-21..22 -- two prompt rows). A strided video
  (``data_info['video_frame_stride']`` s > 1) carries ``(F - 1) * 8 * s`` action rows. RoPE (``MoTConfig.rope``):
  ``aligned`` folds s into the video clock and puts the robot rows on it through video_dit's RoPE modules;
  ``independent`` keeps the video clock and gives the action expert its own 1D clock (state 0, actions 1 .. S; 0 .. S-1
  under state_as_context); ``legacy`` = the tables of every run before rwm/mot 42aee4fa9 (physical clock at stride 1, a
  zero-phase state plus actions 0 .. S-1 above it).
* ``legacy_action_mlp`` -- rwm/mot @ 606e48dd9: the video expert owns ``y_embedder`` + ``attention_y_norm``, the action
  expert its own ``context_mlp`` over the raw prompt; canvas only (the layout predates multiview and strides).

The AttnRes aggregation runs on the vendored preallocated buffers (the inference path of ``BlockAttnResV2``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sana_wam_min.policy_model.config import PolicyConfig
from sana_wam_min.policy_model.geometry import strip_to_view_tokens, view_to_strip_tokens
from sana_wam_min.policy_model.rope import PhysicalTimeWanRotaryPosEmbed, semantic_2x2_position_ids

from .action_dit import ACTION_TEMPORAL_COMPRESSION, ActionDiT, independent_action_rope
from .block import Contexts, MoTLayer
from .checkpoint import CONTEXT_EMBEDDER, LEGACY_ACTION_MLP, SHARED_CAPTION_EMBEDDER, load_mot_state_dict
from .context import ContextEmbedder
from .video_dit import VideoExpert, final_layer_frame_aware

MOT_FACTORY_NAME = "SanaRWMMoTAttnResPolicy_5B_P1_D36"
# video_layout (the adapter's name for the front-end; data.extra.multiview in Sana since 2026-09-21):
# multiview = sana_latent (the packed strip), openwam_canvas = openwam, sana_pixel_canvas = sana_pixel.
VIDEO_LAYOUTS = ("multiview", "openwam_canvas", "sana_pixel_canvas")
CANVAS_VIDEO_LAYOUTS = ("openwam_canvas", "sana_pixel_canvas")
MULTIVIEW_ROPE_LAYOUTS = ("local_reset", "semantic_2x2")
CONTEXT_LAYOUTS = (CONTEXT_EMBEDDER, SHARED_CAPTION_EMBEDDER, LEGACY_ACTION_MLP)
# RoPE regimes: model.extra.rope (rwm/mot 71ac93f43) aligned | independent, or the undeclared tables of a run's era
# (legacy = before 42aee4fa9; the runs between 42aee4fa9 and 71ac93f43 trained the independent table).
ROPE_LEGACY = "legacy"
ROPE_ALIGNED = "aligned"
ROPE_INDEPENDENT = "independent"
MOT_ROPE_MODES = (ROPE_LEGACY, ROPE_ALIGNED, ROPE_INDEPENDENT)


@dataclass(frozen=True)
class MoTConfig:
    """The video trunk's :class:`PolicyConfig` plus the MoT knobs of ``model.extra`` (constructor names) and the checkpoint layout."""

    video: PolicyConfig
    action_hidden_size: int = 1024
    action_mlp_ratio: float = 4.0
    action_cross_attn_heads: int = 8
    action_rope_theta: float = 10000.0
    action_attn_res_block_size: int = 8
    action_state_as_context: bool = False
    video_layout: str = "multiview"
    multiview_spatial_rope_layout: str = "semantic_2x2"
    multiview_spatial_rope_tile_shape: tuple[int, int] = (15, 30)
    context_layout: str = CONTEXT_EMBEDDER
    # prompt rows of a canvas: 1 = ONE row both experts read (before 2026-09-21, and since 2026-09-22), 2 = the composite
    # view row for the video expert + the robot row for the action expert (rwm/mot a26807821 .. 2b4a4dc8d)
    canvas_text_groups: int = 1
    rope: str = ROPE_LEGACY
    # scheduler.action_flow_shift / inference_action_flow_shift declared: the action rows take their own schedule's t
    separate_action_schedule: bool = False

    @property
    def legacy(self) -> bool:
        return self.context_layout == LEGACY_ACTION_MLP

    @property
    def canvas(self) -> bool:
        return self.video_layout in CANVAS_VIDEO_LAYOUTS

    @property
    def shared_caption_embedder(self) -> bool:
        return self.context_layout == SHARED_CAPTION_EMBEDDER

    def validate(self) -> "MoTConfig":
        self.video.validate()
        if self.action_hidden_size <= 0 or self.action_hidden_size % self.action_cross_attn_heads:
            raise ValueError("action_hidden_size must be positive and divisible by action_cross_attn_heads")
        if self.action_mlp_ratio <= 0 or self.action_rope_theta <= 0:
            raise ValueError("action_mlp_ratio and action_rope_theta must be positive")
        if self.action_attn_res_block_size != self.video.attn_res_block_size:
            raise ValueError(
                f"video AttnRes block_size ({self.video.attn_res_block_size}) must equal "
                f"action_attn_res_block_size ({self.action_attn_res_block_size}) for the two experts' group boundaries "
                "to land on the same layer indices."
            )
        if self.video_layout not in VIDEO_LAYOUTS:
            raise ValueError(f"video_layout must be one of {VIDEO_LAYOUTS}, got {self.video_layout!r}")
        if self.multiview_spatial_rope_layout not in MULTIVIEW_ROPE_LAYOUTS:
            raise ValueError(
                f"multiview_spatial_rope_layout must be one of {MULTIVIEW_ROPE_LAYOUTS}, got {self.multiview_spatial_rope_layout!r}"
            )
        tile = tuple(self.multiview_spatial_rope_tile_shape)
        if len(tile) != 2 or min(int(v) for v in tile) <= 0:
            raise ValueError(f"multiview_spatial_rope_tile_shape must be two positive integers, got {tile}")
        if self.context_layout not in CONTEXT_LAYOUTS:
            raise ValueError(f"context_layout must be one of {CONTEXT_LAYOUTS}, got {self.context_layout!r}")
        if self.legacy and self.video_layout != "openwam_canvas":
            raise ValueError(
                "the legacy MoT layout (rwm/mot 606e48dd9, action_dit.context_mlp) existed only for the OpenWAM canvas; "
                f"got video_layout={self.video_layout!r}"
            )
        if self.shared_caption_embedder and not self.canvas:
            raise ValueError(
                "the shared caption embedder (rwm/mot 4e67e1e1d) is built for the canvas modes only; "
                f"got video_layout={self.video_layout!r}"
            )
        if int(self.canvas_text_groups) not in (1, 2):
            raise ValueError(f"canvas_text_groups must be 1 or 2, got {self.canvas_text_groups!r}")
        if int(self.canvas_text_groups) != 1 and (self.shared_caption_embedder or self.legacy):
            raise ValueError("the shared caption embedder and the legacy layout read ONE prompt row (canvas_text_groups 1)")
        if self.rope not in MOT_ROPE_MODES:
            raise ValueError(f"rope must be one of {MOT_ROPE_MODES}, got {self.rope!r}")
        if self.legacy and self.rope != ROPE_LEGACY:
            raise ValueError("the legacy MoT layout (606e48dd9) predates the RoPE modes")
        return self


# -- data_info parsing (ports of sana_qwennext_multiview.py / sana_qwennext_pretrain.py / the camera gate) ---------------


def camera_conditioning_enabled(data_info: dict) -> bool:
    """The batch-uniform camera-geometry gate (``sana_qwennext_camera_condition.camera_conditioning_enabled``)."""

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
    """Per-view RoPE slot ids, unique in [0, 3] (a duplicate or out-of-range slot would place wrong tiles silently)."""

    value = torch.as_tensor(data_info.get("view_slot_ids"))
    if value.ndim == 2:
        if bool((value != value[0]).any()):
            raise ValueError("all scenes in a batch must use the same view_slot_ids")
        value = value[0]
    slots = tuple(int(slot) for slot in value.tolist())
    if len(set(slots)) != len(slots) or any(slot < 0 or slot > 3 for slot in slots):
        raise ValueError(f"view_slot_ids must be unique IDs in [0,3], got {slots}")
    return slots


def _video_frame_stride(data_info: dict) -> int:
    """The batch's video frame stride (video frame j is source row j * stride), 1 when the key is absent."""

    value = data_info.get("video_frame_stride")
    if value is None:
        return 1
    strides = torch.as_tensor(value).reshape(-1)
    if bool((strides != strides[0]).any()):
        raise ValueError("video_frame_stride must be uniform within a batch")
    stride = int(strides[0].item())
    if stride < 1:
        raise ValueError(f"video_frame_stride must be >= 1, got {stride}")
    return stride


def _assert_lockstep(timestep: torch.Tensor, data_info: dict, latent_frames: int, separate_action_schedule: bool = False) -> None:
    """The policy lockstep in the raw domain: frame 0 clean, one shared t over the noisy frames, action rows equal to it
    (``sana_qwennext_action_policy._assert_policy_lockstep`` without the training-time RTC prefix); with a separate action
    schedule (``scheduler.action_flow_shift``, rwm/mot b865ad732) the action rows share one t of their own."""

    t = torch.as_tensor(timestep).reshape(timestep.shape[0], -1)
    if latent_frames < 2 or t.shape[1] != latent_frames:
        raise ValueError(
            "the policy branch requires the frame-indexed timestep format "
            f"([B, F] or [B, 1, F] with F={latent_frames} > 1, frame 0 clean); got {tuple(timestep.shape)}."
        )
    if not bool((t[:, 0] == 0).all()):
        raise ValueError("policy requires a clean first visual frame (frame-0 timestep must be 0).")
    if not bool((t[:, 1:] == t[:, -1:]).all()):
        raise ValueError("lockstep violated: noisy frames carry heterogeneous timesteps.")
    action_timestep = data_info.get("action_timestep")
    if not isinstance(action_timestep, torch.Tensor):
        raise KeyError("the policy branch requires data_info['action_timestep'] ([B, T] or per-batch scalar, raw domain).")
    action_rows = action_timestep.reshape(t.shape[0], -1)
    shared = action_rows[:, -1:] if separate_action_schedule else t[:, -1:]
    if not bool((action_rows == shared).all()):
        raise ValueError("lockstep violated: supplied action_timestep differs from the noisy video t.")


def _to_model_t_domain(value: torch.Tensor, timestep_norm_scale_factor: float) -> torch.Tensor:
    if timestep_norm_scale_factor != 1.0:
        return value.float() / timestep_norm_scale_factor
    return value.long().to(torch.float32)


# -- RoPE tables ---------------------------------------------------------------------------------------------------


def _video_rope_tables(video_dit: VideoExpert, model_fps, batch: int, device, frame_stride: int = 1):
    """Video RoPE tables (linear-layer heads, softmax-layer heads) for the current native ``(f, h, w)`` grid under
    ``model_fps``; ``frame_stride`` > 1 folds the video frame stride into the time axis (``rope: aligned``)."""

    grid = (video_dit.f, video_dit.h, video_dit.w)
    with video_dit.rope_linear.use_model_fps(model_fps, batch_size=batch, device=device):
        with video_dit.rope_softmax.use_model_fps(model_fps, batch_size=batch, device=device):
            return (
                video_dit.rope_linear(grid, device, frame_stride=frame_stride),
                video_dit.rope_softmax(grid, device, frame_stride=frame_stride),
            )


def _multiview_rope_tables(
    video_dit: VideoExpert,
    model_fps,
    batch: int,
    device,
    frames: int,
    view_shapes,
    view_slot_ids,
    rope_layout: str,
    tile_shape,
    video_clock_stride: int = 1,
):
    """Video RoPE tables for view-major multiview tokens: ``semantic_2x2`` tiles (the only layout since 2026-09-21) or a
    pre-2026-09-21 yaml's ``local_reset`` per-view grids; ``video_clock_stride`` > 1 folds the video frame stride into the
    time axis (``rope: aligned``)."""

    tables = []
    with video_dit.rope_linear.use_model_fps(model_fps, batch_size=batch, device=device):
        with video_dit.rope_softmax.use_model_fps(model_fps, batch_size=batch, device=device):
            for rope in (video_dit.rope_linear, video_dit.rope_softmax):
                if rope_layout == "semantic_2x2":
                    position_ids = semantic_2x2_position_ids(
                        frames=frames,
                        view_shapes=view_shapes,
                        view_slot_ids=view_slot_ids,
                        fps=PhysicalTimeWanRotaryPosEmbed._normalize_fps(model_fps, batch, device) / int(video_clock_stride),
                        base_fps=rope.base_fps,
                        tile_shape=tile_shape,
                        device=device,
                    )
                    tables.append(rope.from_position_ids(position_ids))
                else:
                    tables.append(
                        torch.cat(
                            [rope((frames, height, width), device, frame_stride=video_clock_stride) for height, width in view_shapes],
                            dim=2,
                        ).expand(batch, -1, -1, -1)
                    )
    return tables[0], tables[1]


def _robot_rope_table(
    rope: PhysicalTimeWanRotaryPosEmbed, batch: int, action_steps: int, fps: torch.Tensor, device, include_state: bool = True
):
    """Physical-time ``(t, 0, 0)`` RoPE rows of the robot stream from one of video_dit's RoPE modules: the state at t=0
    (unless ``include_state`` is False: the state_as_context stream) and action row k at ``base_fps * k / (8 * fps)``."""

    action_ids = torch.arange(1, action_steps + 1, device=device, dtype=torch.float64)
    time = rope.base_fps * action_ids[None] / (ACTION_TEMPORAL_COMPRESSION * fps[:, None])
    if include_state:
        time = torch.cat((torch.zeros(batch, 1, device=device, dtype=torch.float64), time), dim=1)
    ids = torch.stack((time, torch.zeros_like(time), torch.zeros_like(time)), dim=-1)
    return rope.from_position_ids(ids)


def _action_rope_tables(
    action_dit: ActionDiT,
    video_dit: VideoExpert,
    action_steps: int,
    batch: int,
    model_fps,
    device,
    frame_stride: int = 1,
    rope_mode: str = ROPE_LEGACY,
):
    """Action RoPE tables (gdn heads, softmax heads) of the robot stream.

    ``independent`` (rwm/mot 42aee4fa9 / 71ac93f43): the action expert's own 1D clock at every stride, the state row at 0
    and the actions at 1 .. S (0 .. S-1 under state_as_context, which has no state row). ``aligned``: the physical clock
    the video shares, through video_dit's RoPE modules (state 0, action k at ``base_fps * k / (8 * fps)``; the actions
    alone under state_as_context). ``legacy`` (every run before 42aee4fa9): independent 0 .. S-1 under
    state_as_context; at a stride above 1 a zero-phase state row plus independent 0 .. S-1 (state and first action both
    at 0); at stride 1 the physical clock with the state row.
    """

    heads = (action_dit.gdn_head_dim, action_dit.softmax_head_dim)
    if rope_mode == ROPE_INDEPENDENT:
        rows = action_steps if action_dit.state_as_context else 1 + action_steps
        return tuple(independent_action_rope(head_dim, rows, batch, action_dit.rope_theta, device) for head_dim in heads)
    fps = PhysicalTimeWanRotaryPosEmbed._normalize_fps(model_fps, batch, device)
    if rope_mode == ROPE_ALIGNED:
        include_state = not action_dit.state_as_context
        return (
            _robot_rope_table(video_dit.rope_linear, batch, action_steps, fps, device, include_state),
            _robot_rope_table(video_dit.rope_softmax, batch, action_steps, fps, device, include_state),
        )
    if action_dit.state_as_context:
        return tuple(independent_action_rope(head_dim, action_steps, batch, action_dit.rope_theta, device) for head_dim in heads)
    if int(frame_stride) != 1:
        return tuple(
            torch.cat(
                (
                    independent_action_rope(head_dim, 1, batch, action_dit.rope_theta, device),
                    independent_action_rope(head_dim, action_steps, batch, action_dit.rope_theta, device),
                ),
                dim=2,
            )
            for head_dim in heads
        )
    return (
        _robot_rope_table(video_dit.rope_linear, batch, action_steps, fps, device),
        _robot_rope_table(video_dit.rope_softmax, batch, action_steps, fps, device),
    )


# -- preambles -----------------------------------------------------------------------------------------------------


def _set_grid(video_dit: VideoExpert, x: torch.Tensor) -> None:
    video_dit.f, video_dit.h, video_dit.w = (
        x.shape[-3] // video_dit.patch_size[0],
        x.shape[-2] // video_dit.patch_size[1],
        x.shape[-1] // video_dit.patch_size[2],
    )


def _video_prepare(
    video_dit: VideoExpert,
    x: torch.Tensor,
    timestep: torch.Tensor,
    model_fps,
    *,
    view_shapes=None,
    view_slot_ids=None,
    rope_layout: str = "semantic_2x2",
    tile_shape=(15, 30),
    video_clock_stride: int = 1,
) -> dict:
    """Video preamble of the current layout (patchify, RoPE, time) up to the block loop; sets ``video_dit.f/h/w``.
    ``video_clock_stride``: source frames per latent-frame step on the RoPE time axis (the video frame stride under
    ``rope: aligned``, else 1).

    For V > 1 (``view_shapes`` given) ``x`` is the packed strip ``[B, C, F, 1, sum(h_v * w_v)]``: tokens are reordered
    view-major, ``t`` / ``t0`` are repeated per view (the video expert modulates per (view, frame) block, so the view
    tiles must be equal) and ``prompt_group_spans`` holds one query span per view.
    """

    x = x.to(video_dit.dtype)
    batch = x.shape[0]
    video_timestep = _to_model_t_domain(timestep, video_dit.timestep_norm_scale_factor)
    _set_grid(video_dit, x)
    x_tok = video_dit.x_embedder(x)
    frames = video_dit.f

    if view_shapes is None:
        rope_linear, rope_softmax = _video_rope_tables(video_dit, model_fps, batch, x.device, video_clock_stride)
        num_frames = frames
        spans = ((0, x_tok.shape[1]),)
    else:
        areas = [height * width for height, width in view_shapes]
        if video_dit.h != 1 or video_dit.w != sum(areas):
            raise ValueError(
                "V > 1 multiview expects the packed latent strip [B, C, F, 1, sum(h_v * w_v)]; got a "
                f"[.., {video_dit.h}, {video_dit.w}] grid for view shapes {view_shapes}"
            )
        if len(set(view_shapes)) != 1:
            raise ValueError(f"the MoT video expert modulates per (view, frame) block and needs equal view tiles; got {view_shapes}")
        x_tok = strip_to_view_tokens(x_tok, frames, view_shapes)
        rope_linear, rope_softmax = _multiview_rope_tables(
            video_dit, model_fps, batch, x.device, frames, view_shapes, view_slot_ids, rope_layout, tile_shape,
            video_clock_stride,
        )
        num_frames = len(view_shapes) * frames
        spans, offset = [], 0
        for area in areas:
            spans.append((offset, frames * area))
            offset += frames * area
        spans = tuple(spans)

    t = video_dit.t_embedder(video_timestep.flatten()).unflatten(dim=0, sizes=video_timestep.shape)
    t0 = video_dit.t_block(t)
    if view_shapes is not None:
        t = t.repeat(1, 1, len(view_shapes), 1)
        t0 = t0.repeat(1, 1, len(view_shapes), 1)
    return {
        "x_tok": x_tok,
        "t": t,
        "t0": t0,
        "rope_linear": rope_linear,
        "rope_softmax": rope_softmax,
        "num_frames": num_frames,
        "view_shapes": view_shapes,
        "prompt_group_spans": spans,
    }


def _video_prepare_legacy(video_dit: VideoExpert, x: torch.Tensor, timestep: torch.Tensor, y: torch.Tensor, mask, model_fps) -> dict:
    """Video preamble of the legacy (606e48dd9) layout: patchify, RoPE, time AND the expert's own text path."""

    x = x.to(video_dit.dtype)
    video_timestep = _to_model_t_domain(timestep, video_dit.timestep_norm_scale_factor)
    y = y.to(video_dit.dtype)
    if y.ndim == 5:
        if y.shape[1] not in (1, 2):
            raise ValueError(
                "token-group text for the single-canvas MoT policy must carry G=1 (one prompt shared by both "
                f"experts) or G=2 (view, robot) groups; got y {tuple(y.shape)}"
            )
        if mask is not None and (mask.ndim != 5 or tuple(mask.shape[:2]) != tuple(y.shape[:2])):
            raise ValueError(f"token-group text mask must start with [B, G] = {tuple(y.shape[:2])}; got {tuple(mask.shape)}")
        y_video, y_action = y[:, 0], y[:, -1]
        mask_video, mask_action = (None, None) if mask is None else (mask[:, 0], mask[:, -1])
    elif y.ndim == 4:
        y_video = y_action = y
        mask_video = mask_action = mask
    else:
        raise ValueError(f"y must be [B, 1, L, C] or token-group [B, G, 1, L, C]; got {tuple(y.shape)}")
    _set_grid(video_dit, x)
    x_tok = video_dit.x_embedder(x)
    rope_linear, rope_softmax = _video_rope_tables(video_dit, model_fps, x.shape[0], x.device)

    t = video_dit.t_embedder(video_timestep.flatten()).unflatten(dim=0, sizes=video_timestep.shape)
    t0 = video_dit.t_block(t)

    y_embedded = video_dit.attention_y_norm(video_dit.y_embedder.y_proj(y_video))
    raw_text = y_action.squeeze(1)

    def _key_mask(group_mask):
        if group_mask is None:
            return None
        group_mask = group_mask.to(torch.int16)
        if group_mask.shape[0] != y_embedded.shape[0]:
            group_mask = group_mask.repeat(y_embedded.shape[0] // group_mask.shape[0], 1)
        return group_mask.squeeze(1).squeeze(1)

    y_lens = _key_mask(mask_video)
    action_lens = _key_mask(mask_action)
    return {
        "x_tok": x_tok,
        "t": t,
        "t0": t0,
        "rope_linear": rope_linear,
        "rope_softmax": rope_softmax,
        "num_frames": video_dit.f,
        "view_shapes": None,
        "y": y_embedded,
        "y_lens": y_lens,
        "raw_text": raw_text,
        "text_mask_bool": None if action_lens is None else action_lens != 0,
    }


def _action_prepare(
    action_dit: ActionDiT,
    video_dit: VideoExpert,
    data_info: dict,
    timestep_norm_scale_factor: float,
    model_fps,
    dtype: torch.dtype,
    device,
    *,
    frame_stride: int = 1,
    rope_mode: str = ROPE_LEGACY,
) -> dict:
    """Action preamble (state/action tokens, RoPE, time) up to the block loop, in either state-conditioning mode."""

    action80 = data_info["action80"].to(device, dtype=dtype)
    action_mask80 = data_info["action_mask80"].to(device)
    action_steps = action80.shape[1]
    batch = action80.shape[0]
    expected_steps = (video_dit.f - 1) * ACTION_TEMPORAL_COMPRESSION * int(frame_stride)
    if action_steps != expected_steps:
        raise ValueError(
            "one motion80 row per source-frame transition is required: "
            f"got {action_steps}, expected {expected_steps} for latent F={video_dit.f} at video frame stride {frame_stride}"
        )
    state80 = data_info["initial_state80"].to(device, dtype=dtype).reshape(batch, 1, -1)
    state_mask80 = data_info["initial_state_condition_mask80"].to(device).reshape(batch, 1, -1)

    action_timestep_rows = data_info["action_timestep"].reshape(batch, -1).to(device)
    action_timestep_scalar = _to_model_t_domain(action_timestep_rows, timestep_norm_scale_factor)[:, 0]

    rope_gdn, rope_softmax = _action_rope_tables(
        action_dit, video_dit, action_steps, batch, model_fps, device, frame_stride, rope_mode
    )

    if action_dit.state_as_context:
        x_tok = action_dit.action_embed(action80, action_mask80)
        t = action_dit.t_embedder(action_timestep_scalar)
        t0 = action_dit.t_block(t)
        num_rows = action_steps
    else:
        x_tok = torch.cat(
            (action_dit.state_embed(state80, state_mask80), action_dit.action_embed(action80, action_mask80)),
            dim=1,
        )
        token_timestep = torch.cat(
            (action_timestep_scalar.new_zeros(batch, 1), action_timestep_scalar.unsqueeze(1).expand(-1, action_steps)),
            dim=1,
        ).unsqueeze(1)
        t = action_dit.t_embedder(token_timestep.flatten()).unflatten(dim=0, sizes=token_timestep.shape)
        t0 = action_dit.t_block(t)
        num_rows = 1 + action_steps

    return {
        "x_tok": x_tok,
        "t": t,
        "t0": t0,
        "state80": state80,
        "state_mask80": state_mask80,
        "rope_gdn": rope_gdn,
        "rope_softmax": rope_softmax,
        "action_steps": action_steps,
        "action_mask80": action_mask80,
        "num_rows": num_rows,
    }


def _detach_blocks(expert: nn.Module) -> list:
    """Take an expert's blocks out of its module tree (re-registered once inside the MoTLayer units); leaves ``expert.blocks = None``."""

    blocks = list(expert.blocks)
    del expert.blocks
    expert.blocks = None
    return blocks


class MoTPolicyModel(nn.Module):
    """MoT dual-system policy: ``video_dit`` + ``action_dit`` with their trunks paired into ``blocks[i] = MoTLayer``."""

    unified_policy_contract = True

    def __init__(self, config: MoTConfig) -> None:
        super().__init__()
        config = config.validate()
        self.mot_config = config
        self.video_layout = config.video_layout
        self.context_layout = config.context_layout
        self.rope = config.rope
        self.canvas_text_groups = int(config.canvas_text_groups)
        self.multiview_spatial_rope_layout = config.multiview_spatial_rope_layout
        self.multiview_spatial_rope_tile_shape = tuple(int(v) for v in config.multiview_spatial_rope_tile_shape)
        self.video_dit = VideoExpert(config.video, with_text=config.legacy)
        video_dit = self.video_dit

        depth = len(video_dit.blocks)
        shared_dim = video_dit.hidden_size
        video_softmax_indices = tuple(
            i for i in range(depth) if video_dit.block_attn_types[i] == video_dit.softmax_attn_type
        )
        video_gdn_indices = tuple(i for i in range(depth) if i not in video_softmax_indices)
        if not video_softmax_indices or not video_gdn_indices:
            raise ValueError("the MoT policy expects video_dit's hybrid GDN + softmax schedule (both kinds present)")
        gdn_probe = video_dit.blocks[video_gdn_indices[0]].attn
        softmax_probe = video_dit.blocks[video_softmax_indices[0]].attn

        context_dim = None
        if not config.legacy:
            self.context_embedder = ContextEmbedder(
                config.video.caption_channels,
                config.video.hidden_size,
                config.action_hidden_size,
                norm_eps=config.video.norm_eps,
                state_as_context=config.action_state_as_context,
                shared=config.shared_caption_embedder,
            )
            context_dim = self.context_embedder.context_width
        self.action_dit = ActionDiT(
            hidden_size=config.action_hidden_size,
            depth=depth,
            shared_dim=shared_dim,
            gdn_heads=gdn_probe.heads,
            gdn_head_dim=gdn_probe.dim,
            softmax_heads=softmax_probe.heads,
            softmax_head_dim=softmax_probe.dim,
            raw_context_dim=config.video.caption_channels,
            cross_attn_heads=config.action_cross_attn_heads,
            mlp_ratio=config.action_mlp_ratio,
            softmax_layer_indices=video_softmax_indices,
            rope_theta=config.action_rope_theta,
            attn_res_block_size=config.action_attn_res_block_size,
            qk_norm=True,
            cross_norm=True,
            state_as_context=config.action_state_as_context,
            legacy_context=config.legacy,
            context_dim=context_dim,
        )

        video_blocks = _detach_blocks(video_dit)
        action_blocks = _detach_blocks(self.action_dit)
        self.blocks = nn.ModuleList(
            [MoTLayer(video_blocks[i], action_blocks[i], is_softmax=i in video_softmax_indices) for i in range(depth)]
        )
        self.depth = depth
        self.attn_res_block_size = config.action_attn_res_block_size
        self.softmax_layer_indices = video_softmax_indices
        self.set_fp32_attention(config.video.fp32_attention)
        self.eval()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def f(self) -> int:
        return self.video_dit.f

    @property
    def h(self) -> int:
        return self.video_dit.h

    @property
    def w(self) -> int:
        return self.video_dit.w

    def set_fp32_attention(self, enabled: bool) -> None:
        """Stamp the video attention modules (the joint attention reads the flag from the video side)."""

        for layer in self.blocks:
            layer.video_block.attn.fp32_attention = bool(enabled)

    def load_checkpoint_state(self, state_dict) -> None:
        load_mot_state_dict(self, state_dict)

    @torch.no_grad()
    def _forward_trunk(self, x_v: torch.Tensor, x_a: torch.Tensor, video: dict, action: dict, contexts: Contexts):
        """AttnRes group loop over the paired layers on preallocated buffers; returns the two final aggregates."""

        attn_res_v, attn_res_a = self.video_dit.attn_res, self.action_dit.attn_res
        block_size = self.attn_res_block_size
        num_blocks = math.ceil(self.depth / block_size)

        def _buffers(attn_res, tokens):
            value_buffer = torch.empty((num_blocks + 1, *tokens.shape), device=tokens.device, dtype=tokens.dtype)
            key_buffer = torch.empty_like(value_buffer)
            value_buffer[0] = tokens
            key_buffer[0] = attn_res.key_norm(tokens.unsqueeze(0)).squeeze(0)
            return value_buffer, key_buffer

        value_v, key_v = _buffers(attn_res_v, x_v)
        value_a, key_a = _buffers(attn_res_a, x_a)
        n_active = 1
        partial_v = partial_a = None

        for group in range(num_blocks):
            start, end = group * block_size, min((group + 1) * block_size, self.depth)
            for index in range(start, end):
                layer = self.blocks[index]
                h_attn_v = attn_res_v._attend_buffer(attn_res_v.attn_proj, value_v, key_v, n_active, partial_v)
                h_attn_a = attn_res_a._attend_buffer(attn_res_a.attn_proj, value_a, key_a, n_active, partial_a)
                attn_delta_v, attn_delta_a, post_v, post_a = layer.attention_sublayer(h_attn_v, h_attn_a, video, action, contexts)
                partial_v = attn_delta_v if partial_v is None else partial_v + attn_delta_v
                partial_a = attn_delta_a if partial_a is None else partial_a + attn_delta_a
                h_mlp_v = attn_res_v._attend_buffer(attn_res_v.mlp_proj, value_v, key_v, n_active, partial_v)
                h_mlp_a = attn_res_a._attend_buffer(attn_res_a.mlp_proj, value_a, key_a, n_active, partial_a)
                mlp_delta_v, mlp_delta_a = layer.mlp_sublayer(h_mlp_v, h_mlp_a, post_v, post_a, video)
                partial_v = partial_v + mlp_delta_v
                partial_a = partial_a + mlp_delta_a

            value_v[n_active] = partial_v
            key_v[n_active] = attn_res_v.key_norm(partial_v.unsqueeze(0)).squeeze(0)
            value_a[n_active] = partial_a
            key_a[n_active] = attn_res_a.key_norm(partial_a.unsqueeze(0)).squeeze(0)
            n_active += 1
            partial_v = partial_a = None

        x_final_v = attn_res_v._attend_buffer(attn_res_v.final_proj, value_v, key_v, n_active, None)
        x_final_a = attn_res_a._attend_buffer(attn_res_a.final_proj, value_a, key_a, n_active, None)
        return x_final_v, x_final_a

    def _heads(self, x_final_v: torch.Tensor, x_final_a: torch.Tensor, v: dict, a: dict) -> dict[str, torch.Tensor]:
        """Video flow in the input layout (strip restored for V > 1) and the masked action flow."""

        video_dit, action_dit = self.video_dit, self.action_dit
        video_tokens = final_layer_frame_aware(video_dit.final_layer, x_final_v, v["t"])
        if v["view_shapes"] is not None:
            video_tokens = view_to_strip_tokens(video_tokens, video_dit.f, v["view_shapes"])
        video_output = video_dit.unpatchify(video_tokens)
        if action_dit.state_as_context:
            action_hidden = x_final_a
            action_condition = ActionDiT.expand_for_action_head(a["t"], a["action_steps"])
        else:
            action_hidden = x_final_a[:, 1:]
            action_condition = a["t"][:, :, 1:, :]
        action_pred = action_dit.action_head(action_hidden, action_condition).masked_fill(~a["action_mask80"], 0)
        return {"x": video_output, "action_pred": action_pred}

    @staticmethod
    def _action_dict(a: dict, frame_aware: bool) -> dict:
        return {
            "t0": a["t0"],
            "num_rows": a["num_rows"],
            "rope_gdn": a["rope_gdn"],
            "rope_softmax": a["rope_softmax"],
            "frame_aware": frame_aware,
        }

    @torch.no_grad()
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, y: torch.Tensor, mask=None, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Run the joint video/action policy forward.

        Args:
            x: video latents: the native grid ``[B, C, F, H, W]`` (one view or the canvas) or, with
                ``data_info['view_count'] > 1``, the packed multiview strip ``[B, C, F, 1, sum(h_v * w_v)]``.
            timestep: frame-indexed video timesteps ``[B, 1, F]`` or ``[B, F]`` in the raw domain (frame 0 clean,
                lockstep with the action rows).
            y: token-group text ``[B, G, 1, L, C]`` (or the base layout ``[B, 1, L, C]`` as G = 1): multiview G = V + 1
                (one group per view, then the robot group); canvas G = 1, the one prompt both experts read.
            mask: text key mask matching y's layout, or None.
            **kwargs: ``data_info`` with model_fps, action80, action_mask80, action_timestep, initial_state80,
                initial_state_condition_mask80; view_count, view_latent_shape and view_slot_ids for V > 1; optional
                video_frame_stride.

        Returns:
            ``{"x": video flow in x's layout, "action_pred": [B, A, 80] zeroed outside action_mask80}``.
        """

        for retired in ("action", "action_mask"):
            if retired in kwargs:
                raise TypeError(f"no explicit {retired!r} kwarg; the robot stream rides data_info['action80'/...]")
        if self.training:
            raise RuntimeError("the MoT policy mirror is inference-only; call eval()")
        data_info = kwargs.get("data_info") or {}
        task = data_info.get("rwm_task", "policy")
        if task != "policy":
            raise ValueError(f"the MoT policy serves rwm_task='policy' only; got {task!r}")
        if self.mot_config.legacy:
            return self._forward_legacy(x, timestep, y, mask, data_info)
        return self._forward_current(x, timestep, y, mask, data_info)

    def _common_checks(self, x: torch.Tensor, timestep: torch.Tensor, data_info: dict):
        latent_frames = x.shape[-3] // self.video_dit.patch_size[0]
        _assert_lockstep(timestep, data_info, latent_frames, self.mot_config.separate_action_schedule)
        if timestep.ndim == 5:
            timestep = timestep.reshape(x.shape[0], 1, -1)
        elif timestep.ndim == 2:
            timestep = timestep.unsqueeze(1)
        for key in ("image_vae_embeds", "image_embeds"):
            if data_info.get(key) is not None:
                raise NotImplementedError(f"data_info[{key!r}] (image conditioning of the video stream) is not part of the MoT policy")
        model_fps = data_info.get("model_fps")
        if model_fps is None:
            if latent_frames != 1:
                raise KeyError("video batches require data_info['model_fps']")
            model_fps = 16.0
        return timestep, model_fps

    def _forward_current(self, x, timestep, y, mask, data_info: dict) -> dict[str, torch.Tensor]:
        canvas = self.mot_config.canvas
        view_count = _view_count(data_info) if "view_count" in data_info else 1
        if canvas and view_count != 1:
            raise ValueError(f"video_layout={self.video_layout!r} consumes one composited canvas (view_count=1); got view_count={view_count}")
        if camera_conditioning_enabled(data_info):
            raise ValueError("the MoT video expert carries no camera geometry channel (no Plucker embed); camera conditioning must stay off")
        view_shapes = _view_shapes(data_info) if view_count > 1 else None
        view_slot_ids = (
            _view_slot_ids(data_info) if view_count > 1 and self.multiview_spatial_rope_layout == "semantic_2x2" else None
        )
        timestep, model_fps = self._common_checks(x, timestep, data_info)
        video_dit, action_dit = self.video_dit, self.action_dit
        frame_stride = _video_frame_stride(data_info)
        video_clock_stride = frame_stride if self.rope == ROPE_ALIGNED else 1

        v = _video_prepare(
            video_dit,
            x,
            timestep,
            model_fps,
            view_shapes=view_shapes,
            view_slot_ids=view_slot_ids,
            rope_layout=self.multiview_spatial_rope_layout,
            tile_shape=self.multiview_spatial_rope_tile_shape,
            video_clock_stride=video_clock_stride,
        )
        a = _action_prepare(
            action_dit, video_dit, data_info, video_dit.timestep_norm_scale_factor, model_fps, video_dit.dtype, x.device,
            frame_stride=frame_stride, rope_mode=self.rope,
        )
        text_groups = int(y.shape[1]) if y.ndim == 5 else 1
        if canvas:
            if text_groups != self.canvas_text_groups:
                raise ValueError(
                    f"this canvas checkpoint reads G={self.canvas_text_groups} prompt row(s) "
                    f"({'the shared caption embedder' if self.mot_config.shared_caption_embedder else 'separate embedders'}); "
                    f"got G={text_groups}"
                )
        elif text_groups != view_count + 1:
            raise ValueError(
                f"token-group text must carry G = V + 1 = {view_count + 1} groups (one per view, then the robot group); "
                f"got G={text_groups}"
            )
        context_video, context_action, context_video_mask, context_action_mask = self.context_embedder(
            y, mask, a["state80"], a["state_mask80"], shared_prompt=canvas and self.canvas_text_groups == 1
        )
        video_groups = context_video.shape[1]
        spans = None
        if video_groups == 1:
            context_video = context_video[:, 0]
            context_video_mask = None if context_video_mask is None else context_video_mask[:, 0]
        else:
            spans = v["prompt_group_spans"]
        contexts = Contexts(context_video, context_action, context_video_mask, context_action_mask, spans)
        video = {"t0": v["t0"], "num_frames": v["num_frames"], "rope_linear": v["rope_linear"], "rope_softmax": v["rope_softmax"]}
        action = self._action_dict(a, frame_aware=not action_dit.state_as_context)
        x_final_v, x_final_a = self._forward_trunk(v["x_tok"], a["x_tok"], video, action, contexts)
        return self._heads(x_final_v, x_final_a, v, a)

    def _forward_legacy(self, x, timestep, y, mask, data_info: dict) -> dict[str, torch.Tensor]:
        view_count = data_info.get("view_count", 1)
        view_count = int(view_count.reshape(-1)[0]) if isinstance(view_count, torch.Tensor) else int(view_count)
        if view_count != 1:
            raise ValueError(f"the MoT policy consumes one composited canvas (view_count=1); got view_count={view_count}")
        timestep, model_fps = self._common_checks(x, timestep, data_info)
        video_dit, action_dit = self.video_dit, self.action_dit

        v = _video_prepare_legacy(video_dit, x, timestep, y, mask, model_fps)
        a = _action_prepare(action_dit, video_dit, data_info, video_dit.timestep_norm_scale_factor, model_fps, video_dit.dtype, x.device)
        raw_text = v["raw_text"].to(video_dit.dtype)
        if action_dit.state_as_context:
            context, context_mask = action_dit.build_context(raw_text, v["text_mask_bool"], a["state80"], a["state_mask80"])
        else:
            context, context_mask = action_dit.project_context(raw_text), v["text_mask_bool"]
        contexts = Contexts(v["y"], context, v["y_lens"], context_mask, None)
        video = {"t0": v["t0"], "num_frames": v["num_frames"], "rope_linear": v["rope_linear"], "rope_softmax": v["rope_softmax"]}
        action = self._action_dict(a, frame_aware=not action_dit.state_as_context)
        x_final_v, x_final_a = self._forward_trunk(v["x_tok"], a["x_tok"], video, action, contexts)
        return self._heads(x_final_v, x_final_a, v, a)


def build_mot_policy(config: MoTConfig) -> MoTPolicyModel:
    """Construct the mirror in eval mode with the config's fp32-attention setting stamped."""

    return MoTPolicyModel(config)


__all__ = [
    "CANVAS_VIDEO_LAYOUTS",
    "CONTEXT_EMBEDDER",
    "CONTEXT_LAYOUTS",
    "LEGACY_ACTION_MLP",
    "MOT_FACTORY_NAME",
    "MOT_ROPE_MODES",
    "MULTIVIEW_ROPE_LAYOUTS",
    "ROPE_ALIGNED",
    "ROPE_INDEPENDENT",
    "ROPE_LEGACY",
    "SHARED_CAPTION_EMBEDDER",
    "MoTConfig",
    "MoTPolicyModel",
    "VIDEO_LAYOUTS",
    "build_mot_policy",
    "camera_conditioning_enabled",
]
