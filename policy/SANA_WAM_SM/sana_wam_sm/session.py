"""The per-episode decisions of the sparse-memory policy on a loaded ``SMPolicyModel`` (batch 1).

Sana's deploy is ``dev/rwm/deploy/causal/sparse_memory_session.py`` (cc18ad78f): ``SparseMemoryPolicySession.decide``
samples one decision from the prefix latents [S | M_1..M_n] of its observation row (``vae_encode.encode_causal`` of
``sparse_memory.prefix_rows``): the prefix runs ONCE per text stream into per-layer caches (``forward_memory_prefix``),
then every Euler step runs the target chunk [T | state | 32 actions] alone against them (``forward_memory_target``), the
training sequence's last chunk token for token. ``decide_uncached`` runs every step on the whole sequence instead, its
reference. This session is that loop (the ``deploy_form`` knob: ``cached`` = ``decide``, ``uncached`` =
``decide_uncached``) with the conventions every SANA adapter of XPolicyLab deploys with: bf16 autocast on CUDA, the
video and the action stream on their own flow-match schedules and CFG scales, the gripper slots clamped to the
normalized range of the training data.

Two decisions share nothing: the memory grid moves with the observation row, so every decision brings its own prefix
and its caches die with it. Conditional and unconditional text streams run separate prefixes (the caption reaches every
block through cross-attention, so a prefix token's hidden states depend on it). Latents are ``[1, C_lat, F, H, W]``
canvas grids or the sana_latent views' ``[1, C_lat, F, 1, sum(h_v * w_v)]`` strip (Sana ``encode_prefix``: the views
packed, their latent shapes passed with every decision as ``view_shapes``, their RoPE slots fixed per episode);
actions and states are in the normalized space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.sampler import _action_step, _video_step, make_scheduler  # noqa: E402

from . import sparse_memory as sm  # noqa: E402
from .contract import DEPLOY_CACHED, DEPLOY_FORMS  # noqa: E402
from .layers import GDN_KIND, GDNCache, SoftmaxCache  # noqa: E402
from .model import SMPolicyModel, ViewShapes  # noqa: E402

LEFT_GRIPPER, RIGHT_GRIPPER = 16, 45


@dataclass
class Caption:
    y: torch.Tensor          # [1, 1, 1, L, C_text]
    mask: torch.Tensor       # [1, 1, L]


class SMSession:
    """The decisions of one episode."""

    def __init__(
        self,
        model: SMPolicyModel,
        *,
        caption: Caption,
        unconditional_caption: Optional[Caption],
        steps: int,
        flow_shift: float,
        action_flow_shift: Optional[float],
        video_cfg_scale: float,
        action_cfg_scale: float,
        model_fps: float,
        deploy_form: str,
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
        view_slot_ids: tuple[int, ...] = (0,),
    ) -> None:
        self.model = model
        self.memory = dict(model.memory)
        if deploy_form not in DEPLOY_FORMS:
            raise ValueError(f"deploy_form must be one of {DEPLOY_FORMS}, got {deploy_form!r}")
        self.deploy_form = deploy_form
        self.chunk_actions = int(model.chunk_actions)
        self.steps = int(steps)
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        self.flow_shift = float(flow_shift)
        self.action_flow_shift = float(flow_shift if action_flow_shift is None else action_flow_shift)
        for name, value in (("flow_shift", self.flow_shift), ("action_flow_shift", self.action_flow_shift)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
        self.video_cfg_scale = float(video_cfg_scale)
        self.action_cfg_scale = float(action_cfg_scale)
        for name, value in (("video_cfg_scale", self.video_cfg_scale), ("action_cfg_scale", self.action_cfg_scale)):
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{name} must be finite and >= 1")
        self.guided = self.video_cfg_scale > 1 or self.action_cfg_scale > 1
        if self.guided and unconditional_caption is None:
            raise ValueError("text CFG needs the unconditional caption")
        self.model_fps = float(model_fps)
        self.autocast_dtype = autocast_dtype
        self.captions = [caption] + ([unconditional_caption] if self.guided else [])
        self.view_slot_ids = tuple(int(slot) for slot in view_slot_ids)
        self.decision = 0
        self._grid: Optional[tuple] = None

    # -- helpers -----------------------------------------------------------------------------------------------------

    def _autocast(self, device: torch.device):
        if self.autocast_dtype is None or device.type != "cuda":
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=self.autocast_dtype)

    def _check_prefix(self, prefix_latents: torch.Tensor, view_shapes: Optional[ViewShapes]) -> int:
        if prefix_latents.ndim != 5 or prefix_latents.shape[0] != 1 or prefix_latents.shape[2] < 1:
            raise ValueError(f"prefix_latents must be [1, C, 1 + n, H, W], got {tuple(prefix_latents.shape)}")
        views = None if view_shapes is None else tuple((int(h), int(w)) for h, w in view_shapes)
        if views is not None and len(views) > 1 and len(views) != len(self.view_slot_ids):
            raise ValueError(f"{len(views)} views, the session holds the RoPE slots {self.view_slot_ids}")
        grid = (int(prefix_latents.shape[-2]), int(prefix_latents.shape[-1]), views)
        if self._grid is None:
            self._grid = grid
        elif grid != self._grid:
            raise ValueError(f"prefix_latents has the latent layout {grid}, the session runs {self._grid}")
        n = int(prefix_latents.shape[2]) - 1
        if n != sm.memory_latents(self.memory):
            raise ValueError(f"a memory of {self.memory[sm.MEMORY_FRAMES]} rows holds {sm.memory_latents(self.memory)} "
                             f"latents, the prefix {n}")
        return n

    def _prefix_caches(self, prefix: torch.Tensor, views: dict) -> list[list]:
        """Run the prefix once per text stream (Sana ``forward_memory_prefix``) and return, per stream, the caches every
        target step reads: each GDN layer's state after M_n, each softmax layer's rotated K and V."""

        zeros = torch.zeros(1, prefix.shape[2], device=prefix.device)
        held = []
        for caption in self.captions:
            caches = [
                GDNCache(commit=True) if block.attn_type == GDN_KIND else SoftmaxCache(commit=True)
                for block in self.model.blocks
            ]
            self.model.forward_sm(
                prefix, zeros, caption.y, caption.mask, form=sm.FORM_PREFIX, model_fps=self.model_fps, caches=caches,
                **views,
            )
            for index, cache in enumerate(caches):
                if cache.out is None:
                    raise RuntimeError(f"layer {index}: the prefix forward left no cache")
            held.append([GDNCache(seed=c.out) if isinstance(c, GDNCache) else SoftmaxCache(prefix=c.out) for c in caches])
        return held

    # -- one decision -------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def decide(
        self,
        prefix_latents: torch.Tensor,
        *,
        view_shapes: Optional[ViewShapes] = None,
        anchor_state: torch.Tensor,
        anchor_state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        video_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        gripper_bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample decision ``self.decision`` from its prefix latents ``[1, C, 1 + n, H, W]`` ([S | M_1..M_n]; the strip
        ``[1, C, 1 + n, 1, S]`` with its ``view_shapes``); returns (normalized actions [1, C, 80], the target latent in the
        prefix's layout) and advances to the next decision. The noise not supplied is drawn from ``generator`` as Sana
        draws it: the target latent's first, then the actions'."""

        n = self._check_prefix(prefix_latents, view_shapes)
        views = {"view_shapes": view_shapes, "view_slot_ids": self.view_slot_ids}
        chunk = self.chunk_actions
        action_mask = action_mask.to(dtype=torch.bool)
        if action_mask.ndim != 3 or action_mask.shape[:2] != (1, chunk):
            raise ValueError(f"action_mask must be [1, {chunk}, D], got {tuple(action_mask.shape)}")
        device, dtype = prefix_latents.device, prefix_latents.dtype
        if video_noise is None:
            video_noise = torch.randn(
                (1, prefix_latents.shape[1], 1, *prefix_latents.shape[3:]), generator=generator, device=device, dtype=dtype
            )
        if action_noise is None:
            action_noise = torch.randn(tuple(action_mask.shape), generator=generator, device=device, dtype=dtype)
        video = video_noise.to(device=device, dtype=dtype).clone()
        action = action_noise.to(device=device, dtype=dtype).masked_fill(~action_mask, 0)
        video_scheduler = make_scheduler(self.steps, self.flow_shift, device)
        action_scheduler = make_scheduler(self.steps, self.action_flow_shift, device)
        state = anchor_state.reshape(1, -1)
        state_mask = anchor_state_mask.reshape(1, -1).to(torch.bool)
        cached = self.deploy_form == DEPLOY_CACHED
        with self._autocast(device):
            caches = self._prefix_caches(prefix_latents, views) if cached else None
            for step_index, timestep in enumerate(video_scheduler.timesteps):
                action_timestep = action_scheduler.timesteps[step_index]
                video_t = timestep.reshape(1, 1).to(torch.float32)
                action_t = action_timestep.reshape(1, 1).expand(1, chunk).to(torch.float32)
                robot = {
                    "state80": state,
                    "state_mask80": state_mask,
                    "action80": action,
                    "action_mask80": action_mask,
                    "action_timesteps": action_t,
                }
                outputs = []
                for stream, caption in enumerate(self.captions):
                    if cached:
                        out = self.model.forward_sm(
                            video, video_t, caption.y, caption.mask, form=sm.FORM_TARGET, model_fps=self.model_fps,
                            robot=robot, caches=caches[stream], memory_latents=n, **views,
                        )
                    else:
                        out = self.model.forward_sm(
                            torch.cat((prefix_latents, video), dim=2),
                            torch.cat((video_t.new_zeros(1, n + 1), video_t), dim=1),
                            caption.y, caption.mask, form=sm.FORM_SEQUENCE, model_fps=self.model_fps, robot=robot,
                            **views,
                        )
                    outputs.append({"x": out["x"][:, :, -1:], "action_pred": out["action_pred"]})
                cond = outputs[0]
                uncond = outputs[1] if self.guided else None
                video_prediction = cond["x"]
                if uncond is not None and self.video_cfg_scale > 1:
                    video_prediction = uncond["x"] + self.video_cfg_scale * (video_prediction - uncond["x"])
                action_prediction = cond["action_pred"]
                if uncond is not None and self.action_cfg_scale > 1:
                    action_prediction = uncond["action_pred"] + self.action_cfg_scale * (
                        action_prediction - uncond["action_pred"]
                    )
                video = _video_step(video_scheduler, video_prediction.to(video.dtype), timestep, video, video_t)
                action = _action_step(
                    action_scheduler, action_prediction.to(action.dtype), action_timestep, action, action_t
                ).masked_fill(~action_mask, 0)
        if gripper_bounds is not None:
            for slot, (low, high) in zip((LEFT_GRIPPER, RIGHT_GRIPPER), gripper_bounds):
                action[..., slot] = action[..., slot].clamp(float(low), float(high))
        if not bool(torch.isfinite(video).all()) or not bool(torch.isfinite(action.masked_select(action_mask)).all()):
            raise FloatingPointError("the decision produced non-finite values")
        self.decision += 1
        return action, video


__all__ = ["Caption", "SMSession"]
