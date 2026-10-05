"""Teacher-forced streaming session of the chunk-causal policy (Sana ``deploy/causal/session.py``, inference subset).

One emitted chunk per cycle, mirroring the robot loop:

0) ``commit_observation`` -- once per episode: the first frame enters memory as its own video-only clean entry E_0
   (GDN state S_0, softmax K/V of entry 0);
1) ``generate_chunk`` -- denoise the next chunk against frozen memory. Chunk 0 is the cache-free window
   [observation | chunk-0 targets] (``obs_in_first_chunk``); chunk c >= 1 is the cached window of its targets, which
   reads the GDN state after the last commit and the softmax entries of the window rule;
2) the caller executes the returned chunk and observes the outcome;
3) ``commit_chunk`` -- ONE clean forward (t = 0) of the EXECUTED chunk: its target latents (the observed C+1-row window,
   strided, VAE-encoded, the anchor latent dropped), the executed actions and the chunk-start state; it reads like a
   generation of that chunk, then writes the next GDN state and appends its K/V as entry E_{c+1}.

Softmax read rule (``window_rule``):

* ``training`` (default): the forward on chunk c reads the newest ``min(c + 1, N)`` entries of [E_0, E_1, .., E_c],
  N = ``sliding_window_chunks``; the observation entry evicts like any other once c >= N. This is the two-stream
  training forward's rule (Sana ``CachedChunkCausalPolicySoftmaxAttention._two_stream_spans``: noisy chunk c reads
  clean entries ``max(0, c + 1 - N) .. c``; clean entry j reads ``max(0, j - N) .. j``), at the same physical positions.
* ``keep_obs``: E_0 plus the newest ``N - 1`` chunk entries (Sana's deploy KV manager always reads the observation
  entry; this keeps its read set at physical positions, as ``kv_position_mode="absolute"`` would).

Conditional and unconditional text streams hold separate memories (the caption reaches every layer through
cross-attention). Latents are ``[1, C_lat, F, H, W]`` canvas grids; actions and states are in the normalized space.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.sampler import _action_step, _video_step, make_scheduler  # noqa: E402

from .layers import GDN_KIND, LayerCache  # noqa: E402
from .model import CausalPolicyModel  # noqa: E402

WINDOW_RULES = ("training", "keep_obs")
LEFT_GRIPPER, RIGHT_GRIPPER = 16, 45


class CausalMemory:
    """One text stream's committed memory: every GDN layer's ``(S, z)`` and every softmax layer's entries."""

    def __init__(self, model: CausalPolicyModel, window: Optional[int], rule: str = "training") -> None:
        if rule not in WINDOW_RULES:
            raise ValueError(f"window_rule must be one of {WINDOW_RULES}, got {rule!r}")
        if window is not None and int(window) < 1:
            raise ValueError(f"the softmax window must be >= 1 entries or None, got {window}")
        if rule == "keep_obs" and window is not None and int(window) < 2:
            raise ValueError("keep_obs needs a window of at least 2 entries (the observation + one chunk)")
        self.window = None if window is None else int(window)
        self.rule = rule
        self.kinds = [block.attn_type for block in model.blocks]
        self.gdn: dict[int, Optional[tuple[torch.Tensor, torch.Tensor]]] = {
            i: None for i, kind in enumerate(self.kinds) if kind == GDN_KIND
        }
        self.obs: dict[int, Optional[tuple[torch.Tensor, torch.Tensor]]] = {
            i: None for i, kind in enumerate(self.kinds) if kind != GDN_KIND
        }
        keep = None if self.window is None else (self.window if rule == "training" else self.window - 1)
        self.chunks: dict[int, deque] = {
            i: deque(maxlen=keep) for i, kind in enumerate(self.kinds) if kind != GDN_KIND
        }
        self.entries = 0  # committed entries, the observation included

    def read_entry_ids(self) -> list[int]:
        """Entry ids (0 = observation, k + 1 = chunk k) the next forward reads besides its own window."""

        ids = list(range(self.entries))
        if self.window is None:
            return ids
        if self.rule == "training":
            return ids[-self.window :]
        return ids[:1] + ids[1:][-(self.window - 1) :]

    def _context(self, index: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self.entries == 0:
            return None
        ids = self.read_entry_ids()
        kept = list(self.chunks[index])          # the newest chunk entries, oldest first
        first_kept = self.entries - len(kept)    # entry id of kept[0]
        parts = []
        for entry in ids:
            if entry == 0:
                if self.obs[index] is None:
                    raise RuntimeError("the observation entry was evicted but the read rule asks for it")
                parts.append(self.obs[index])
            else:
                position = entry - first_kept
                if position < 0:
                    raise RuntimeError(f"entry {entry} was evicted but the read rule asks for it")
                parts.append(kept[position])
        if not parts:
            return None
        return torch.cat([k for k, _ in parts], dim=1), torch.cat([v for _, v in parts], dim=1)

    def layer_caches(self, *, commit: bool) -> list[LayerCache]:
        caches = []
        for index, kind in enumerate(self.kinds):
            if kind == GDN_KIND:
                caches.append(LayerCache(commit=commit, gdn_state=self.gdn[index]))
            else:
                caches.append(LayerCache(commit=commit, softmax_context=self._context(index)))
        return caches

    def absorb(self, caches: Sequence[LayerCache]) -> None:
        """Keep what a commit forward left in its caches: the next GDN states and the new softmax entry."""

        for index, (kind, cache) in enumerate(zip(self.kinds, caches)):
            if kind == GDN_KIND:
                if cache.gdn_next is None:
                    raise RuntimeError(f"layer {index}: a commit forward left no GDN state")
                self.gdn[index] = cache.gdn_next
            else:
                if cache.softmax_new is None:
                    raise RuntimeError(f"layer {index}: a commit forward left no softmax entry")
                if self.entries == 0:
                    self.obs[index] = cache.softmax_new
                else:
                    self.chunks[index].append(cache.softmax_new)
        if self.rule == "training" and self.window is not None and self.entries + 1 > self.window:
            for index in self.obs:
                self.obs[index] = None  # the observation slid out of every future read
        self.entries += 1


@dataclass
class Caption:
    y: torch.Tensor          # [1, 1, 1, L, C_text]
    mask: torch.Tensor       # [1, 1, L]


class CausalPolicySession:
    """The chunk cycle of one episode on a loaded ``CausalPolicyModel`` (batch 1)."""

    def __init__(
        self,
        model: CausalPolicyModel,
        *,
        caption: Caption,
        unconditional_caption: Optional[Caption],
        steps: int,
        flow_shift: float,
        action_flow_shift: Optional[float],
        video_cfg_scale: float,
        action_cfg_scale: float,
        model_fps: float,
        window: Optional[int],
        window_rule: str = "training",
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        self.model = model
        self.chunk_actions = int(model.chunk_actions)
        self.f_chunk = int(model.latent_frames_per_chunk)
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
        self.streams = [(caption, CausalMemory(model, window, window_rule))]
        if self.guided:
            self.streams.append((unconditional_caption, CausalMemory(model, window, window_rule)))
        self.window = window
        self.window_rule = window_rule
        self.chunk_idx = 0
        self.obs_latent: Optional[torch.Tensor] = None

    # -- helpers -----------------------------------------------------------------------------------------------------

    def _autocast(self, device: torch.device):
        if self.autocast_dtype is None or device.type != "cuda":
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=self.autocast_dtype)

    def _window(self, chunk: int, targets: torch.Tensor, video_t: torch.Tensor):
        """(latent, per-frame timesteps, frame offset, step offset) of chunk ``chunk``'s window."""

        if chunk == 0:
            latent = torch.cat((self.obs_latent.to(targets.dtype), targets), dim=2)
            timesteps = torch.cat((video_t.new_zeros(1, 1), video_t), dim=1)
            return latent, timesteps, 0, 0
        return targets, video_t, chunk * self.f_chunk + 1, chunk * self.chunk_actions

    def read_entry_ids(self) -> list[int]:
        """Entry ids the next forward reads (0 = observation, k + 1 = chunk k); the same in every stream."""

        return self.streams[0][1].read_entry_ids()

    # -- the cycle ---------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def commit_observation(self, obs_latent: torch.Tensor) -> None:
        if self.obs_latent is not None or self.chunk_idx != 0:
            raise RuntimeError("commit_observation is the cold start: once per episode, before the first chunk")
        if obs_latent.ndim != 5 or obs_latent.shape[0] != 1 or obs_latent.shape[2] != 1:
            raise ValueError(f"obs_latent must be [1, C, 1, H, W], got {tuple(obs_latent.shape)}")
        video_t = torch.zeros(1, 1, device=obs_latent.device, dtype=torch.float32)
        with self._autocast(obs_latent.device):
            for caption, memory in self.streams:
                caches = memory.layer_caches(commit=True)
                self.model.forward_window(
                    obs_latent, video_t, caption.y, caption.mask,
                    model_fps=self.model_fps, frame_offset=0, step_offset=0, robot=None, caches=caches,
                )
                memory.absorb(caches)
        self.obs_latent = obs_latent

    @torch.inference_mode()
    def generate_chunk(
        self,
        *,
        anchor_state: torch.Tensor,
        anchor_state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        video_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        gripper_bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise chunk ``chunk_idx`` against frozen memory; returns (normalized actions [1, C, 80], target latents
        [1, C_lat, f_c, H, W]). Nothing is written."""

        if self.obs_latent is None:
            raise RuntimeError("commit_observation first")
        chunk = self.chunk_actions
        action_mask = action_mask.to(dtype=torch.bool)
        if action_mask.shape[:2] != (1, chunk):
            raise ValueError(f"action_mask must be [1, {chunk}, D], got {tuple(action_mask.shape)}")
        like = self.obs_latent
        device = like.device
        if video_noise is None:
            video_noise = torch.randn(
                (1, like.shape[1], self.f_chunk, *like.shape[3:]), generator=generator, device=device, dtype=like.dtype
            )
        if action_noise is None:
            action_noise = torch.randn(
                (1, chunk, action_mask.shape[2]), generator=generator, device=device, dtype=like.dtype
            )
        video = video_noise.to(device=device, dtype=like.dtype)
        action = action_noise.to(device=device, dtype=like.dtype).masked_fill(~action_mask, 0)
        video_scheduler = make_scheduler(self.steps, self.flow_shift, device)
        action_scheduler = make_scheduler(self.steps, self.action_flow_shift, device)
        caches_by_stream = [None if self.chunk_idx == 0 else memory.layer_caches(commit=False) for _, memory in self.streams]
        state = anchor_state.reshape(1, -1)
        state_mask = anchor_state_mask.reshape(1, -1).to(torch.bool)
        with self._autocast(device):
            for step_index, timestep in enumerate(video_scheduler.timesteps):
                action_timestep = action_scheduler.timesteps[step_index]
                video_t = timestep.reshape(1, 1).expand(1, self.f_chunk).to(torch.float32)
                action_t = action_timestep.reshape(1, 1).expand(1, chunk).to(torch.float32)
                latent, window_t, frame_offset, step_offset = self._window(self.chunk_idx, video, video_t)
                robot = {
                    "state80": state,
                    "state_mask80": state_mask,
                    "action80": action,
                    "action_mask80": action_mask,
                    "action_timesteps": action_t,
                }
                outputs = []
                for (caption, _memory), caches in zip(self.streams, caches_by_stream):
                    outputs.append(
                        self.model.forward_window(
                            latent, window_t, caption.y, caption.mask,
                            model_fps=self.model_fps, frame_offset=frame_offset, step_offset=step_offset,
                            robot=robot, caches=caches,
                        )
                    )
                cond = outputs[0]
                uncond = outputs[1] if self.guided else None
                targets = slice(latent.shape[2] - self.f_chunk, None)
                video_prediction = cond["x"][:, :, targets]
                if uncond is not None and self.video_cfg_scale > 1:
                    video_prediction = uncond["x"][:, :, targets] + self.video_cfg_scale * (
                        video_prediction - uncond["x"][:, :, targets]
                    )
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
            raise FloatingPointError("chunk generation produced non-finite values")
        return action, video

    @torch.inference_mode()
    def commit_chunk(
        self,
        chunk_latent: torch.Tensor,
        executed_action: torch.Tensor,
        *,
        anchor_state: torch.Tensor,
        anchor_state_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        """Fold the EXECUTED chunk ``chunk_idx`` into memory (one clean forward per stream) and advance."""

        if self.obs_latent is None:
            raise RuntimeError("commit_observation first")
        chunk = self.chunk_actions
        if chunk_latent.ndim != 5 or chunk_latent.shape[0] != 1 or chunk_latent.shape[2] != self.f_chunk:
            raise ValueError(f"chunk_latent must be [1, C, {self.f_chunk}, H, W], got {tuple(chunk_latent.shape)}")
        action_mask = action_mask.to(dtype=torch.bool)
        if executed_action.shape[:2] != (1, chunk) or action_mask.shape != executed_action.shape:
            raise ValueError(
                f"executed_action / action_mask must be [1, {chunk}, D], got {tuple(executed_action.shape)} / "
                f"{tuple(action_mask.shape)}"
            )
        device = chunk_latent.device
        robot = {
            "state80": anchor_state.reshape(1, -1),
            "state_mask80": anchor_state_mask.reshape(1, -1).to(torch.bool),
            "action80": executed_action.masked_fill(~action_mask, 0),
            "action_mask80": action_mask,
            "action_timesteps": torch.zeros(1, chunk, device=device),
        }
        video_t = torch.zeros(1, self.f_chunk, device=device)
        frame_offset = self.chunk_idx * self.f_chunk + 1
        step_offset = self.chunk_idx * chunk
        with self._autocast(device):
            for caption, memory in self.streams:
                caches = memory.layer_caches(commit=True)
                self.model.forward_window(
                    chunk_latent, video_t, caption.y, caption.mask,
                    model_fps=self.model_fps, frame_offset=frame_offset, step_offset=step_offset,
                    robot=robot, caches=caches,
                )
                memory.absorb(caches)
        self.chunk_idx += 1


__all__ = ["Caption", "CausalMemory", "CausalPolicySession", "WINDOW_RULES"]
