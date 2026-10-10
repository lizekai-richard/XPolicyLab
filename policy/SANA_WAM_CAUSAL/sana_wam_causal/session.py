"""Teacher-forced streaming session of the chunk-causal policy (Sana ``deploy/causal/session.py``, inference subset),
per-chunk observation protocol (Sana ``61aad2d58`` / ``3a6b2bf07``, checked against ``9ee9c2d9d``).

One emitted chunk per cycle, mirroring the robot loop:

1) ``generate_chunk(observation_latent, ...)`` -- denoise chunk c against frozen memory. Every chunk is the window
   [observation_c | targets_c] + [state_c | C actions]: its own observation (the chunk-start frame encoded alone, the
   anchor latent of the chunk's own C+1-row encode under the causal VAE) at t = 0, the noisy targets, the measured
   chunk-start state and the noisy actions. Chunk 0 reads nothing (the bidirectional single window); chunk c >= 1 reads
   the GDN state after the commits of chunks 0..c-1 and the softmax entries of the read rule below;
2) the caller executes the returned chunk and observes the outcome;
3) ``commit_chunk`` -- ONE clean forward (t = 0) of the EXECUTED chunk c: its target latents (the observed C+1-row
   window, strided, VAE-encoded, the anchor latent dropped) and, under ``memory_content`` video_and_action, its
   chunk-start state and C committed actions (video_only: the targets alone, no robot rows). It reads like the
   generation of chunk c, then writes the next GDN state and keeps its softmax K/V as entry c. The observation is never
   committed: the next chunk brings its own.

Softmax read, W = ``sliding_window_chunks``: the forwards on chunk c (its denoising steps and its commit) read the newest
``min(c, W)`` committed chunks, ``c - W .. c - 1``. This is the two-stream training forward's rule (Sana
``CachedChunkCausalPolicySoftmaxAttention._two_stream_spans``: noisy chunk c reads clean chunks ``max(0, c - W) .. c - 1``
plus itself, clean chunk j reads clean chunks ``max(0, j - W) .. j``) and the default read of Sana's deploy session (W - 1
older chunks plus the most recent one).

Positions. Softmax entries are kept BEFORE the rotary embedding and rotated on the fly at every read, on a clock
numbered from 0 at the read's ORIGIN O = the oldest chunk read (``read_plan``): chunk k at slot k - O (targets from
latent frame ``(k - O) * f_c + 1``, state at step ``(k - O) * C``, its actions after it) and the window of chunk c at
slot c - O (its observation at frame ``(c - O) * f_c``, its targets after it). Before anything slides out O = 0, the
physical clock; afterwards every read sits where training's noisy chunk W sits in a window that starts at its first
chunk, so the softmax layers only ever see position ids of a training window. This is Sana's deploy
``kv_position_mode="rank"``; its default ``"absolute"`` differs by one common shift, which the relative RoPE of the
aligned clock does not see. The GDN layers are never windowed: their state keeps keys rotated at the physical
positions of the whole episode, and their windows run on that clock.

Conditional and unconditional text streams hold separate memories (the caption reaches every layer through
cross-attention). Latents are ``[1, C_lat, F, H, W]`` canvas grids; actions and states are in the normalized space.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.sampler import _action_step, _video_step, make_scheduler  # noqa: E402
from sana_wam_min.policy_model.attention import apply_rotary_emb_fp64_softmax  # noqa: E402

from .contract import MEMORY_VIDEO_AND_ACTION  # noqa: E402
from .layers import GDN_KIND, LayerCache  # noqa: E402
from .model import CausalPolicyModel  # noqa: E402

LEFT_GRIPPER, RIGHT_GRIPPER = 16, 45


class CausalMemory:
    """One text stream's committed memory: every GDN layer's ``(S, z)`` and every softmax layer's committed chunks, the
    latter as (PRE-RoPE k, v), rotated by ``layer_caches`` on the clock of each read (module docstring, "Positions")."""

    def __init__(self, model: CausalPolicyModel, window: Optional[int]) -> None:
        if window is not None and int(window) < 1:
            raise ValueError(f"the softmax window must be >= 1 committed chunks or None, got {window}")
        self.window = None if window is None else int(window)
        self.kinds = [block.attn_type for block in model.blocks]
        self.gdn: dict[int, Optional[tuple[torch.Tensor, torch.Tensor]]] = {
            i: None for i, kind in enumerate(self.kinds) if kind == GDN_KIND
        }
        # only the newest W chunks can ever be read again
        self.chunks: dict[int, deque] = {
            i: deque(maxlen=self.window) for i, kind in enumerate(self.kinds) if kind != GDN_KIND
        }
        self.committed = 0

    def read_chunk_ids(self) -> list[int]:
        """Committed chunks the next forward reads besides its own window, oldest first."""

        ids = list(range(self.committed))
        return ids if self.window is None else ids[-self.window :]

    def read_plan(self) -> tuple[list[int], int]:
        """(chunks read, origin O): O = the oldest chunk read (0 when none is). The read is numbered from 0 at O: chunk k
        at slot k - O, the forward's own chunk c at slot c - O."""

        ids = self.read_chunk_ids()
        return ids, (ids[0] if ids else 0)

    def layer_caches(self, *, commit: bool, rope_of: Optional[Callable[[int], torch.Tensor]] = None) -> list[LayerCache]:
        """One ``LayerCache`` per block for one forward. The softmax entries read are rotated here, once for every
        forward that shares this read (the denoising steps of a chunk reuse the caches): ``rope_of(slot)`` is the
        softmax rotary table of a committed chunk at chunk slot ``slot``, the model's ``softmax_entry_rope``."""

        ids, origin = self.read_plan()
        rope = None
        if ids:
            if rope_of is None:
                raise ValueError("a read needs rope_of: softmax entries are kept before the rotary embedding")
            rope = torch.cat([rope_of(chunk - origin) for chunk in ids], dim=2)
        caches = []
        for index, kind in enumerate(self.kinds):
            if kind == GDN_KIND:
                caches.append(LayerCache(commit=commit, gdn_state=self.gdn[index]))
                continue
            context = None
            if ids:
                kept = list(self.chunks[index])   # the newest committed chunks, oldest first
                if len(kept) < len(ids):
                    raise RuntimeError(f"layer {index} keeps {len(kept)} chunks, the read asks for {len(ids)}")
                parts = kept[len(kept) - len(ids) :]
                k = torch.cat([k for k, _ in parts], dim=1)
                v = torch.cat([v for _, v in parts], dim=1)
                context = (apply_rotary_emb_fp64_softmax(k, rope.to(k.device)), v)
            caches.append(LayerCache(commit=commit, softmax_context=context))
        return caches

    def absorb(self, caches) -> None:
        """Keep what a commit forward left in its caches: the next GDN states and the chunk's (pre-RoPE) softmax entry."""

        for index, (kind, cache) in enumerate(zip(self.kinds, caches)):
            if kind == GDN_KIND:
                if cache.gdn_next is None:
                    raise RuntimeError(f"layer {index}: a commit forward left no GDN state")
                self.gdn[index] = cache.gdn_next
            else:
                if cache.softmax_new is None:
                    raise RuntimeError(f"layer {index}: a commit forward left no softmax entry")
                self.chunks[index].append(cache.softmax_new)
        self.committed += 1


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
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        self.model = model
        self.chunk_actions = int(model.chunk_actions)
        self.f_chunk = int(model.latent_frames_per_chunk)
        self.memory_content = str(model.memory_content)
        self.robot_memory = self.memory_content == MEMORY_VIDEO_AND_ACTION
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
        self.window = window
        self.captions = [caption] + ([unconditional_caption] if self.guided else [])
        self.memories: list[CausalMemory] = []
        self.chunk_idx = 0
        self._grid: Optional[tuple[int, int]] = None   # (H, W) of the canvas latent, fixed by the first window
        self._rope_tables: dict = {}                   # softmax rotary table per chunk slot of a read
        self.reset_memory()

    def reset_memory(self) -> None:
        """Forget every committed chunk (fresh GDN states and softmax entries in every stream); the captions stay. The
        next chunk is then generated like chunk 0 of an episode (``memory_mode: none`` in the adapter)."""

        self.memories = [CausalMemory(self.model, self.window) for _ in self.captions]
        self.chunk_idx = 0

    # -- helpers -----------------------------------------------------------------------------------------------------

    @property
    def streams(self) -> list[tuple[Caption, CausalMemory]]:
        return list(zip(self.captions, self.memories))

    def _autocast(self, device: torch.device):
        if self.autocast_dtype is None or device.type != "cuda":
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=self.autocast_dtype)

    def _check_latent(self, latent: torch.Tensor, frames: int, what: str) -> None:
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != frames:
            raise ValueError(f"{what} must be [1, C, {frames}, H, W], got {tuple(latent.shape)}")
        grid = (int(latent.shape[-2]), int(latent.shape[-1]))
        if self._grid is None:
            self._grid = grid
        elif grid != self._grid:
            raise ValueError(f"{what} has the latent grid {grid}, the session runs {self._grid}")

    def read_chunk_ids(self) -> list[int]:
        """Committed chunks the next forward reads (the same in every stream)."""

        return self.memories[0].read_chunk_ids()

    def softmax_origin(self) -> int:
        """The next read's origin O (the oldest chunk read): the softmax layers number that read from 0 there."""

        return self.memories[0].read_plan()[1]

    def _rope_of(self, slot: int) -> torch.Tensor:
        if slot not in self._rope_tables:
            device = next(self.model.parameters()).device
            patch = self.model.patch_size
            self._rope_tables[slot] = self.model.softmax_entry_rope(
                slot, self.model_fps, self._grid[0] // patch[1], self._grid[1] // patch[2], device
            )
        return self._rope_tables[slot]

    def _softmax_offsets(self, frame_offset: int, step_offset: int) -> tuple[int, int]:
        origin = self.softmax_origin()
        return frame_offset - origin * self.f_chunk, step_offset - origin * self.chunk_actions

    # -- the cycle ---------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def generate_chunk(
        self,
        observation_latent: torch.Tensor,
        *,
        anchor_state: torch.Tensor,
        anchor_state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        video_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        gripper_bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise chunk ``chunk_idx`` from its observation ``[1, C_lat, 1, H, W]`` against frozen memory; returns
        (normalized actions [1, C, 80], target latents [1, C_lat, f_c, H, W]). Nothing is written."""

        self._check_latent(observation_latent, 1, "observation_latent")
        chunk = self.chunk_actions
        action_mask = action_mask.to(dtype=torch.bool)
        if action_mask.ndim != 3 or action_mask.shape[:2] != (1, chunk):
            raise ValueError(f"action_mask must be [1, {chunk}, D], got {tuple(action_mask.shape)}")
        like = observation_latent
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
        observation = observation_latent.to(dtype=video.dtype)
        video_scheduler = make_scheduler(self.steps, self.flow_shift, device)
        action_scheduler = make_scheduler(self.steps, self.action_flow_shift, device)
        caches_by_stream = [
            None if memory.committed == 0 else memory.layer_caches(commit=False, rope_of=self._rope_of)
            for memory in self.memories
        ]
        frame_offset = self.chunk_idx * self.f_chunk          # the observation; the targets follow it
        step_offset = self.chunk_idx * chunk
        soft_frame, soft_step = self._softmax_offsets(frame_offset, step_offset)
        state = anchor_state.reshape(1, -1)
        state_mask = anchor_state_mask.reshape(1, -1).to(torch.bool)
        with self._autocast(device):
            for step_index, timestep in enumerate(video_scheduler.timesteps):
                action_timestep = action_scheduler.timesteps[step_index]
                video_t = timestep.reshape(1, 1).expand(1, self.f_chunk).to(torch.float32)
                action_t = action_timestep.reshape(1, 1).expand(1, chunk).to(torch.float32)
                latent = torch.cat((observation, video), dim=2)
                window_t = torch.cat((video_t.new_zeros(1, 1), video_t), dim=1)
                robot = {
                    "state80": state,
                    "state_mask80": state_mask,
                    "action80": action,
                    "action_mask80": action_mask,
                    "action_timesteps": action_t,
                }
                outputs = []
                for caption, caches in zip(self.captions, caches_by_stream):
                    outputs.append(
                        self.model.forward_window(
                            latent, window_t, caption.y, caption.mask,
                            model_fps=self.model_fps, frame_offset=frame_offset, step_offset=step_offset,
                            robot=robot, caches=caches, softmax_frame_offset=soft_frame, softmax_step_offset=soft_step,
                        )
                    )
                cond = outputs[0]
                uncond = outputs[1] if self.guided else None
                targets = slice(1, None)
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
        """Fold the EXECUTED chunk ``chunk_idx`` into memory (one clean forward per stream) and advance. Under
        ``memory_content`` video_only the memory keeps the target latents alone: the actions and the state are checked
        but not committed (Sana's session does the same)."""

        self._check_latent(chunk_latent, self.f_chunk, "chunk_latent")
        chunk = self.chunk_actions
        action_mask = action_mask.to(dtype=torch.bool)
        if executed_action.shape[:2] != (1, chunk) or action_mask.shape != executed_action.shape:
            raise ValueError(
                f"executed_action / action_mask must be [1, {chunk}, D], got {tuple(executed_action.shape)} / "
                f"{tuple(action_mask.shape)}"
            )
        device = chunk_latent.device
        robot = None
        if self.robot_memory:
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
        soft_frame, soft_step = self._softmax_offsets(frame_offset, step_offset)   # the read of chunk c's generation
        with self._autocast(device):
            for caption, memory in zip(self.captions, self.memories):
                caches = memory.layer_caches(commit=True, rope_of=self._rope_of)
                self.model.forward_window(
                    chunk_latent, video_t, caption.y, caption.mask,
                    model_fps=self.model_fps, frame_offset=frame_offset, step_offset=step_offset,
                    robot=robot, caches=caches, softmax_frame_offset=soft_frame, softmax_step_offset=soft_step,
                )
                memory.absorb(caches)
        self.chunk_idx += 1


__all__ = ["Caption", "CausalMemory", "CausalPolicySession"]
