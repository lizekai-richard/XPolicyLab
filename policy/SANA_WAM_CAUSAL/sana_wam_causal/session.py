"""Teacher-forced streaming session of the chunk-causal policy (Sana ``deploy/causal/session.py``, inference subset),
episode protocol (Sana ``d37817788`` / ``fcccb4c88`` / ``3cbe3d85e``, deploy ``4bd1b674c``; inference handoff 03).

An episode opens at its first frame with two commits, the training's phase window whose head is one whole chunk:

0) ``commit_sink(sink_latent)`` -- the episode's first observation (each view's first frame encoded alone, the first
   latent of the view's causal VAE stream): one clean forward per text stream on video frame 0 without robot rows. Its
   GDN state is the state's first write and its softmax K/V the sink every later forward reads besides its window;
   then ``commit_chunk`` of chunk 0, the frozen first chunk: the first frame repeated over the chunk's C / s frames,
   encoded on after the sink in the same stream (under ``memory_content`` video_and_action its rows hold the robot at
   its first state). Chunk 0 is memory only, as the training's padded chunk 0, and is never generated.

Then one emitted chunk per cycle, mirroring the robot loop, from chunk 1 at the first frame on:

1) ``generate_chunk(observation_latent, ...)`` -- denoise chunk c against frozen memory. The window is [observation_c |
   targets_c] + [state_c | C actions]: its own observation (the chunk-start frame encoded alone; at chunk 1 the first
   frame, i.e. the sink's latent) at t = 0, the noisy targets, the measured chunk-start state and the noisy actions.
   Under ``chunk_observation`` false the window is [targets_c] + rows and the current frame is read from the memory
   (the last frame of chunk c - 1's targets); pass ``observation_latent=None``. It reads the GDN state after the sink
   and the commits of chunks 0..c-1 and the softmax entries of the read rule below;
2) the caller executes the returned chunk and observes the outcome;
3) ``commit_chunk`` -- ONE clean forward (t = 0) of the EXECUTED chunk c: its target latents (the next latents of the
   per-view causal VAE streams over the chunk's observed frames) and, under ``memory_content`` video_and_action, its
   chunk-start state and C committed actions (video_only: the targets alone, no robot rows). It reads like the
   generation of chunk c, then writes the next GDN state and keeps its softmax K/V as entry c. No observation but the
   sink is ever committed.

Softmax read, W = ``sliding_window_chunks``: the forwards on chunk c (its denoising steps and its commit) read the sink
and the newest ``min(c, W)`` committed chunks, ``c - W .. c - 1``. This is the two-stream training forward's rule (Sana
``CachedChunkCausalPolicySoftmaxAttention._two_stream_spans``: noisy chunk c reads the sink plus clean chunks
``max(0, c - W) .. c - 1`` plus itself, clean chunk k reads the sink plus clean chunks ``max(0, k - W) .. k``) and the
default read of Sana's deploy session (W - 1 older chunks plus the most recent one, besides the sink).

Positions. Softmax entries are kept BEFORE the rotary embedding and rotated on the fly at every read, on a clock
numbered from 0 at the read's ORIGIN O = the oldest chunk read (``read_plan``; O = max(0, c - W)): chunk k at slot
k - O (targets from latent frame ``(k - O) * f_c + 1``, state at step ``(k - O) * C``, its actions after it), the
window of chunk c at slot c - O (its observation at frame ``(c - O) * f_c``, its targets after it) and the sink at
frame 0, one frame before the earliest frame the read takes: the training canvas's StreamingLLM sink, which an evicting
reader takes next to its window (``_sink_shift_phases``), and Sana's deploy KV store, which puts it at the first read
frame - 1 (``kv_position_mode="absolute"``) or at 0 (``"rank"``); the relative RoPE of the aligned clock sees one
common shift between the three. The GDN layers are never windowed: their state keeps keys rotated at the physical
positions of the whole episode (the sink at frame 0), and their windows run on that clock.

Conditional and unconditional text streams hold separate memories (the caption reaches every layer through
cross-attention). Latents are ``[1, C_lat, F, H, W]`` canvas grids or ``[1, C_lat, F, 1, sum(H_v * W_v)]`` view-major
strips; actions and states are in the normalized space.
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
from .model import CausalPolicyModel, ViewShapes  # noqa: E402

LEFT_GRIPPER, RIGHT_GRIPPER = 16, 45


class CausalMemory:
    """One text stream's committed memory: every GDN layer's ``(S, z)``, every softmax layer's sink and committed chunks,
    the latter two as (PRE-RoPE k, v), rotated by ``layer_caches`` on the clock of each read (module docstring,
    "Positions")."""

    def __init__(self, model: CausalPolicyModel, window: Optional[int]) -> None:
        if window is not None and int(window) < 1:
            raise ValueError(f"the softmax window must be >= 1 committed chunks or None, got {window}")
        self.window = None if window is None else int(window)
        self.kinds = [block.attn_type for block in model.blocks]
        self.gdn: dict[int, Optional[tuple[torch.Tensor, torch.Tensor]]] = {
            i: None for i, kind in enumerate(self.kinds) if kind == GDN_KIND
        }
        self.sink: dict[int, Optional[tuple[torch.Tensor, torch.Tensor]]] = {
            i: None for i, kind in enumerate(self.kinds) if kind != GDN_KIND
        }
        # only the newest W chunks can ever be read again
        self.chunks: dict[int, deque] = {
            i: deque(maxlen=self.window) for i, kind in enumerate(self.kinds) if kind != GDN_KIND
        }
        self.has_sink = False
        self.committed = 0

    def read_chunk_ids(self) -> list[int]:
        """Committed chunks the next forward reads besides the sink and its own window, oldest first."""

        ids = list(range(self.committed))
        return ids if self.window is None else ids[-self.window :]

    def read_plan(self) -> tuple[list[int], int]:
        """(chunks read, origin O): O = the oldest chunk read (the forward's own chunk when none is). The read is
        numbered from 0 at O: chunk k at slot k - O, the forward's own chunk c at slot c - O, the sink at frame 0."""

        ids = self.read_chunk_ids()
        return ids, (ids[0] if ids else self.committed)

    def layer_caches(
        self,
        *,
        commit: bool,
        rope_of: Optional[Callable[[int], torch.Tensor]] = None,
        sink_rope: Optional[torch.Tensor] = None,
    ) -> list[LayerCache]:
        """One ``LayerCache`` per block for one forward. The softmax entries read are rotated here, once for every
        forward that shares this read (the denoising steps of a chunk reuse the caches): ``sink_rope`` is the softmax
        rotary table of the sink at frame 0 and ``rope_of(slot)`` that of a committed chunk at chunk slot ``slot``
        (the model's ``softmax_entry_rope``)."""

        ids, origin = self.read_plan()
        rope = None
        if ids:
            if rope_of is None:
                raise ValueError("a read needs rope_of: softmax entries are kept before the rotary embedding")
            rope = torch.cat([rope_of(chunk - origin) for chunk in ids], dim=2)
        if self.has_sink and sink_rope is None:
            raise ValueError("a read needs the sink's rope: the sink is kept before the rotary embedding")
        caches = []
        for index, kind in enumerate(self.kinds):
            if kind == GDN_KIND:
                caches.append(LayerCache(commit=commit, gdn_state=self.gdn[index]))
                continue
            keys, values = [], []
            if self.has_sink:
                sink_k, sink_v = self.sink[index]
                keys.append(apply_rotary_emb_fp64_softmax(sink_k, sink_rope.to(sink_k.device)))
                values.append(sink_v)
            if ids:
                kept = list(self.chunks[index])   # the newest committed chunks, oldest first
                if len(kept) < len(ids):
                    raise RuntimeError(f"layer {index} keeps {len(kept)} chunks, the read asks for {len(ids)}")
                parts = kept[len(kept) - len(ids) :]
                k = torch.cat([k for k, _ in parts], dim=1)
                keys.append(apply_rotary_emb_fp64_softmax(k, rope.to(k.device)))
                values.append(torch.cat([v for _, v in parts], dim=1))
            context = (torch.cat(keys, dim=1), torch.cat(values, dim=1)) if keys else None
            caches.append(LayerCache(commit=commit, softmax_context=context))
        return caches

    def _absorb_gdn(self, index: int, cache: LayerCache) -> None:
        if cache.gdn_next is None:
            raise RuntimeError(f"layer {index}: a commit forward left no GDN state")
        self.gdn[index] = cache.gdn_next

    def absorb_sink(self, caches) -> None:
        """Keep what the sink forward left: the GDN states after its write and its (pre-RoPE) softmax entry."""

        if self.has_sink or self.committed:
            raise RuntimeError("the sink opens the memory: committed once, before the first chunk")
        for index, (kind, cache) in enumerate(zip(self.kinds, caches)):
            if kind == GDN_KIND:
                self._absorb_gdn(index, cache)
            else:
                if cache.softmax_new is None:
                    raise RuntimeError(f"layer {index}: the sink forward left no softmax entry")
                self.sink[index] = cache.softmax_new
        self.has_sink = True

    def absorb(self, caches) -> None:
        """Keep what a chunk commit left in its caches: the next GDN states and the chunk's (pre-RoPE) softmax entry."""

        if not self.has_sink:
            raise RuntimeError("commit the sink first: the episode's first observation opens the memory")
        for index, (kind, cache) in enumerate(zip(self.kinds, caches)):
            if kind == GDN_KIND:
                self._absorb_gdn(index, cache)
            else:
                if cache.softmax_new is None:
                    raise RuntimeError(f"layer {index}: a commit forward left no softmax entry")
                self.chunks[index].append(cache.softmax_new)
        self.committed += 1


@dataclass
class Caption:
    y: torch.Tensor          # [1, G, 1, L, C_text]
    mask: torch.Tensor       # [1, G, L]


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
        view_shapes: Optional[ViewShapes] = None,
        view_slot_ids: tuple[int, ...] = (0,),
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        self.model = model
        self.chunk_actions = int(model.chunk_actions)
        self.f_chunk = int(model.latent_frames_per_chunk)
        self.memory_content = str(model.memory_content)
        self.robot_memory = self.memory_content == MEMORY_VIDEO_AND_ACTION
        self.chunk_observation = bool(model.chunk_observation)
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
        self.view_shapes = None if view_shapes is None else tuple((int(h), int(w)) for h, w in view_shapes)
        self.view_slot_ids = tuple(int(slot) for slot in view_slot_ids)
        self.captions = [caption] + ([unconditional_caption] if self.guided else [])
        self.memories: list[CausalMemory] = []
        self.chunk_idx = 0
        self._grid: Optional[tuple[int, int]] = None   # (H, W) of the latent grid / strip, fixed by the sink
        self._latent_like: Optional[torch.Tensor] = None  # device / dtype of the noise (Sana ``_latent_like``)
        self._rope_tables: dict = {}                   # softmax rotary tables per entry of a read
        self.probe = None                              # optional attention_probe.AttentionProbe (generation only)
        self.reset_memory()

    def reset_memory(self) -> None:
        """Forget the episode (fresh GDN states, no sink, no softmax entries in every stream); the captions stay. The
        next call must be ``commit_sink`` of the frame the memory reopens at."""

        self.memories = [CausalMemory(self.model, self.window) for _ in self.captions]
        self.chunk_idx = 0

    # -- helpers -----------------------------------------------------------------------------------------------------

    @property
    def streams(self) -> list[tuple[Caption, CausalMemory]]:
        return list(zip(self.captions, self.memories))

    @property
    def has_sink(self) -> bool:
        return self.memories[0].has_sink

    def _autocast(self, device: torch.device):
        if self.autocast_dtype is None or device.type != "cuda":
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=self.autocast_dtype)

    def _check_latent(self, latent: Optional[torch.Tensor], frames: int, what: str) -> None:
        if latent is None or latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != frames:
            shape = None if latent is None else tuple(latent.shape)
            raise ValueError(f"{what} must be [1, C, {frames}, H, W], got {shape}")
        grid = (int(latent.shape[-2]), int(latent.shape[-1]))
        if self._grid is None:
            self._grid = grid
        elif grid != self._grid:
            raise ValueError(f"{what} has the latent grid {grid}, the session runs {self._grid}")

    def _views(self) -> ViewShapes:
        """The per-view latent shapes the model reads (the one canvas grid unless the session was given a strip's)."""

        if self.view_shapes is not None:
            return self.view_shapes
        patch = self.model.patch_size
        return ((self._grid[0] // patch[1], self._grid[1] // patch[2]),)

    def read_chunk_ids(self) -> list[int]:
        """Committed chunks the next forward reads besides the sink (the same in every stream)."""

        return self.memories[0].read_chunk_ids()

    def softmax_origin(self) -> int:
        """The next read's origin O (the oldest chunk read): the softmax layers number that read from 0 there."""

        return self.memories[0].read_plan()[1]

    def _rope_of(self, slot: int) -> torch.Tensor:
        key = ("chunk", slot)
        if key not in self._rope_tables:
            device = next(self.model.parameters()).device
            self._rope_tables[key] = self.model.softmax_entry_rope(
                frames=self.f_chunk, frame_offset=slot * self.f_chunk + 1, robot_rows=self.model.chunk_robot_rows(),
                step_offset=slot * self.chunk_actions, model_fps=self.model_fps, view_shapes=self._views(),
                view_slot_ids=self.view_slot_ids, device=device,
            )
        return self._rope_tables[key]

    def _sink_rope(self) -> Optional[torch.Tensor]:
        if not self.has_sink:
            return None
        key = ("sink",)
        if key not in self._rope_tables:
            device = next(self.model.parameters()).device
            self._rope_tables[key] = self.model.softmax_entry_rope(
                frames=1, frame_offset=0, robot_rows=0, step_offset=0, model_fps=self.model_fps,
                view_shapes=self._views(), view_slot_ids=self.view_slot_ids, device=device,
            )
        return self._rope_tables[key]

    def _caches(self, memory: CausalMemory, *, commit: bool) -> list[LayerCache]:
        return memory.layer_caches(commit=commit, rope_of=self._rope_of, sink_rope=self._sink_rope())

    def _softmax_offsets(self, memory: CausalMemory, frame_offset: int, step_offset: int) -> tuple[int, int]:
        # the origin of THIS stream's read: a commit runs the streams one after the other, so another stream's memory
        # may already hold the chunk being committed
        origin = memory.read_plan()[1]
        return frame_offset - origin * self.f_chunk, step_offset - origin * self.chunk_actions

    def _forward(self, latent, timesteps, caption: Caption, memory: CausalMemory, *, frame_offset, step_offset, robot,
                 caches):
        soft_frame, soft_step = self._softmax_offsets(memory, frame_offset, step_offset)
        return self.model.forward_window(
            latent, timesteps, caption.y, caption.mask,
            model_fps=self.model_fps, frame_offset=frame_offset, step_offset=step_offset, view_shapes=self.view_shapes,
            view_slot_ids=self.view_slot_ids, robot=robot, caches=caches, softmax_frame_offset=soft_frame,
            softmax_step_offset=soft_step,
        )

    # -- the episode opening -----------------------------------------------------------------------------------------

    @torch.inference_mode()
    def commit_sink(self, sink_latent: torch.Tensor) -> None:
        """Open the episode's memory with its first observation ``[1, C_lat, 1, H, W]`` (Sana ``commit_sink``): one
        clean forward per stream on video frame 0 without robot rows, reading nothing; its GDN state and softmax K/V
        open every layer's memory. The frozen first chunk follows through ``commit_chunk`` (chunk 0)."""

        if self.has_sink or self.chunk_idx:
            raise RuntimeError("the sink opens the memory: commit it once, before the first chunk")
        self._check_latent(sink_latent, 1, "sink_latent")
        self._latent_like = sink_latent
        device = sink_latent.device
        video_t = torch.zeros(1, 1, device=device)
        with self._autocast(device):
            for caption, memory in self.streams:
                caches = self._caches(memory, commit=True)
                self._forward(sink_latent, video_t, caption, memory, frame_offset=0, step_offset=0, robot=None,
                              caches=caches)
                memory.absorb_sink(caches)

    # -- the cycle ---------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def generate_chunk(
        self,
        observation_latent: Optional[torch.Tensor],
        *,
        anchor_state: torch.Tensor,
        anchor_state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        video_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        gripper_bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise chunk ``chunk_idx`` >= 1 from its observation ``[1, C_lat, 1, H, W]`` (None without chunk
        observations) against frozen memory; returns (normalized actions [1, C, 80], target latents
        [1, C_lat, f_c, H, W]). Nothing is written."""

        if not self.has_sink or self.chunk_idx < 1:
            raise RuntimeError(
                "the first emission is chunk 1: commit_sink, then commit_chunk of the frozen first chunk (chunk 0 is "
                "memory only, as the training's padded chunk)"
            )
        if self.chunk_observation:
            self._check_latent(observation_latent, 1, "observation_latent")
            self._latent_like = observation_latent
        elif observation_latent is not None:
            raise ValueError(
                "the model trains without chunk observations (chunk_observation false): pass observation_latent=None, "
                "the chunk reads the current frame from the memory"
            )
        chunk = self.chunk_actions
        action_mask = action_mask.to(dtype=torch.bool)
        if action_mask.ndim != 3 or action_mask.shape[:2] != (1, chunk):
            raise ValueError(f"action_mask must be [1, {chunk}, D], got {tuple(action_mask.shape)}")
        like = self._latent_like
        device, latent_dtype = like.device, like.dtype
        if video_noise is None:
            channels = int(self.model.in_channels)
            video_noise = torch.randn(
                (1, channels, self.f_chunk, *self._grid), generator=generator, device=device, dtype=latent_dtype
            )
        if action_noise is None:
            action_noise = torch.randn(
                (1, chunk, action_mask.shape[2]), generator=generator, device=device, dtype=latent_dtype
            )
        video = video_noise.to(device=device, dtype=latent_dtype)
        self._check_latent(video, self.f_chunk, "video_noise")
        action = action_noise.to(device=device, dtype=latent_dtype).masked_fill(~action_mask.to(device), 0)
        action_mask = action_mask.to(device)
        observation = [] if observation_latent is None else [observation_latent.to(dtype=video.dtype)]
        obs_frames = len(observation)
        video_scheduler = make_scheduler(self.steps, self.flow_shift, device)
        action_scheduler = make_scheduler(self.steps, self.action_flow_shift, device)
        caches_by_stream = [self._caches(memory, commit=False) for memory in self.memories]
        # the observation sits at c * f_c, the targets after it (at c * f_c + 1 without an observation)
        frame_offset = self.chunk_idx * self.f_chunk + (0 if obs_frames else 1)
        step_offset = self.chunk_idx * chunk
        state = anchor_state.reshape(1, -1).to(device)
        state_mask = anchor_state_mask.reshape(1, -1).to(device=device, dtype=torch.bool)
        probe = self.probe
        if probe is not None:
            read_ids, origin = self.memories[0].read_plan()
            patch = self.model.patch_size
            probe.begin_chunk(
                chunk=self.chunk_idx, read_ids=read_ids, origin=origin,
                tokens_per_frame=(self._grid[0] // patch[1]) * (self._grid[1] // patch[2]),
                latent_frames_per_chunk=self.f_chunk, chunk_actions=chunk, robot_memory=self.robot_memory,
                steps=self.steps, streams=len(self.captions), sink=self.has_sink, observation=bool(obs_frames),
            )
        try:
            with self._autocast(device):
                for step_index, timestep in enumerate(video_scheduler.timesteps):
                    action_timestep = action_scheduler.timesteps[step_index]
                    video_t = timestep.reshape(1, 1).expand(1, self.f_chunk).to(torch.float32)
                    action_t = action_timestep.reshape(1, 1).expand(1, chunk).to(torch.float32)
                    latent = torch.cat(observation + [video], dim=2)
                    window_t = torch.cat((video_t.new_zeros(1, obs_frames), video_t), dim=1)
                    robot = {
                        "state80": state,
                        "state_mask80": state_mask,
                        "action80": action,
                        "action_mask80": action_mask,
                        "action_timesteps": action_t,
                    }
                    outputs = []
                    for stream_index, (caption, memory, caches) in enumerate(
                        zip(self.captions, self.memories, caches_by_stream)
                    ):
                        if probe is not None:
                            probe.set_forward(step_index, stream_index, float(timestep), float(action_timestep))
                        outputs.append(
                            self._forward(
                                latent, window_t, caption, memory, frame_offset=frame_offset, step_offset=step_offset,
                                robot=robot, caches=caches,
                            )
                        )
                    cond = outputs[0]
                    uncond = outputs[1] if self.guided else None
                    targets = slice(obs_frames, None)
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
        finally:
            if probe is not None:
                probe.pause()
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
        """Fold chunk ``chunk_idx`` into memory (one clean forward per stream) and advance: chunk 0 is the frozen first
        chunk (its rows hold the robot at its first state), every later chunk the EXECUTED one. Under
        ``memory_content`` video_only the memory keeps the target latents alone: the actions and the state are checked
        but not committed (Sana's session does the same)."""

        if not self.has_sink:
            raise RuntimeError("commit_sink first: the episode's first observation opens the memory")
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
                "state80": anchor_state.reshape(1, -1).to(device),
                "state_mask80": anchor_state_mask.reshape(1, -1).to(device=device, dtype=torch.bool),
                "action80": executed_action.to(device).masked_fill(~action_mask.to(device), 0),
                "action_mask80": action_mask.to(device),
                "action_timesteps": torch.zeros(1, chunk, device=device),
            }
        video_t = torch.zeros(1, self.f_chunk, device=device)
        frame_offset = self.chunk_idx * self.f_chunk + 1
        step_offset = self.chunk_idx * chunk
        with self._autocast(device):
            for caption, memory in self.streams:
                caches = self._caches(memory, commit=True)
                self._forward(
                    chunk_latent, video_t, caption, memory, frame_offset=frame_offset, step_offset=step_offset,
                    robot=robot, caches=caches,
                )
                memory.absorb(caches)
        self.chunk_idx += 1


__all__ = ["Caption", "CausalMemory", "CausalPolicySession"]
