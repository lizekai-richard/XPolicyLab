"""The per-episode deploy cycle of the hierarchical-memory policy on a loaded ``HMPolicyModel`` (batch 1).

Sana keeps no deploy session for this mode yet (``causal_sparse_memory_plan.md`` sections 5, 12, 14 list the inference
side as requirements); this session implements them. Decision c anchors at row t = r0 + 32c (r0 = the episode's first
observation, tick 0) and denoises the noisy chunk [obs | target] + [state | 32 actions] -- the target alone + the rows
under ``chunk_observation: false`` (Sana 6564adc38), where the current frame enters through the memory only -- with:

* the sparse memory of decision c (when a block kind reads it): ``memory.sparse_rows(t, r0, m)`` encoded as one causal
  sequence [A_0, A_1..] -- new every decision, since the sampled rows move with t;
* the dense memory (when a block kind reads it): L_0 (the first frame alone) and one latent per executed chunk, the
  incremental encode of one persistent causal stream (``vae_stream.LatentStream``), handed in with ``add_dense``.

Two ways to evaluate a decision (the ``deploy_form`` knob):

* ``streaming`` (the default, every memory choice): every memory latent is COMMITTED ONCE through the blocks that read
  its memory (``memory.stream_layout``): a dense latent when it is produced (the dense memory's GDN state and / or
  softmax K / V grow over the episode), the sparse memory once per decision (its GDN state and / or softmax K / V start
  afresh); every denoising step then runs the noisy chunk alone, each block reading its memory's committed state / K /
  V. This is exact because a memory token never reads the noisy chunk (the GDN scan reads it first, the softmax window
  is prefix-causal), its t is 0, and without AttnRes time conditioning its depth mixing is its own. With a dense memory
  the positions sit on the episode clock (rows from r0: L_j at 32j, decision c's sparse latents at their sampled rows,
  obs at 32c, the target at 32c + 32, the robot rows from step 32c) -- the training geometry shifted by t - r0 for every
  token, so equal to training up to floating-point round-off; without one, on the training clock (offsets from t)
  exactly;
* ``full`` (reference): every denoising step runs the TRAINING forward of the decision's sample [L_0..L_c | A_0.. | obs |
  target] + rows (no obs without ``chunk_observation``) on the training clock -- the training computation itself, at a
  cost that grows with c when a dense memory is read.

Conditional and unconditional text streams hold separate memories (the caption reaches every block through
cross-attention, so a memory token's hidden states depend on it). Latents are ``[1, C_lat, F, H, W]`` canvas grids;
actions and states are in the normalized space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.sampler import _action_step, _video_step, make_scheduler  # noqa: E402

from . import memory as hm  # noqa: E402
from .contract import DEPLOY_FORMS, DEPLOY_FULL, DEPLOY_STREAMING  # noqa: E402
from .layers import GDN_KIND, GDNCache, SoftmaxCache  # noqa: E402
from .model import HMPolicyModel  # noqa: E402

LEFT_GRIPPER, RIGHT_GRIPPER = 16, 45


@dataclass
class Caption:
    y: torch.Tensor          # [1, 1, 1, L, C_text]
    mask: torch.Tensor       # [1, 1, L]


@dataclass
class CommittedMemory:
    """What one memory has left in the blocks that read it, for one text stream: every GDN layer's (S, z) and every
    softmax layer's (k post-RoPE, v)."""

    gdn: dict = field(default_factory=dict)
    kv: dict = field(default_factory=dict)
    latents: int = 0


class HMSession:
    """The decision cycle of one episode."""

    def __init__(
        self,
        model: HMPolicyModel,
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
    ) -> None:
        self.model = model
        self.memory = dict(model.memory)
        self.m = int(self.memory["sparse_frames"])
        self.streams = hm.memory_streams(self.memory)
        self.chunk_observation = hm.observes(self.memory)
        if deploy_form not in DEPLOY_FORMS:
            raise ValueError(f"deploy_form must be one of {DEPLOY_FORMS}, got {deploy_form!r}")
        self.deploy_form = deploy_form
        self.episode_clock = hm.episode_clock(self.streams)
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
        self.block_kinds = [hm.GDN if block.attn_type == GDN_KIND else hm.SOFTMAX for block in model.blocks]
        # committed[stream][memory kind]
        self.committed: list[dict[str, CommittedMemory]] = [{kind: CommittedMemory() for kind in self.streams} for _ in self.captions]
        self.dense_latents: list[torch.Tensor] = []
        self.dense_count = 0
        self.decision = 0
        self._grid: Optional[tuple[int, int]] = None
        self._last_dense: Optional[torch.Tensor] = None

    # -- helpers -----------------------------------------------------------------------------------------------------

    @property
    def reads_dense(self) -> bool:
        return hm.DENSE in self.streams

    @property
    def reads_sparse(self) -> bool:
        return hm.SPARSE in self.streams

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

    def _commit(self, kind: str, latents: torch.Tensor, rows: list[float], *, fresh: bool) -> None:
        """Commit memory ``kind``'s ``latents`` (at ``rows``) once per text stream through the blocks that read it,
        continuing that memory's committed state / K / V, or starting it afresh."""

        device = latents.device
        offsets = hm.offsets_tensor(rows).to(device)
        zeros = torch.zeros(1, latents.shape[2], device=device)
        with self._autocast(device):
            for stream, caption in enumerate(self.captions):
                held = CommittedMemory() if fresh else self.committed[stream][kind]
                caches = []
                for index, block in enumerate(self.block_kinds):
                    if self.memory[block] != kind:
                        caches.append(None)
                    elif block == hm.GDN:
                        caches.append(GDNCache(seed=held.gdn.get(index), commit=True))
                    else:
                        caches.append(SoftmaxCache(prefix=held.kv.get(index), commit=True))
                self.model.forward_hm(
                    latents, zeros, caption.y, caption.mask, form=hm.FORM_MEMORY, k=0, frame_offsets_x8=offsets,
                    model_fps=self.model_fps, caches=caches, memory_kind=kind,
                )
                updated = CommittedMemory(latents=held.latents + int(latents.shape[2]))
                for index, cache in enumerate(caches):
                    if cache is None:
                        continue
                    if cache.out is None:
                        raise RuntimeError(f"layer {index}: a memory commit left nothing")
                    (updated.gdn if isinstance(cache, GDNCache) else updated.kv)[index] = cache.out
                self.committed[stream][kind] = updated

    def _noisy_caches(self, stream: int) -> list:
        """Every block reads its memory's committed state (GDN) or K / V (softmax); nothing is written."""

        caches = []
        for index, block in enumerate(self.block_kinds):
            held = self.committed[stream][self.memory[block]]
            if block == hm.GDN:
                caches.append(GDNCache(seed=held.gdn.get(index)))
            else:
                caches.append(SoftmaxCache(prefix=held.kv.get(index)))
        return caches

    # -- the dense memory ---------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def add_dense(self, latent: torch.Tensor) -> None:
        """Absorb the dense stream's next latent L_j (j = the number absorbed so far): L_0 = the first frame alone at the
        episode start, then L_c once chunk c - 1 has executed (before decision c). Streaming commits it once (episode
        clock, row 32j); full keeps it for the training sequence."""

        if not self.reads_dense:
            raise RuntimeError(f"this checkpoint reads no dense memory ({self.memory})")
        self._check_latent(latent, 1, "a dense latent")
        j = self.dense_count
        if j != self.decision:
            raise RuntimeError(f"L_{j} must be absorbed right before decision {j}; the session is at decision {self.decision}")
        if self.deploy_form == DEPLOY_STREAMING:
            self._commit(hm.DENSE, latent, [hm.commit_rows(j)], fresh=j == 0)
        else:
            self.dense_latents.append(latent)
        self._last_dense = latent
        self.dense_count += 1

    # -- one decision -------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        observation_latent: Optional[torch.Tensor],
        sparse_latents: Optional[torch.Tensor] = None,
        *,
        anchor_state: torch.Tensor,
        anchor_state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        video_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        gripper_bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise decision ``self.decision``'s chunk from its observation ``[1, C, 1, H, W]`` (None without
        ``chunk_observation``: the noisy chunk is the target alone) and (when read) its sparse memory
        ``[1, C, 1 + m/8, H, W]``; returns (normalized actions [1, C, 80], target latent [1, C, 1, H, W]) and advances to
        the next decision."""

        c = self.decision
        if self.chunk_observation:
            if observation_latent is None:
                raise ValueError(f"decision {c} opens its noisy chunk with the observation latent (chunk_observation)")
            self._check_latent(observation_latent, 1, "observation_latent")
        elif observation_latent is not None:
            raise ValueError(
                "this checkpoint's noisy chunk is the target alone (chunk_observation false); got an observation latent"
            )
        if self.reads_sparse:
            if sparse_latents is None:
                raise ValueError(f"decision {c} needs its sparse memory (the blocks read {self.memory})")
            self._check_latent(sparse_latents, hm.sparse_latents(self.m), "sparse_latents")
        elif sparse_latents is not None:
            raise ValueError(f"this checkpoint reads no sparse memory ({self.memory}); got sparse latents")
        if self.reads_dense and self.dense_count != c + 1:
            raise RuntimeError(f"decision {c} reads the dense latents L_0..L_{c}; {self.dense_count} were absorbed")
        chunk = self.chunk_actions
        action_mask = action_mask.to(dtype=torch.bool)
        if action_mask.ndim != 3 or action_mask.shape[:2] != (1, chunk):
            raise ValueError(f"action_mask must be [1, {chunk}, D], got {tuple(action_mask.shape)}")
        # the latent the noise follows (device, dtype, shape): the observation, else the sparse memory, else L_c
        like = next(latent for latent in (observation_latent, sparse_latents, self._last_dense) if latent is not None)
        device = like.device
        if video_noise is None:
            video_noise = torch.randn((1, like.shape[1], 1, *like.shape[3:]), generator=generator, device=device, dtype=like.dtype)
        if action_noise is None:
            action_noise = torch.randn((1, chunk, action_mask.shape[2]), generator=generator, device=device, dtype=like.dtype)
        video = video_noise.to(device=device, dtype=like.dtype)
        action = action_noise.to(device=device, dtype=like.dtype).masked_fill(~action_mask, 0)

        if self.deploy_form == DEPLOY_STREAMING:
            if self.reads_sparse:
                self._commit(hm.SPARSE, sparse_latents.to(dtype=video.dtype), hm.sparse_memory_rows(c, self.m, self.episode_clock), fresh=True)
            prefix = []
            form, k = hm.FORM_NOISY, 0
            rows = hm.noisy_rows(c, self.episode_clock, self.chunk_observation)
            step_offset = hm.noisy_step_offset(c, self.episode_clock)
        else:
            prefix = list(self.dense_latents) if self.reads_dense else []
            if self.reads_sparse:
                prefix.append(sparse_latents)
            form, k = hm.FORM_TRAIN, c
            rows = hm.train_rows(c, self.m, self.streams, self.chunk_observation)
            step_offset = 0
        if observation_latent is not None:
            prefix.append(observation_latent)
        context = torch.cat([p.to(dtype=video.dtype) for p in prefix], dim=2) if prefix else None
        offsets = hm.offsets_tensor(rows).to(device)
        frames = (0 if context is None else context.shape[2]) + 1
        if offsets.shape[1] != frames:
            raise RuntimeError(f"decision {c}: {frames} latent frames but {offsets.shape[1]} frame rows")
        video_scheduler = make_scheduler(self.steps, self.flow_shift, device)
        action_scheduler = make_scheduler(self.steps, self.action_flow_shift, device)
        state = anchor_state.reshape(1, -1)
        state_mask = anchor_state_mask.reshape(1, -1).to(torch.bool)
        with self._autocast(device):
            for step_index, timestep in enumerate(video_scheduler.timesteps):
                action_timestep = action_scheduler.timesteps[step_index]
                video_t = timestep.reshape(1, 1).to(torch.float32)
                action_t = action_timestep.reshape(1, 1).expand(1, chunk).to(torch.float32)
                latent = video if context is None else torch.cat((context, video), dim=2)
                window_t = torch.cat((video_t.new_zeros(1, frames - 1), video_t), dim=1)
                robot = {
                    "state80": state,
                    "state_mask80": state_mask,
                    "action80": action,
                    "action_mask80": action_mask,
                    "action_timesteps": action_t,
                }
                outputs = []
                for stream, caption in enumerate(self.captions):
                    outputs.append(
                        self.model.forward_hm(
                            latent, window_t, caption.y, caption.mask,
                            form=form, k=k, frame_offsets_x8=offsets, model_fps=self.model_fps, step_offset=step_offset,
                            robot=robot, caches=self._noisy_caches(stream) if form == hm.FORM_NOISY else None,
                        )
                    )
                cond = outputs[0]
                uncond = outputs[1] if self.guided else None
                video_prediction = cond["x"][:, :, -1:]
                if uncond is not None and self.video_cfg_scale > 1:
                    video_prediction = uncond["x"][:, :, -1:] + self.video_cfg_scale * (
                        video_prediction - uncond["x"][:, :, -1:]
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
        self.decision += 1
        return action, video


__all__ = ["Caption", "CommittedMemory", "HMSession"]
