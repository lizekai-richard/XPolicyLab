"""Geometry of the sparse memory (Sana rwm/sparse-memory ``dev/rwm/diffusion/model/utils/sparse_memory.py``, checked
against 3fa3e9689), the token layout of the three forwards (the net's ``_forward_multiview`` / ``_sparse_memory_layers``
and the deploy entries ``forward_memory_prefix`` / ``forward_memory_target``) and the deploy clock.

Rows are 25 Hz control steps; a span opens at its first row r_0 (the deploy's first observation, tick 0). A sample (one
decision) anchored at the observation row t is ONE causal VAE encode of the canvases of ``sequence_rows`` -- the sink
(row r_0), the memory rows of ``chunk_causal_policy.memory_type``, the 8 target rows t + 4i -- into the latents
[S | M_1..M_n | T], followed by the 33 robot rows [state at t | 32 actions]:

* ``memory_type: strided`` (``memory_stride`` s): the rows t, t - s, t - 2s, ... above r_0 in time order, led by copies of
  r_0 up to a multiple of 8 (the causal VAE folds 8 frames into a latent), so n = ceil(ceil((t - r_0) / s) / 8) grows
  with t (0 at t = r_0);
* ``memory_type: fixed`` (``memory_frames`` m, a multiple of 8): m rows sampled uniformly over [r_0 + 1, t], led by
  copies of r_0 while the span holds fewer than m rows, so n = m / 8 at every row (the rows of the hierarchical memory's
  sparse stream).

Either way the last memory row is t once the span has a row past r_0: the current frame reaches the policy through the
memory (there is no observation latent). Every block (GDN and softmax) runs on the whole sequence with one chunk per
sink / memory latent and the target chunk [T | state | actions] last; RoPE counts latents (latent i at frame i, the
target at n + 1) and starts the robot rows at action step 32n (the state on latent n, the last action on the target).

The functions down to ``video_timesteps`` are Sana's (names, arguments, results), so the live tests compare them one to
one; the rest lays out the adapter's forwards and the deploy clock.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

CHUNK_ROWS = 32
FRAME_STRIDE = 4
FRAMES_PER_LATENT = CHUNK_ROWS // FRAME_STRIDE
TOKENS_PER_FRAME = 160
ROBOT_ROWS = 1 + CHUNK_ROWS

MEMORY_TYPE = "memory_type"
MEMORY_FRAMES = "memory_frames"
MEMORY_STRIDE = "memory_stride"
MEMORY_KEYS = (MEMORY_TYPE, MEMORY_FRAMES, MEMORY_STRIDE)
FIXED = "fixed"
STRIDED = "strided"
MEMORY_TYPES = (FIXED, STRIDED)

# the adapter's forwards: the training sequence, and the deploy's prefix / target split of it
FORM_SEQUENCE = "sequence"
FORM_PREFIX = "prefix"
FORM_TARGET = "target"
FORMS = (FORM_SEQUENCE, FORM_PREFIX, FORM_TARGET)


def get_extra_value(extra, key, default):
    """Read a key from ``config.model.extra``, a dict or a namespace (Sana ``causal_rwm_train.get_extra_value``)."""

    if extra is None:
        return default
    if isinstance(extra, Mapping):
        return extra.get(key, default)
    return getattr(extra, key, default)


def memory_from_policy(policy) -> dict | None:
    """The sparse memory ``model.extra.chunk_causal_policy`` declares: ``{"memory_type": "fixed", "memory_frames": m}``
    or ``{"memory_type": "strided", "memory_stride": s}``; None for a policy without a memory.

    Raises:
        ValueError: a memory_type other than fixed or strided, memory_frames that is not a positive multiple of 8, or
            memory_frames / memory_stride without a memory_type (the policy would run as one without a memory).
    """

    memory_type = get_extra_value(policy, MEMORY_TYPE, None)
    if memory_type is None:
        declared = [key for key in (MEMORY_FRAMES, MEMORY_STRIDE) if get_extra_value(policy, key, None) is not None]
        if declared:
            raise ValueError(f"chunk_causal_policy declares {declared} without memory_type (fixed or strided)")
        return None
    if memory_type == FIXED:
        memory_frames = int(get_extra_value(policy, MEMORY_FRAMES, None))
        if memory_frames <= 0 or memory_frames % FRAMES_PER_LATENT:
            raise ValueError(
                f"chunk_causal_policy.memory_frames must be a positive multiple of {FRAMES_PER_LATENT}, got "
                f"{memory_frames}"
            )
        return {MEMORY_TYPE: FIXED, MEMORY_FRAMES: memory_frames}
    if memory_type == STRIDED:
        return {MEMORY_TYPE: STRIDED, MEMORY_STRIDE: int(get_extra_value(policy, MEMORY_STRIDE, None))}
    raise ValueError(f"chunk_causal_policy.memory_type must be fixed or strided, got {memory_type!r}")


def fixed_rows(obs_row: int, first_row: int, memory_frames: int) -> list[int]:
    """``memory_frames`` rows sampled uniformly over [first_row + 1, obs_row] in time order: with m = memory_frames and
    n = obs_row - first_row candidates, first_row + 1 + floor(i (n - 1) / (m - 1) + 1/2) for n >= m (the last one
    obs_row), else the m - n missing rows as copies of first_row ahead of first_row + 1..obs_row."""

    obs_row, first_row, memory_frames = int(obs_row), int(first_row), int(memory_frames)
    candidates = obs_row - first_row
    if candidates < memory_frames:
        return [first_row] * (memory_frames - candidates) + list(range(first_row + 1, obs_row + 1))
    return [
        first_row + 1 + (2 * index * (candidates - 1) + memory_frames - 1) // (2 * (memory_frames - 1))
        for index in range(memory_frames)
    ]


def strided_rows(obs_row: int, first_row: int, memory_stride: int) -> list[int]:
    """obs_row, obs_row - s, obs_row - 2s, ... above first_row in time order, preceded by copies of first_row up to a
    multiple of 8."""

    strided = list(range(int(obs_row), int(first_row), -int(memory_stride)))[::-1]
    return [int(first_row)] * (-len(strided) % FRAMES_PER_LATENT) + strided


def strided_latents(obs_row: int, first_row: int, memory_stride: int) -> int:
    """The strided memory latent count n = ceil(ceil((obs_row - first_row) / s) / 8)."""

    frames = -(-(int(obs_row) - int(first_row)) // int(memory_stride))
    return -(-frames // FRAMES_PER_LATENT)


def memory_rows(obs_row: int, first_row: int, memory: dict) -> list[int]:
    """The memory rows of an observation row under ``memory`` (``memory_from_policy``), 8 per memory latent."""

    if memory[MEMORY_TYPE] == FIXED:
        return fixed_rows(obs_row, first_row, memory[MEMORY_FRAMES])
    return strided_rows(obs_row, first_row, memory[MEMORY_STRIDE])


def memory_latents(obs_row: int, first_row: int, memory: dict) -> int:
    """n, the memory latents between the sink and the target: memory_frames / 8 (fixed), the strided count otherwise."""

    if memory[MEMORY_TYPE] == FIXED:
        return int(memory[MEMORY_FRAMES]) // FRAMES_PER_LATENT
    return strided_latents(obs_row, first_row, memory[MEMORY_STRIDE])


def target_rows(obs_row: int, last_row: int | None = None) -> list[int]:
    """The 8 target rows obs_row + 4i (i = 1..8), the frames of the target latent; a row past ``last_row`` (a tail
    sample's span end) repeats it (the freeze padding of the SFT tail windows)."""

    rows = [int(obs_row) + FRAME_STRIDE * i for i in range(1, FRAMES_PER_LATENT + 1)]
    return rows if last_row is None else [min(row, int(last_row)) for row in rows]


def sequence_rows(obs_row: int, first_row: int, memory: dict, last_row: int | None = None) -> list[int]:
    """The rows of a sample's causal VAE encode: the sink (first_row), the memory rows, the target rows."""

    return [int(first_row)] + memory_rows(obs_row, first_row, memory) + target_rows(obs_row, last_row)


def sampler_rung(memory_latents: int) -> int:
    """The sampler rung 1 + (n + 1) * 32 of a sample with n memory latents (one bucket per n)."""

    return 1 + (int(memory_latents) + 1) * CHUNK_ROWS


def chunk_bounds(frames: int) -> list[int]:
    """Chunk bounds of ``frames`` latent frames followed by the robot rows: one 160-token chunk per sink and memory
    latent, then the target latent with the 33 robot rows."""

    return [TOKENS_PER_FRAME * index for index in range(int(frames))] + [TOKENS_PER_FRAME * int(frames) + ROBOT_ROWS]


def pixels_from_uint8(frames: torch.Tensor, dtype: torch.dtype, device=None) -> torch.Tensor:
    """uint8 canvases ``[B, F, 3, H, W]`` as the causal VAE's ``[B, 3, F, H, W]`` input in [-1, 1] (float32
    normalization, then the cast)."""

    pixels = frames.to(device=device).permute(0, 2, 1, 3, 4).to(torch.float32)
    return pixels.div_(127.5).sub_(1.0).to(dtype).contiguous()


def video_timesteps(target_timesteps: torch.Tensor, frames: int) -> torch.Tensor:
    """``[B, 1, F]`` video timesteps: 0 on the sink and memory latents, ``target_timesteps`` on the target."""

    out = target_timesteps.new_zeros(target_timesteps.shape[0], 1, int(frames))
    out[:, 0, -1] = target_timesteps
    return out


# -- the adapter's forwards ------------------------------------------------------------------------------------------


def prefix_rows(obs_row: int, first_row: int, memory: dict) -> list[int]:
    """The rows of a decision's prefix encode (Sana ``SparseMemoryEpisode.prefix_rows``): the sink, then the memory
    rows -- ``sequence_rows`` without the target rows."""

    return [int(first_row)] + memory_rows(obs_row, first_row, memory)


def prefix_bounds(frames: int) -> list[int]:
    """Chunk bounds of a prefix of ``frames`` latents without robot rows: one 160-token chunk per latent (the net's
    ``_sparse_memory_layers`` without robot rows)."""

    return [TOKENS_PER_FRAME * index for index in range(int(frames) + 1)]


def form_layout(form: str, frames: int, memory_latents: int | None = None) -> dict:
    """The layout of one forward of ``form`` holding ``frames`` latent frames:

    * ``sequence``: [S | M_1..M_n | T] + robot rows (F = n + 2), the training forward: latent i at RoPE frame i, the robot
      rows from step 32n, the chunk bounds ``chunk_bounds(F)``;
    * ``prefix``: [S | M_1..M_n] (F = n + 1), clean and without robot rows: latent i at frame i, one chunk per latent;
    * ``target``: [T] + robot rows (F = 1): T at frame n + 1 and the rows from step 32n, their places in the training
      sequence of n memory latents (``memory_latents``), one chunk.

    Returns ``{"frames", "rows", "phi" (the RoPE frame of every latent), "step_offset", "bounds", "memory_latents"}``.
    """

    frames = int(frames)
    if form == FORM_SEQUENCE:
        if frames < 2:
            raise ValueError(f"a sparse-memory sequence holds the sink and the target latent at least, got {frames} frames")
        n = frames - 2
        return {"frames": frames, "rows": ROBOT_ROWS, "phi": list(range(frames)), "step_offset": CHUNK_ROWS * n,
                "bounds": chunk_bounds(frames), "memory_latents": n}
    if form == FORM_PREFIX:
        if frames < 1:
            raise ValueError("a sparse-memory prefix holds the sink at least")
        return {"frames": frames, "rows": 0, "phi": list(range(frames)), "step_offset": 0,
                "bounds": prefix_bounds(frames), "memory_latents": frames - 1}
    if form == FORM_TARGET:
        if frames != 1:
            raise ValueError(f"the target chunk holds the target latent alone, got {frames} frames")
        if memory_latents is None or int(memory_latents) < 0:
            raise ValueError("a target forward needs the decision's memory latent count n >= 0")
        n = int(memory_latents)
        return {"frames": 1, "rows": ROBOT_ROWS, "phi": [n + 1], "step_offset": CHUNK_ROWS * n,
                "bounds": chunk_bounds(1), "memory_latents": n}
    raise ValueError(f"unknown sparse-memory form {form!r}; expected one of {FORMS}")


# -- the deploy clock ------------------------------------------------------------------------------------------------


def decision_row(c: int, first_row: int = 0) -> int:
    """Decision c's observation row t = r_0 + 32c: RoboDojo asks for the next chunk once the previous 32 actions ran."""

    return int(first_row) + CHUNK_ROWS * int(c)


__all__ = [
    "CHUNK_ROWS",
    "FIXED",
    "FORMS",
    "FORM_PREFIX",
    "FORM_SEQUENCE",
    "FORM_TARGET",
    "FRAMES_PER_LATENT",
    "FRAME_STRIDE",
    "MEMORY_FRAMES",
    "MEMORY_KEYS",
    "MEMORY_STRIDE",
    "MEMORY_TYPE",
    "MEMORY_TYPES",
    "ROBOT_ROWS",
    "STRIDED",
    "TOKENS_PER_FRAME",
    "chunk_bounds",
    "decision_row",
    "fixed_rows",
    "form_layout",
    "get_extra_value",
    "memory_from_policy",
    "memory_latents",
    "memory_rows",
    "pixels_from_uint8",
    "prefix_bounds",
    "prefix_rows",
    "sampler_rung",
    "sequence_rows",
    "strided_latents",
    "strided_rows",
    "target_rows",
    "video_timesteps",
]
