"""Geometry of the sparse memory (Sana rwm/sparse-memory ``dev/rwm/diffusion/model/utils/sparse_memory.py``, checked
against cc18ad78f), the token layout of the three forwards (the net's ``_forward_multiview`` / ``_sparse_memory_layers``
and the deploy entries ``forward_memory_prefix`` / ``forward_memory_target``) and the deploy clock.

Rows are 25 Hz control steps; a span opens at its first row r_0 (the deploy's first observation, tick 0). A sample (one
decision) anchored at the observation row t is ONE causal VAE encode of the frames (the canvas, or every sana_latent
view) of ``sequence_rows`` -- the sink (row r_0), the ``memory_frames`` m memory rows of
``chunk_causal_policy.memory_type``, the 8 target rows t + 4i -- into the latents [S | M_1..M_n | T] (n = m / 8 at
every row), followed by the 33 robot rows [state at t | 32 actions]:

* ``memory_type: global`` (``memory_frames`` m): m rows sampled uniformly over [r_0 + 1, t], led by copies of r_0 while
  the span holds fewer than m rows (Sana eef7beeab renamed it from ``fixed``, whose rows were the same: a config of
  that name, e.g. the fixed-16 run's steps 20k-40k, is read as global here, where Sana now refuses it);
* ``memory_type: strided`` (``memory_frames`` m and ``memory_stride`` s, since eef7beeab): the m rows t - (m - 1) s, ...,
  t - s, t, those at or before r_0 copies of r_0 ahead of the rest. The unbounded strided memory before eef7beeab (every
  s rows back to r_0, no ``memory_frames``) is retired upstream and refused.

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
TOKENS_PER_FRAME = 160          # the 320 x 512 canvas (10 x 16 latent tokens); the sana_latent strip's is sum(h_v * w_v)
ROBOT_ROWS = 1 + CHUNK_ROWS

MEMORY_TYPE = "memory_type"
MEMORY_FRAMES = "memory_frames"
MEMORY_STRIDE = "memory_stride"
MEMORY_KEYS = (MEMORY_TYPE, MEMORY_FRAMES, MEMORY_STRIDE)
GLOBAL = "global"
STRIDED = "strided"
MEMORY_TYPES = (GLOBAL, STRIDED)
# the name of the global memory before Sana eef7beeab (the same rows): read as global
LEGACY_FIXED = "fixed"
SEQUENCE_ATTENTION = "sequence_attention"

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
    """The sparse memory ``model.extra.chunk_causal_policy`` declares: ``memory_frames`` m rows between the sink and the
    target, taken by ``memory_type``: ``{"memory_type": "global", "memory_frames": m}`` samples them uniformly over the
    history, ``{"memory_type": "strided", "memory_frames": m, "memory_stride": s}`` takes every s-th row back from the
    observation row; None for a policy without a memory. ``memory_type: fixed`` (the global memory's name before Sana
    eef7beeab, the same rows) is read as global.

    Raises:
        ValueError: a memory_type other than global, fixed or strided; memory_frames that is not a positive multiple of 8
            (the causal VAE folds 8 frames into a latent); a strided memory without a positive memory_stride (the
            unbounded strided memory before eef7beeab, which declares no memory_frames, is refused by the first rule),
            or a global one with a memory_stride; memory_frames / memory_stride without a memory_type.
    """

    memory_type = get_extra_value(policy, MEMORY_TYPE, None)
    memory_stride = get_extra_value(policy, MEMORY_STRIDE, None)
    if memory_type is None:
        declared = [key for key in (MEMORY_FRAMES, MEMORY_STRIDE) if get_extra_value(policy, key, None) is not None]
        if declared:
            raise ValueError(f"chunk_causal_policy declares {declared} without memory_type (global or strided)")
        return None
    if memory_type == LEGACY_FIXED:
        memory_type = GLOBAL
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"chunk_causal_policy.memory_type must be global or strided, got {memory_type!r}")
    memory_frames = int(get_extra_value(policy, MEMORY_FRAMES, None) or 0)
    if memory_frames <= 0 or memory_frames % FRAMES_PER_LATENT:
        raise ValueError(
            f"chunk_causal_policy.memory_frames must be a positive multiple of {FRAMES_PER_LATENT}, got {memory_frames}"
            + (" (the unbounded strided memory before Sana eef7beeab is retired)" if memory_type == STRIDED else "")
        )
    if memory_type == GLOBAL:
        if memory_stride is not None:
            raise ValueError("chunk_causal_policy.memory_stride belongs to a strided memory; a global one is uniform")
        return {MEMORY_TYPE: GLOBAL, MEMORY_FRAMES: memory_frames}
    if memory_stride is None or int(memory_stride) <= 0:
        raise ValueError(f"a strided memory needs a positive chunk_causal_policy.memory_stride, got {memory_stride!r}")
    return {MEMORY_TYPE: STRIDED, MEMORY_FRAMES: memory_frames, MEMORY_STRIDE: int(memory_stride)}


def full_sequence_attention(policy) -> bool:
    """Whether ``chunk_causal_policy.sequence_attention`` asks for full attention over the whole sequence (``full``)
    instead of one causal chunk per latent (``causal``, also when the key is absent).

    Raises:
        ValueError: a value other than causal or full.
    """

    value = get_extra_value(policy, SEQUENCE_ATTENTION, "causal")
    if value not in ("causal", "full"):
        raise ValueError(f"chunk_causal_policy.sequence_attention must be causal or full, got {value!r}")
    return value == "full"


def global_rows(obs_row: int, first_row: int, memory_frames: int) -> list[int]:
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


def strided_rows(obs_row: int, first_row: int, memory_frames: int, memory_stride: int) -> list[int]:
    """The ``memory_frames`` rows obs_row - (m - 1) s, ..., obs_row - s, obs_row in time order; those at or before
    first_row are copies of first_row ahead of the rest, as in ``global_rows``."""

    strided = list(range(int(obs_row), int(first_row), -int(memory_stride)))[: int(memory_frames)][::-1]
    return [int(first_row)] * (int(memory_frames) - len(strided)) + strided


def memory_rows(obs_row: int, first_row: int, memory: dict) -> list[int]:
    """The ``memory_frames`` memory rows of an observation row under ``memory`` (``memory_from_policy``)."""

    if memory[MEMORY_TYPE] == GLOBAL:
        return global_rows(obs_row, first_row, memory[MEMORY_FRAMES])
    return strided_rows(obs_row, first_row, memory[MEMORY_FRAMES], memory[MEMORY_STRIDE])


def memory_latents(memory: dict) -> int:
    """n = memory_frames / 8, the memory latents between the sink and the target, the same at every row."""

    return int(memory[MEMORY_FRAMES]) // FRAMES_PER_LATENT


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


def chunk_bounds(frames: int, frame_tokens: int = TOKENS_PER_FRAME, robot_rows: int = ROBOT_ROWS) -> list[int]:
    """Chunk bounds of ``frames`` latent frames of ``frame_tokens`` tokens each (160 on the canvas, the views' sum
    ``sum(h_v * w_v)`` on the sana_latent strip) followed by ``robot_rows`` robot rows: one chunk per sink and memory
    latent, then the target latent with the robot rows (Sana ``sparse_memory.chunk_bounds``)."""

    frames, frame_tokens, robot_rows = int(frames), int(frame_tokens), int(robot_rows)
    return [frame_tokens * index for index in range(frames)] + [frame_tokens * frames + robot_rows]


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


def prefix_bounds(frames: int, frame_tokens: int = TOKENS_PER_FRAME) -> list[int]:
    """Chunk bounds of a prefix of ``frames`` latents without robot rows: one ``frame_tokens`` chunk per latent (the
    net's ``_sparse_memory_layers`` without robot rows)."""

    return chunk_bounds(frames, frame_tokens, 0)


def form_layout(
    form: str, frames: int, memory_latents: int | None = None, frame_tokens: int = TOKENS_PER_FRAME
) -> dict:
    """The layout of one forward of ``form`` holding ``frames`` latent frames of ``frame_tokens`` tokens each:

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
                "bounds": chunk_bounds(frames, frame_tokens), "memory_latents": n}
    if form == FORM_PREFIX:
        if frames < 1:
            raise ValueError("a sparse-memory prefix holds the sink at least")
        return {"frames": frames, "rows": 0, "phi": list(range(frames)), "step_offset": 0,
                "bounds": prefix_bounds(frames, frame_tokens), "memory_latents": frames - 1}
    if form == FORM_TARGET:
        if frames != 1:
            raise ValueError(f"the target chunk holds the target latent alone, got {frames} frames")
        if memory_latents is None or int(memory_latents) < 0:
            raise ValueError("a target forward needs the decision's memory latent count n >= 0")
        n = int(memory_latents)
        return {"frames": 1, "rows": ROBOT_ROWS, "phi": [n + 1], "step_offset": CHUNK_ROWS * n,
                "bounds": chunk_bounds(1, frame_tokens), "memory_latents": n}
    raise ValueError(f"unknown sparse-memory form {form!r}; expected one of {FORMS}")


# -- the deploy clock ------------------------------------------------------------------------------------------------


def decision_row(c: int, first_row: int = 0) -> int:
    """Decision c's observation row t = r_0 + 32c: RoboDojo asks for the next chunk once the previous 32 actions ran."""

    return int(first_row) + CHUNK_ROWS * int(c)


__all__ = [
    "CHUNK_ROWS",
    "GLOBAL",
    "LEGACY_FIXED",
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
    "SEQUENCE_ATTENTION",
    "STRIDED",
    "TOKENS_PER_FRAME",
    "chunk_bounds",
    "decision_row",
    "full_sequence_attention",
    "global_rows",
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
    "strided_rows",
    "target_rows",
    "video_timesteps",
]
