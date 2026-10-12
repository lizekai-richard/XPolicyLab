"""Geometry of the hierarchical memory (Sana rwm/sparse-memory ``dev/rwm/diffusion/model/utils/hierarchical_memory.py``
and ``sana_qwennext_policy_causal.memory_layout``, checked against e8dde6b56), plus the deploy clock.

Rows are 25 Hz control steps counted from the episode's first paired row r0 (the deploy's first observation, tick 0);
a chunk is 32 rows and one latent frame (VAE stride 8 x video frame stride 4). A sample anchored at row t carries:

* the dense stream (read by the blocks configured ``dense``): the stride-4 rows t - 32K + 4i (i = 0..8K, rows before r0
  are the first frame) encoded as ONE causal sequence -> [L_0 (the first frame) | L_1..L_K (one per history chunk)];
* the sparse stream (read by the blocks configured ``sparse``): [r0, s_0..s_{m-1}] with the m rows sampled uniformly over
  [r0 + 1, t] (s_{m-1} = t; fewer than m candidates are padded with copies of r0 ahead of them) encoded as one causal
  sequence -> [A_0 (the first frame), A_1.. (one summary latent per 8 sampled rows)];
* the noisy chunk [obs (row t, encoded alone) | target] + 33 robot rows (state, 32 actions).

The training sequence is [dense latents | sparse latents | obs | target] + rows, each stream present only when a block
kind reads it; every GDN block computes on (its memory + the noisy chunk) as a chunk scan, every softmax block on the
same kind of subset as a prefix-causal window; the other tokens pass the block unchanged. Positions are row offsets on
the aligned clock (one latent frame = 32 rows): L_0 and A_0 at the first frame's row, L_j at the end of its chunk, a
sparse summary latent at the mean of its 8 sampled rows, obs at t, the target at t + 32.

The functions here are pure and Sana's names are kept so the live-parity tests can compare them one to one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

SPARSE_FRAMES = 16
CHUNK_ROWS = 32
FRAME_STRIDE = 4
FRAMES_PER_LATENT = CHUNK_ROWS // FRAME_STRIDE
LATENT_TOKENS = 160
ROBOT_ROWS = 33
OFFSET_SCALE = 8
ONE_CHUNK_FRAMES = 1 + CHUNK_ROWS
DENSE = "dense"
SPARSE = "sparse"
MEMORY_KINDS = (DENSE, SPARSE)
GDN = "gdn"
SOFTMAX = "softmax"
BLOCK_KINDS = (GDN, SOFTMAX)
MEMORY_KEY = "hierarchical_memory"
MEMORY_FIELDS = (GDN, SOFTMAX, "sparse_frames")

FORM_TRAIN = "train"
FORM_COMMIT = "commit"
FORM_GENERATE = "generate"
FORMS = (FORM_TRAIN, FORM_COMMIT, FORM_GENERATE)
# the adapter's commit-once forms (exact for every memory choice; see ``stream_layout``)
FORM_MEMORY = "memory"
FORM_NOISY = "noisy"
STREAM_FORMS = (FORM_MEMORY, FORM_NOISY)
# the one memory choice Sana defines the streaming commit / generate forms for
STREAMING_MEMORY = (DENSE, SPARSE)


def parse_memory(block) -> dict:
    """``{gdn, softmax, sparse_frames}`` of ``model.extra.chunk_causal_policy.hierarchical_memory`` (Sana
    ``parse_memory``): exactly the three keys, each block kind reading ``dense`` or ``sparse``, a positive multiple of
    8 sparse images (the causal VAE folds 8 frames into one latent)."""

    items = dict(block) if isinstance(block, Mapping) else dict(vars(block))
    if set(items) != set(MEMORY_FIELDS):
        raise ValueError(
            f"model.extra.chunk_causal_policy.{MEMORY_KEY} must declare exactly {list(MEMORY_FIELDS)}, got {sorted(items)}"
        )
    choices = {kind: str(items[kind]) for kind in BLOCK_KINDS}
    wrong = {kind: choice for kind, choice in choices.items() if choice not in MEMORY_KINDS}
    if wrong:
        raise ValueError(f"{MEMORY_KEY}: every block kind reads one of {MEMORY_KINDS}; got {wrong}")
    m = items["sparse_frames"]
    if isinstance(m, bool) or int(m) != m or int(m) <= 0 or int(m) % FRAMES_PER_LATENT:
        raise ValueError(
            f"{MEMORY_KEY}.sparse_frames must be a positive multiple of {FRAMES_PER_LATENT} (the causal VAE folds "
            f"{FRAMES_PER_LATENT} sampled frames into one latent), got {m!r}"
        )
    return {**choices, "sparse_frames": int(m)}


def memory_streams(memory: Mapping) -> tuple:
    """The memory kinds the two block kinds read, dense first."""

    return tuple(kind for kind in MEMORY_KINDS if kind in (memory[GDN], memory[SOFTMAX]))


def sparse_latents(m: int) -> int:
    """Latents of the sparse stream: the first frame, then one per 8 sampled rows."""

    return 1 + int(m) // FRAMES_PER_LATENT


def memory_k(t: int, r0: int) -> int:
    """K = ceil((t - r0) / 32): the history chunks before the anchor row t (the decision index at deploy)."""

    t, r0 = int(t), int(r0)
    if t < r0:
        raise ValueError(f"anchor row t={t} precedes the first paired row r0={r0}")
    return -(-(t - r0) // CHUNK_ROWS)


def target_rows(t: int) -> list[int]:
    """The 8 rows t + 4i (i = 1..8) of the target chunk."""

    return [int(t) + FRAME_STRIDE * i for i in range(1, FRAMES_PER_LATENT + 1)]


def dense_rows(t: int, r0: int) -> list[int]:
    """The dense stream's rows: the stride-4 history t - 32K + 4i (i = 0..8K, clamped to r0), then the 8 target rows."""

    k = memory_k(t, r0)
    start = int(t) - CHUNK_ROWS * k
    history = [max(int(r0), start + FRAME_STRIDE * i) for i in range(FRAMES_PER_LATENT * k + 1)]
    return history + target_rows(t)


def sparse_rows(t: int, r0: int, m: int = SPARSE_FRAMES) -> list[int]:
    """The sparse stream's rows: r0, then m rows sampled uniformly over [r0 + 1, t].

    n = t - r0 >= m: s_i = r0 + 1 + floor(i (n - 1) / (m - 1) + 1/2) in exact integer arithmetic (distinct, s_0 = r0 + 1,
    s_{m-1} = t); n < m: the m - n missing rows are copies of r0 ahead of r0 + 1..t.
    """

    t, r0, m = int(t), int(r0), int(m)
    if t < r0:
        raise ValueError(f"anchor row t={t} precedes the first paired row r0={r0}")
    if m <= 0 or m % FRAMES_PER_LATENT:
        raise ValueError(f"sparse_frames={m} must be a positive multiple of {FRAMES_PER_LATENT}")
    n = t - r0
    if n >= m:
        sampled = [r0 + 1 + (2 * i * (n - 1) + (m - 1)) // (2 * (m - 1)) for i in range(m)]
    else:
        sampled = [r0] * (m - n) + list(range(r0 + 1, t + 1))
    return [r0] + sampled


def stream_rows(t: int, r0: int, m: int, streams) -> dict:
    """The rows every stream of a TRAINING sample encodes, in encode order: the target rides the dense stream when there
    is one (its last latent), else it follows the sparse rows (Sana ``stream_rows``)."""

    rows = {}
    if DENSE in streams:
        rows[DENSE] = dense_rows(t, r0)
    if SPARSE in streams:
        rows[SPARSE] = sparse_rows(t, r0, m) + ([] if DENSE in streams else target_rows(t))
    return rows


def frame_layout(k: int, m: int, streams) -> dict:
    """The latent-frame layout [L_0..L_K (dense) | A_0.. (sparse) | obs | target] + rows of a training sample."""

    k = int(k)
    if k < 0:
        raise ValueError(f"k={k} must be >= 0")
    dense = list(range(k + 1)) if DENSE in streams else []
    first = len(dense)
    sparse = list(range(first, first + sparse_latents(m))) if SPARSE in streams else []
    obs = len(dense) + len(sparse)
    return {
        DENSE: dense,
        SPARSE: sparse,
        "obs": obs,
        "target": obs + 1,
        "frames": obs + 2,
        "tokens": LATENT_TOKENS * (obs + 2) + ROBOT_ROWS,
    }


def frame_offsets(t: int, r0: int, m: int, streams) -> list[float]:
    """The row offset from t of every latent frame in ``frame_layout`` order (the training clock)."""

    k = memory_k(t, r0)
    offsets = []
    if DENSE in streams:
        offsets += [float(r0 - t)] + [float(-CHUNK_ROWS * (k - j)) for j in range(1, k + 1)]
    if SPARSE in streams:
        sampled = sparse_rows(t, r0, m)[1:]
        offsets += [float(r0 - t)] + [
            sum(sampled[u : u + FRAMES_PER_LATENT]) / FRAMES_PER_LATENT - t
            for u in range(0, len(sampled), FRAMES_PER_LATENT)
        ]
    return offsets + [0.0, float(CHUNK_ROWS)]


def offsets_x8(offsets) -> list[int]:
    """Row offsets scaled by 8 as exact ints (every offset is a multiple of 1/8)."""

    scaled = [float(value) * OFFSET_SCALE for value in offsets]
    if any(value != round(value) for value in scaled):
        raise ValueError(f"row offsets {list(offsets)} are not multiples of 1/{OFFSET_SCALE}")
    return [int(round(value)) for value in scaled]


def block_frames(layout: Mapping, choice: str) -> list[int]:
    """The frames a block reading memory ``choice`` computes on: that memory's latents, then obs and target."""

    return list(layout[choice]) + [layout["obs"], layout["target"]]


def block_tokens(layout: Mapping, choice: str) -> torch.Tensor:
    """The full-sequence token positions of ``block_frames(layout, choice)`` in frame order, then the robot rows."""

    frames = torch.tensor(block_frames(layout, choice), dtype=torch.long)
    frame_tokens = (frames[:, None] * LATENT_TOKENS + torch.arange(LATENT_TOKENS)).reshape(-1)
    robot_tokens = LATENT_TOKENS * int(layout["frames"]) + torch.arange(ROBOT_ROWS)
    return torch.cat([frame_tokens, robot_tokens])


def block_bounds(layout: Mapping, choice: str) -> list[int]:
    """The chunk bounds inside a block's subset: one 160-token chunk per memory latent, then the 353-token noisy chunk."""

    memory = len(layout[choice])
    bounds = [LATENT_TOKENS * i for i in range(memory + 1)]
    return bounds + [bounds[-1] + 2 * LATENT_TOKENS + ROBOT_ROWS]


def memory_layout(form: str, k: int, memory: Mapping) -> dict:
    """The token layout of a forward of ``form`` (Sana ``memory_layout``).

    ``train``: [L_0..L_K | A_0.. | obs | target] + rows, each block kind on its memory + the noisy chunk.
    ``generate`` / ``commit`` exist for {gdn: dense, softmax: sparse} alone: generate is [A_0.. | obs | target] + rows
    with the GDN blocks on the noisy chunk only (they read the committed state) and the softmax blocks on all of it;
    commit is one dense latent without rows through the GDN blocks alone.

    Returns:
        ``{"frames": F, "rows": R, "subsets": {block kind: (token index LongTensor, chunk bounds) or None}}``.
    """

    m, streams = memory["sparse_frames"], memory_streams(memory)
    if form == FORM_TRAIN:
        layout = frame_layout(k, m, streams)
        subsets = {kind: (block_tokens(layout, memory[kind]), block_bounds(layout, memory[kind])) for kind in BLOCK_KINDS}
        return {"frames": layout["frames"], "rows": ROBOT_ROWS, "subsets": subsets}
    if form not in FORMS:
        raise ValueError(f"hierarchical-memory form={form!r} is not one of {FORMS}")
    if (memory[GDN], memory[SOFTMAX]) != STREAMING_MEMORY:
        raise ValueError(
            f"the streaming {form} form exists for {{gdn: dense, softmax: sparse}} alone; got gdn={memory[GDN]}, "
            f"softmax={memory[SOFTMAX]}"
        )
    anchor = frame_layout(0, m, streams)
    dense_index, dense_bounds = block_tokens(anchor, DENSE), block_bounds(anchor, DENSE)
    if form == FORM_COMMIT:
        return {"frames": 1, "rows": 0, "subsets": {GDN: (dense_index[:LATENT_TOKENS], dense_bounds[:2]), SOFTMAX: None}}
    shift = LATENT_TOKENS
    subsets = {
        GDN: (dense_index[shift:] - shift, [bound - shift for bound in dense_bounds[1:]]),
        SOFTMAX: (block_tokens(anchor, SPARSE) - shift, block_bounds(anchor, SPARSE)),
    }
    return {"frames": anchor["frames"] - 1, "rows": ROBOT_ROWS, "subsets": subsets}


def stream_layout(form: str, memory: Mapping, *, kind: str | None = None, latents: int | None = None) -> dict:
    """The token layout of the adapter's commit-once forms (not Sana forms; Sana's ``generate`` re-runs the sparse
    memory inside every denoising step).

    A memory token never reads the noisy chunk (the GDN scan reads it before the noisy chunk, the softmax window is
    prefix-causal), its t is 0, and without AttnRes time conditioning its depth mixing is its own: its hidden states,
    and so the GDN state it writes and the softmax K / V it leaves, are fixed once the memory before it is. So every
    memory latent can be committed ONCE through the blocks that read its memory:

    ``memory``: ``latents`` clean latents of memory ``kind`` (``dense`` / ``sparse``) without robot rows, one 160-token
    chunk each, through the block kinds that read ``kind`` (a GDN block continues that memory's scan, a softmax block
    its prefix-causal window); the other block kind is skipped.
    ``noisy``: the noisy chunk [obs | target] + rows alone, one 353-token chunk, in every block: a GDN block reads its
    memory's committed state, a softmax block attends to its memory's committed K / V and then to itself.
    """

    if form == FORM_MEMORY:
        if kind not in MEMORY_KINDS:
            raise ValueError(f"a memory commit names the memory it carries ({MEMORY_KINDS}); got {kind!r}")
        if latents is None or int(latents) < 1:
            raise ValueError(f"a memory commit carries >= 1 latents, got {latents!r}")
        readers = [block for block in BLOCK_KINDS if memory[block] == kind]
        if not readers:
            raise ValueError(f"no block kind reads the {kind} memory ({dict(memory)})")
        index = torch.arange(LATENT_TOKENS * int(latents))
        bounds = [LATENT_TOKENS * i for i in range(int(latents) + 1)]
        subsets = {block: ((index, bounds) if memory[block] == kind else None) for block in BLOCK_KINDS}
        return {"frames": int(latents), "rows": 0, "subsets": subsets}
    if form == FORM_NOISY:
        tokens = 2 * LATENT_TOKENS + ROBOT_ROWS
        index = torch.arange(tokens)
        return {"frames": 2, "rows": ROBOT_ROWS, "subsets": {block: (index, [0, tokens]) for block in BLOCK_KINDS}}
    raise ValueError(f"commit-once form={form!r} is not one of {STREAM_FORMS}")


# -- the deploy clock ---------------------------------------------------------------------------------------------------
#
# A streaming session keeps ONE clock across its forwards (Sana ``forward_chunk``: "decision c's observation at c * C when
# chunk_offset is c"): rows counted from r0. Every training offset is "row - t", so the episode clock is the training
# clock shifted by t - r0 = 32c for every frame of decision c, and a committed dense latent L_j sits at 32j for every
# later reader (RoPE only sees differences, so this is the training geometry up to floating-point round-off; Sana
# ``test_the_episode_clock_is_the_relative_clock_up_to_round_off``).


def decision_row(c: int, r0: int = 0) -> int:
    """The anchor row t of decision c (t = r0 + 32c)."""

    if int(c) < 0:
        raise ValueError(f"decision index c={c} must be >= 0")
    return int(r0) + CHUNK_ROWS * int(c)


def commit_rows(j: int) -> float:
    """Episode-clock row of the committed dense latent L_j (L_0 = the first frame at 0, L_j ends chunk j - 1 at 32j)."""

    if int(j) < 0:
        raise ValueError(f"dense latent index j={j} must be >= 0")
    return float(CHUNK_ROWS * int(j))


def generate_rows(c: int, m: int) -> list[float]:
    """Episode-clock rows of decision c's generate window [A_0, A_1.. | obs | target] (the sparse layout of
    ``frame_offsets`` shifted by t - r0 = 32c)."""

    t = decision_row(c)
    return [value + float(t) for value in frame_offsets(t, 0, m, (SPARSE,))]


def train_rows(c: int, m: int, streams) -> list[float]:
    """Training-clock rows (offsets from t) of decision c's full sequence: exactly ``frame_offsets`` at t = r0 + 32c."""

    return frame_offsets(decision_row(c), 0, m, streams)


def sparse_ticks(c: int, m: int) -> list[int]:
    """The ticks (rows - r0) of decision c's sparse stream [r0, s_0..s_{m-1}]."""

    return sparse_rows(decision_row(c), 0, m)


def dense_chunk_ticks(j: int) -> list[int]:
    """The 8 stride-4 ticks a dense chunk latent L_j (j >= 1) folds: 32(j-1)+4, .., 32j (training's D rows of chunk j-1)."""

    if int(j) < 1:
        raise ValueError(f"L_{j}: chunk latents start at j = 1 (L_0 is the first frame alone)")
    return [CHUNK_ROWS * (int(j) - 1) + FRAME_STRIDE * i for i in range(1, FRAMES_PER_LATENT + 1)]


def episode_clock(streams) -> bool:
    """A commit-once session keeps the episode clock when a dense memory persists across decisions (its committed latents
    keep their positions), else the training clock (offsets from t; every decision starts its memory afresh)."""

    return DENSE in streams


def sparse_memory_rows(c: int, m: int, episode: bool) -> list[float]:
    """Rows of decision c's sparse latents [A_0, A_1..] on the episode clock (from r0) or the training clock (from t)."""

    t = decision_row(c)
    rows = frame_offsets(t, 0, m, (SPARSE,))[:-2]
    return [value + float(t) for value in rows] if episode else rows


def noisy_rows(c: int, episode: bool) -> list[float]:
    """Rows of decision c's [obs, target]: 32c, 32c + 32 on the episode clock, 0, 32 on the training clock."""

    t = float(decision_row(c)) if episode else 0.0
    return [t, t + CHUNK_ROWS]


def noisy_step_offset(c: int, episode: bool) -> int:
    """The robot rows' first step on the same clock as ``noisy_rows`` (the state sits at the observation's instant)."""

    return decision_row(c) if episode else 0


def offsets_tensor(rows: Sequence[float]) -> torch.Tensor:
    """``[1, F]`` int64 eighths of ``rows`` (``data_info['hierarchical_memory']['frame_offsets_x8']``)."""

    return torch.tensor([offsets_x8(rows)], dtype=torch.int64)


__all__ = [
    "BLOCK_KINDS",
    "CHUNK_ROWS",
    "DENSE",
    "FORMS",
    "FORM_COMMIT",
    "FORM_GENERATE",
    "FORM_MEMORY",
    "FORM_NOISY",
    "FORM_TRAIN",
    "FRAMES_PER_LATENT",
    "FRAME_STRIDE",
    "GDN",
    "LATENT_TOKENS",
    "MEMORY_FIELDS",
    "MEMORY_KEY",
    "MEMORY_KINDS",
    "OFFSET_SCALE",
    "ROBOT_ROWS",
    "SOFTMAX",
    "SPARSE",
    "SPARSE_FRAMES",
    "STREAMING_MEMORY",
    "STREAM_FORMS",
    "block_bounds",
    "block_frames",
    "block_tokens",
    "offsets_tensor",
    "commit_rows",
    "decision_row",
    "dense_chunk_ticks",
    "dense_rows",
    "episode_clock",
    "frame_layout",
    "frame_offsets",
    "generate_rows",
    "memory_k",
    "memory_layout",
    "memory_streams",
    "noisy_rows",
    "noisy_step_offset",
    "offsets_x8",
    "parse_memory",
    "sparse_latents",
    "sparse_memory_rows",
    "sparse_rows",
    "sparse_ticks",
    "stream_layout",
    "stream_rows",
    "target_rows",
    "train_rows",
]
