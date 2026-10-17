"""The sparse memory's geometry: worked examples, the forwards' layouts, the deploy clock, and every row function against
Sana's ``sparse_memory`` (live, skipped without the checkout) and the episode's prefix rows against Sana's
``SparseMemoryEpisode``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import _paths  # noqa: F401
import _live

from sana_wam_sm import sparse_memory as sm

GLOBAL = {"memory_type": "global", "memory_frames": 16}
STRIDED = {"memory_type": "strided", "memory_frames": 16, "memory_stride": 16}
STRIDED_M8S25 = {"memory_type": "strided", "memory_frames": 8, "memory_stride": 25}
MEMORIES = {"global": GLOBAL, "strided_m16s16": STRIDED, "strided_m8s25": STRIDED_M8S25}
# more memories than the recipes train: strides around the latent size, other multiples of 8
SWEEP = [GLOBAL, STRIDED, STRIDED_M8S25, {"memory_type": "strided", "memory_frames": 8, "memory_stride": 1},
         {"memory_type": "strided", "memory_frames": 24, "memory_stride": 32},
         {"memory_type": "global", "memory_frames": 8}, {"memory_type": "global", "memory_frames": 24}]


def _sweep_id(memory):
    return f"{memory['memory_type']}_m{memory['memory_frames']}" + (f"s{memory['memory_stride']}" if "memory_stride" in memory else "")


def test_the_worked_examples():
    # strided m = 8, s = 25 from r_0 = 0: the newest 8 rows every 25 back from t, r_0 copies ahead of a short span
    for obs_row, rows in [
        (0, [0] * 8),
        (1, [0] * 7 + [1]),
        (100, [0, 0, 0, 0, 25, 50, 75, 100]),
        (175, [0, 25, 50, 75, 100, 125, 150, 175]),
        (500, list(range(325, 501, 25))),
    ]:
        assert sm.memory_rows(obs_row, 0, STRIDED_M8S25) == rows
        assert sm.sequence_rows(obs_row, 0, STRIDED_M8S25) == [0] + rows + list(range(obs_row + 4, obs_row + 33, 4))
    # global m = 16: the hierarchical memory's noobs rows (padding while the span is short, then uniform)
    assert sm.global_rows(1, 1, 16) == [1] * 16
    assert sm.global_rows(12, 1, 16) == [1] * 5 + list(range(2, 13))
    assert sm.global_rows(18, 1, 16) == [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
    assert sm.global_rows(97, 1, 16) == [2, 8, 15, 21, 27, 34, 40, 46, 53, 59, 65, 72, 78, 84, 91, 97]
    # n = m / 8 at every row
    assert [sm.memory_latents(m) for m in (GLOBAL, STRIDED, STRIDED_M8S25)] == [2, 2, 1]
    # the tail freeze of the target rows
    assert sm.target_rows(90, last_row=100) == [94, 98, 100, 100, 100, 100, 100, 100]


def test_the_layouts_of_the_three_forwards():
    for n in (0, 1, 3):
        sequence = sm.form_layout(sm.FORM_SEQUENCE, n + 2)
        assert sequence["phi"] == list(range(n + 2)) and sequence["step_offset"] == 32 * n
        assert sequence["bounds"] == [160 * i for i in range(n + 2)] + [160 * (n + 2) + 33]
        assert sequence["rows"] == 33 and sequence["memory_latents"] == n
        prefix = sm.form_layout(sm.FORM_PREFIX, n + 1)
        assert prefix["phi"] == list(range(n + 1)) and prefix["rows"] == 0
        assert prefix["bounds"] == [160 * i for i in range(n + 2)]
        target = sm.form_layout(sm.FORM_TARGET, 1, n)
        assert target["phi"] == [n + 1] and target["step_offset"] == 32 * n and target["bounds"] == [0, 193]
        # the prefix + target split the sequence: the same RoPE frames, the same robot clock, the target chunk last
        assert prefix["phi"] + target["phi"] == sequence["phi"]
        assert prefix["bounds"] + [prefix["bounds"][-1] + b for b in target["bounds"][1:]] == sequence["bounds"]
    # the sana_latent strip: three 8 x 10 views per latent, 240 tokens per chunk of the sink and the memory
    strip = sm.form_layout(sm.FORM_SEQUENCE, 3, frame_tokens=240)
    assert strip["bounds"] == [0, 240, 480, 753] and sm.prefix_bounds(2, 240) == [0, 240, 480]
    assert sm.form_layout(sm.FORM_TARGET, 1, 1, frame_tokens=240)["bounds"] == [0, 273]
    with pytest.raises(ValueError, match="memory latent count"):
        sm.form_layout(sm.FORM_TARGET, 1)
    with pytest.raises(ValueError, match="the target latent alone"):
        sm.form_layout(sm.FORM_TARGET, 2, 0)
    with pytest.raises(ValueError, match="sink and the target"):
        sm.form_layout(sm.FORM_SEQUENCE, 1)


@pytest.mark.parametrize("memory", MEMORIES.values(), ids=list(MEMORIES))
def test_the_deploy_clock_reads_the_current_frame_through_the_memory(memory):
    for c in range(60):
        t = sm.decision_row(c)
        assert t == 32 * c
        rows = sm.prefix_rows(t, 0, memory)
        assert rows[0] == 0 and len(rows) == 1 + 8 * sm.memory_latents(memory)
        assert rows == sm.sequence_rows(t, 0, memory)[: -sm.FRAMES_PER_LATENT]
        assert max(rows) == t and all(a <= b for a, b in zip(rows, rows[1:]))
        assert sm.target_rows(t)[-1] == sm.decision_row(c + 1)       # the target's last frame is the next decision's row


def test_memory_from_policy_and_its_refusals():
    assert sm.memory_from_policy({"memory_type": "global", "memory_frames": 16}) == GLOBAL
    # the global memory's name before Sana eef7beeab (the same rows) reads as global
    assert sm.memory_from_policy({"memory_type": "fixed", "memory_frames": 16}) == GLOBAL
    assert sm.memory_from_policy(SimpleNamespace(memory_type="strided", memory_frames=8, memory_stride="25")) == STRIDED_M8S25
    assert sm.memory_from_policy({"actions_per_chunk": {"25": 32}}) is None and sm.memory_from_policy(None) is None
    for policy, match in (
        ({"memory_type": "uniform", "memory_frames": 16}, "global or strided"),
        ({"memory_type": "global", "memory_frames": 12}, "multiple of 8"),
        ({"memory_type": "fixed", "memory_frames": 0}, "multiple of 8"),
        # the unbounded strided memory before eef7beeab declares no memory_frames
        ({"memory_type": "strided", "memory_stride": 25}, "unbounded strided memory"),
        ({"memory_type": "strided", "memory_frames": 16}, "positive chunk_causal_policy.memory_stride"),
        ({"memory_type": "strided", "memory_frames": 16, "memory_stride": 0}, "positive chunk_causal_policy.memory_stride"),
        ({"memory_type": "global", "memory_frames": 16, "memory_stride": 25}, "belongs to a strided memory"),
        ({"memory_stride": 25}, "without memory_type"),
        ({"memory_frames": 16}, "without memory_type"),
    ):
        with pytest.raises(ValueError, match=match):
            sm.memory_from_policy(policy)
    assert not sm.full_sequence_attention({}) and not sm.full_sequence_attention({"sequence_attention": "causal"})
    assert sm.full_sequence_attention({"sequence_attention": "full"})
    with pytest.raises(ValueError, match="causal or full"):
        sm.full_sequence_attention({"sequence_attention": "dense"})


# -- live ------------------------------------------------------------------------------------------------------------


def test_the_constants_are_sanas():
    live = _live.live_sparse_memory()
    for name in ("CHUNK_ROWS", "FRAME_STRIDE", "FRAMES_PER_LATENT", "ROBOT_ROWS"):
        assert getattr(sm, name) == getattr(live, name), name


@pytest.mark.parametrize("first_row", [0, 1, 7])
@pytest.mark.parametrize("memory", SWEEP, ids=_sweep_id)
def test_the_rows_are_sanas(memory, first_row):
    live = _live.live_sparse_memory()
    for obs_row in range(first_row, first_row + 1601, 3):
        assert sm.memory_rows(obs_row, first_row, memory) == live.memory_rows(obs_row, first_row, memory)
        assert sm.sequence_rows(obs_row, first_row, memory) == live.sequence_rows(obs_row, first_row, memory)
        last_row = obs_row + 17
        assert sm.sequence_rows(obs_row, first_row, memory, last_row) == live.sequence_rows(obs_row, first_row, memory, last_row)
    n = sm.memory_latents(memory)
    assert n == live.memory_latents(memory) and sm.sampler_rung(n) == live.sampler_rung(n)
    for frame_tokens in (sm.TOKENS_PER_FRAME, 240):
        layout = sm.form_layout(sm.FORM_SEQUENCE, n + 2, frame_tokens=frame_tokens)
        assert layout["bounds"] == live.chunk_bounds(n + 2, frame_tokens, sm.ROBOT_ROWS)
        assert sm.prefix_bounds(n + 1, frame_tokens) == live.chunk_bounds(n + 1, frame_tokens, 0)
    m = memory["memory_frames"]
    if memory["memory_type"] == "strided":
        s = memory["memory_stride"]
        assert all(sm.strided_rows(t, first_row, m, s) == live.strided_rows(t, first_row, m, s) for t in range(first_row, 2000))
    else:
        assert all(sm.global_rows(t, first_row, m) == live.global_rows(t, first_row, m) for t in range(first_row, 2000))


def test_memory_from_policy_is_sanas():
    live = _live.live_sparse_memory()
    for policy in (GLOBAL, STRIDED, STRIDED_M8S25, SimpleNamespace(memory_type="strided", memory_frames=8, memory_stride="25"),
                   {"actions_per_chunk": 32}, None):
        assert sm.memory_from_policy(policy) == live.memory_from_policy(policy)
    for policy in ({"memory_type": "uniform"}, {"memory_type": "global", "memory_frames": 12}, {"memory_stride": 25},
                   {"memory_type": "strided", "memory_frames": 16}, {"memory_type": "global", "memory_frames": 16, "memory_stride": 4}):
        with pytest.raises(ValueError) as ours:
            sm.memory_from_policy(policy)
        with pytest.raises(ValueError) as theirs:
            live.memory_from_policy(policy)
        assert str(ours.value) == str(theirs.value)
    # where the adapter departs from Sana on purpose: the pre-eef7beeab name fixed, which Sana now refuses
    with pytest.raises(ValueError, match="global or strided"):
        live.memory_from_policy({"memory_type": "fixed", "memory_frames": 16})
    assert sm.memory_from_policy({"memory_type": "fixed", "memory_frames": 16}) == live.memory_from_policy(GLOBAL)
    for policy in ({}, {"sequence_attention": "causal"}, {"sequence_attention": "full"}):
        assert sm.full_sequence_attention(policy) == live.full_sequence_attention(policy)


def test_the_pixel_path_and_the_timesteps_are_sanas():
    live = _live.live_sparse_memory()
    g = torch.Generator().manual_seed(0)
    frames = torch.randint(0, 256, (2, 5, 3, 4, 6), generator=g, dtype=torch.uint8)
    for dtype in (torch.float32, torch.bfloat16):
        assert torch.equal(sm.pixels_from_uint8(frames, dtype), live.pixels_from_uint8(frames, dtype))
    t = torch.tensor([700.0, 30.0])
    assert torch.equal(sm.video_timesteps(t, 4), live.video_timesteps(t, 4))


@pytest.mark.parametrize("memory", MEMORIES.values(), ids=list(MEMORIES))
@pytest.mark.parametrize("first_row", [0, 1])
def test_the_prefix_rows_are_the_live_episodes(memory, first_row):
    session = _live.live_session()
    for obs_row in range(first_row, first_row + 430, 7):
        episode = session.SparseMemoryEpisode(first_row, torch.zeros(1), memory)
        assert sm.prefix_rows(obs_row, first_row, memory) == episode.prefix_rows(obs_row)
