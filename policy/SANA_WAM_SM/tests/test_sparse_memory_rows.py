"""The sparse memory's geometry: the plan's worked examples, the forwards' layouts, the deploy clock, and every row
function against Sana's ``sparse_memory`` (live, skipped without the checkout) and the episode's prefix rows against
Sana's ``SparseMemoryEpisode``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import _paths  # noqa: F401
import _live

from sana_wam_sm import sparse_memory as sm

STRIDED = {"memory_type": "strided", "memory_stride": 25}
FIXED = {"memory_type": "fixed", "memory_frames": 16}
MEMORIES = {"strided": STRIDED, "fixed": FIXED}
# more memories than the recipes train: strides around the latent size, other multiples of 8
SWEEP = [STRIDED, FIXED, {"memory_type": "strided", "memory_stride": 1}, {"memory_type": "strided", "memory_stride": 32},
         {"memory_type": "fixed", "memory_frames": 8}, {"memory_type": "fixed", "memory_frames": 24}]


def test_the_plan_examples():
    # strided s = 25 from r_0 = 0: (t, the strided rows, the r_0 padding, n)
    for obs_row, strided, pad, latents in [
        (0, [], 0, 0),
        (1, [1], 7, 1),
        (60, [10, 35, 60], 5, 1),
        (200, list(range(25, 201, 25)), 0, 1),
        (201, list(range(1, 202, 25)), 7, 2),
    ]:
        assert sm.memory_rows(obs_row, 0, STRIDED) == [0] * pad + strided
        assert sm.memory_latents(obs_row, 0, STRIDED) == latents
        assert sm.sequence_rows(obs_row, 0, STRIDED) == [0] + [0] * pad + strided + list(range(obs_row + 4, obs_row + 33, 4))
    # fixed m = 16: the hierarchical memory's noobs rows (padding while the span is short, then uniform)
    assert sm.fixed_rows(1, 1, 16) == [1] * 16
    assert sm.fixed_rows(12, 1, 16) == [1] * 5 + list(range(2, 13))
    assert sm.fixed_rows(18, 1, 16) == [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
    assert sm.fixed_rows(97, 1, 16) == [2, 8, 15, 21, 27, 34, 40, 46, 53, 59, 65, 72, 78, 84, 91, 97]
    assert all(sm.memory_latents(t, 0, FIXED) == 2 for t in (0, 1, 17, 500, 1504))
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
        assert rows[0] == 0 and len(rows) == 1 + 8 * sm.memory_latents(t, 0, memory)
        assert rows == sm.sequence_rows(t, 0, memory)[: -sm.FRAMES_PER_LATENT]
        assert max(rows) == t and all(a <= b for a, b in zip(rows, rows[1:]))
        assert sm.target_rows(t)[-1] == sm.decision_row(c + 1)       # the target's last frame is the next decision's row


def test_memory_from_policy_and_its_refusals():
    assert sm.memory_from_policy({"memory_type": "fixed", "memory_frames": 16}) == FIXED
    assert sm.memory_from_policy(SimpleNamespace(memory_type="strided", memory_stride="25")) == STRIDED
    assert sm.memory_from_policy({"actions_per_chunk": {"25": 32}}) is None and sm.memory_from_policy(None) is None
    for policy in ({"memory_type": "uniform", "memory_frames": 16}, {"memory_type": "fixed", "memory_frames": 12},
                   {"memory_type": "fixed", "memory_frames": 0}, {"memory_stride": 25}, {"memory_frames": 16}):
        with pytest.raises(ValueError):
            sm.memory_from_policy(policy)


# -- live ------------------------------------------------------------------------------------------------------------


def test_the_constants_are_sanas():
    live = _live.live_sparse_memory()
    for name in ("CHUNK_ROWS", "FRAME_STRIDE", "FRAMES_PER_LATENT", "TOKENS_PER_FRAME", "ROBOT_ROWS"):
        assert getattr(sm, name) == getattr(live, name), name


@pytest.mark.parametrize("first_row", [0, 1, 7])
@pytest.mark.parametrize("memory", SWEEP, ids=lambda m: f"{m['memory_type']}_{m.get('memory_frames', m.get('memory_stride'))}")
def test_the_rows_are_sanas(memory, first_row):
    live = _live.live_sparse_memory()
    for obs_row in range(first_row, first_row + 1601, 3):
        assert sm.memory_rows(obs_row, first_row, memory) == live.memory_rows(obs_row, first_row, memory)
        assert sm.memory_latents(obs_row, first_row, memory) == live.memory_latents(obs_row, first_row, memory)
        assert sm.sequence_rows(obs_row, first_row, memory) == live.sequence_rows(obs_row, first_row, memory)
        last_row = obs_row + 17
        assert sm.sequence_rows(obs_row, first_row, memory, last_row) == live.sequence_rows(obs_row, first_row, memory, last_row)
        n = sm.memory_latents(obs_row, first_row, memory)
        assert sm.sampler_rung(n) == live.sampler_rung(n)
        assert sm.form_layout(sm.FORM_SEQUENCE, n + 2)["bounds"] == live.chunk_bounds(n + 2)
    if memory["memory_type"] == "strided":
        s = memory["memory_stride"]
        assert all(sm.strided_latents(t, first_row, s) == live.strided_latents(t, first_row, s) for t in range(first_row, 2000))
    else:
        m = memory["memory_frames"]
        assert all(sm.fixed_rows(t, first_row, m) == live.fixed_rows(t, first_row, m) for t in range(first_row, 2000))


def test_memory_from_policy_is_sanas():
    live = _live.live_sparse_memory()
    for policy in (FIXED, STRIDED, {"memory_type": "fixed", "memory_frames": 16, "memory_stride": 25},
                   SimpleNamespace(memory_type="strided", memory_stride="25"), {"actions_per_chunk": 32}, None):
        assert sm.memory_from_policy(policy) == live.memory_from_policy(policy)
    for policy in ({"memory_type": "uniform"}, {"memory_type": "fixed", "memory_frames": 12}, {"memory_stride": 25}):
        with pytest.raises(ValueError) as ours:
            sm.memory_from_policy(policy)
        with pytest.raises(ValueError) as theirs:
            live.memory_from_policy(policy)
        assert str(ours.value) == str(theirs.value)


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
