"""The hierarchical memory's geometry: the plan's worked examples, the deploy clock, and every function against Sana's
``hierarchical_memory`` / ``memory_layout`` (live, skipped without the checkout)."""

from __future__ import annotations

import pytest
import torch

import _paths  # noqa: F401
import _live

from sana_wam_hm import memory as hm

SPLIT = {"gdn": "dense", "softmax": "sparse", "sparse_frames": 16}
SPARSE = {"gdn": "sparse", "softmax": "sparse", "sparse_frames": 16}
DENSE_DENSE = {"gdn": "dense", "softmax": "dense", "sparse_frames": 16}
SPARSE_DENSE = {"gdn": "sparse", "softmax": "dense", "sparse_frames": 16}
ALL_MEMORIES = [SPLIT, SPARSE, DENSE_DENSE, SPARSE_DENSE]


def test_the_plan_examples():
    r0 = 1
    # plan 2.2: n = 40 -> s = r0 + {1, 4, 6, 9, 11, 14, 17, 19, 22, 24, 27, 30, 32, 35, 37, 40}
    assert hm.sparse_rows(r0 + 40, r0) == [r0] + [r0 + d for d in (1, 4, 6, 9, 11, 14, 17, 19, 22, 24, 27, 30, 32, 35, 37, 40)]
    # n = 5 -> [r0 | r0 x11, r0+1 .. r0+5]
    assert hm.sparse_rows(r0 + 5, r0) == [r0] + [r0] * 11 + [r0 + 1, r0 + 2, r0 + 3, r0 + 4, r0 + 5]
    assert hm.sparse_rows(r0, r0) == [r0] * 17
    # plan 2.1 table: (t, K, rows of L_1)
    assert hm.memory_k(r0, r0) == 0 and hm.dense_rows(r0, r0) == [r0] + hm.target_rows(r0)
    assert hm.memory_k(r0 + 1, r0) == 1 and hm.dense_rows(r0 + 1, r0)[:9] == [r0] * 8 + [r0 + 1]
    assert hm.memory_k(r0 + 32, r0) == 1 and hm.dense_rows(r0 + 32, r0)[:9] == [r0 + 4 * i for i in range(9)]
    assert hm.memory_k(r0 + 33, r0) == 2
    # plan 2.5 positions at n = 40: A_0 -1.25 frames (r0 - t = -40 rows), A_1 -0.934 frames, A_2 -0.285 frames
    offsets = hm.frame_offsets(r0 + 40, r0, 16, (hm.SPARSE,))
    assert offsets[0] == -40.0 and offsets[1] == 10.125 - 40 and offsets[2] == 30.875 - 40 and offsets[3:] == [0.0, 32.0]


@pytest.mark.parametrize("c", [0, 1, 2, 5, 13])
def test_the_episode_clock_is_the_training_clock_shifted_by_t(c):
    """Decision c (t = 32c): every generate / commit row equals its training offset + (t - r0); the dense chunk ticks
    are the training D rows of each history chunk."""

    t = hm.decision_row(c)
    sparse_train = hm.frame_offsets(t, 0, 16, (hm.SPARSE,))
    assert hm.generate_rows(c, 16) == [v + t for v in sparse_train]
    dense_train = hm.frame_offsets(t, 0, 16, (hm.DENSE,))[: c + 1]      # L_0..L_c
    assert [hm.commit_rows(j) for j in range(c + 1)] == [v + t for v in dense_train]
    rows = hm.dense_rows(t, 0)
    assert rows[0] == 0
    for j in range(1, c + 1):
        assert hm.dense_chunk_ticks(j) == rows[1 + 8 * (j - 1) : 1 + 8 * j]
    assert hm.sparse_ticks(c, 16) == hm.sparse_rows(t, 0, 16)
    assert hm.train_rows(c, 16, (hm.DENSE, hm.SPARSE)) == hm.frame_offsets(t, 0, 16, (hm.DENSE, hm.SPARSE))


def test_every_offset_is_a_multiple_of_an_eighth():
    for n in range(0, 600):
        for streams in ((hm.SPARSE,), (hm.DENSE, hm.SPARSE)):
            hm.offsets_x8(hm.frame_offsets(1 + n, 1, 16, streams))
    with pytest.raises(ValueError, match="multiples of 1/8"):
        hm.offsets_x8([0.0625])


def test_the_streaming_layouts():
    commit = hm.memory_layout(hm.FORM_COMMIT, 3, SPLIT)
    assert (commit["frames"], commit["rows"]) == (1, 0) and commit["subsets"][hm.SOFTMAX] is None
    assert commit["subsets"][hm.GDN][0].tolist() == list(range(160)) and commit["subsets"][hm.GDN][1] == [0, 160]
    generate = hm.memory_layout(hm.FORM_GENERATE, 3, SPLIT)
    assert (generate["frames"], generate["rows"]) == (5, 33)
    gdn_index, gdn_bounds = generate["subsets"][hm.GDN]
    assert gdn_index.tolist() == list(range(480, 833)) and gdn_bounds == [0, 353]
    softmax_index, softmax_bounds = generate["subsets"][hm.SOFTMAX]
    assert softmax_index.tolist() == list(range(833)) and softmax_bounds == [0, 160, 320, 480, 833]
    for memory in (SPARSE, DENSE_DENSE, SPARSE_DENSE):
        for form in (hm.FORM_COMMIT, hm.FORM_GENERATE):
            with pytest.raises(ValueError, match="streaming"):
                hm.memory_layout(form, 0, memory)


def test_parse_memory_refuses_what_sana_refuses():
    assert hm.parse_memory(SPLIT) == SPLIT
    for bad, match in (
        ({"gdn": "dense", "softmax": "sparse"}, "exactly"),
        ({**SPLIT, "extra": 1}, "exactly"),
        ({**SPLIT, "gdn": "uniform"}, "reads one of"),
        ({**SPLIT, "sparse_frames": 12}, "multiple of 8"),
        ({**SPLIT, "sparse_frames": True}, "multiple of 8"),
    ):
        with pytest.raises(ValueError, match=match):
            hm.parse_memory(bad)


# -- live -----------------------------------------------------------------------------------------------------------------


def test_the_row_functions_are_sanas():
    live = _live.live_hm()
    for r0 in (0, 1, 7):
        for n in list(range(0, 80)) + [100, 333, 1000, 1503, 3000]:
            t = r0 + n
            assert hm.memory_k(t, r0) == live.memory_k(t, r0)
            assert hm.target_rows(t) == live.target_rows(t)
            assert hm.dense_rows(t, r0) == live.dense_rows(t, r0)
            for m in (8, 16, 24):
                assert hm.sparse_rows(t, r0, m) == live.sparse_rows(t, r0, m)
            for memory in ALL_MEMORIES:
                streams = hm.memory_streams(memory)
                assert streams == live.memory_streams(memory)
                assert hm.stream_rows(t, r0, 16, streams) == live.stream_rows(t, r0, 16, streams)
                assert hm.frame_offsets(t, r0, 16, streams) == live.frame_offsets(t, r0, 16, streams)
                assert hm.offsets_x8(hm.frame_offsets(t, r0, 16, streams)) == live.offsets_x8(live.frame_offsets(t, r0, 16, streams))


def test_the_layouts_are_sanas():
    _live.live_hm()
    net = _live.live_causal()
    for memory in ALL_MEMORIES:
        streams = hm.memory_streams(memory)
        for k in (0, 1, 3, 47):
            ours, theirs = hm.frame_layout(k, 16, streams), _live.live_hm().frame_layout(k, 16, streams)
            assert ours == theirs
            forms = [hm.FORM_TRAIN] + ([hm.FORM_COMMIT, hm.FORM_GENERATE] if memory == SPLIT else [])
            for form in forms:
                a, b = hm.memory_layout(form, k, memory), net.memory_layout(form, k, memory)
                assert (a["frames"], a["rows"]) == (b["frames"], b["rows"])
                for kind in hm.BLOCK_KINDS:
                    if b["subsets"][kind] is None:
                        assert a["subsets"][kind] is None
                        continue
                    assert torch.equal(a["subsets"][kind][0], b["subsets"][kind][0])
                    assert list(a["subsets"][kind][1]) == list(b["subsets"][kind][1])
