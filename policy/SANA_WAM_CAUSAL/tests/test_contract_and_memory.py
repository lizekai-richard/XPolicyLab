"""Contract resolution from a causal training yaml and the session memory's softmax read rule, episode protocol (CPU,
no Sana)."""

from __future__ import annotations

import copy
import math
import os
from types import SimpleNamespace

import pytest
import torch
import yaml

import _paths

from sana_wam_causal.contract import (
    ALIGNED_PHASE_FIRST_FRAME,
    ALIGNED_PHASE_PADDED,
    MEMORY_VIDEO_AND_ACTION,
    MEMORY_VIDEO_ONLY,
    CausalContract,
    bidirectional_view,
    resolve_causal_contract,
)
from sana_wam_causal.layers import LayerCache
from sana_wam_causal.model import CausalPolicyModel
from sana_wam_causal.session import Caption, CausalMemory, CausalPolicySession
from sana_wam_min.config import THREE_VIEW_POLICY_FACTORY, policy_config_from_train_config
from sana_wam_min.policy_model.config import PolicyConfig

EPISODE_VA = os.path.join(_paths.FIXTURES, "config_causal_pantheon33k_episode_n24_videoandaction_0081aa702.yaml")
EPISODE_NOOBS = os.path.join(_paths.FIXTURES, "config_causal_pantheon33k_episode_n24_videoonly_noobs_0081aa702.yaml")
# the episode recipes of rwm/zekai-merge 7de7711c2 (the single form): the N24 no-observation recipe, which dropped the
# key, the N1 recipe (softmax window 1), the sana_pixel N2 recipe (require_video_target) and the N24 video-only recipe
# trained with chunk observations, which declares chunk_observation true and is refused
RECIPE_NOOBS = os.path.join(_paths.FIXTURES, "recipe_causal_pantheon33k_sana_latent_episode_n24_nsc_videoonly_noobs_7de7711c2.yaml")
RECIPE_N1 = os.path.join(_paths.FIXTURES, "recipe_causal_pantheon33k_sana_latent_episode_n1_nsc_videoonly_7de7711c2.yaml")
RECIPE_PIXEL_N2 = os.path.join(_paths.FIXTURES, "recipe_causal_pantheon14k_sana_pixel_episode_n2_nsc_videoonly_7de7711c2.yaml")
RECIPE_OBSERVATION = os.path.join(_paths.FIXTURES, "recipe_causal_pantheon33k_sana_latent_episode_n24_nsc_videoonly_7de7711c2.yaml")
# the released no-observation checkpoint, trained on the first-frame window (data.extra.aligned_phase_window first_frame)
FIRST_FRAME_NOOBS = os.path.join(_paths.FIXTURES, "config_causal_pantheon33k_episode_n24_videoonly_noobs_first_frame_s3324.yaml")
SINKLESS_VA = os.path.join(_paths.FIXTURES, "config_causal_pantheon14k_m48n24_videoandaction_s5000.yaml")
SINKLESS_VO = os.path.join(_paths.FIXTURES, "config_causal_pantheon14k_m48n24_videoonly_s5000.yaml")
RETIRED_CHUNK0 = os.path.join(_paths.FIXTURES, "config_causal_vanilla52k_m48n24_s7646.yaml")


def _load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def causal_cfg():
    return _load(EPISODE_NOOBS)


@pytest.mark.parametrize(
    "path,memory_content,window,phase",
    [(EPISODE_NOOBS, MEMORY_VIDEO_ONLY, 24, ALIGNED_PHASE_PADDED),
     (RECIPE_NOOBS, MEMORY_VIDEO_ONLY, 24, ALIGNED_PHASE_FIRST_FRAME),
     (RECIPE_N1, MEMORY_VIDEO_ONLY, 1, ALIGNED_PHASE_FIRST_FRAME),
     (EPISODE_VA, MEMORY_VIDEO_AND_ACTION, 24, ALIGNED_PHASE_PADDED)],
    ids=["noobs-0081aa702", "noobs-7de7711c2", "n1-7de7711c2", "no-key-0081aa702"],
)
def test_the_pantheon33k_episode_recipes_resolve(path, memory_content, window, phase):
    """The single form: every recipe without chunk observations resolves. A config without the key reads as without
    them (Sana refuse_chunk_observation) -- also the 0081aa702 video_and_action config, which in fact trained with
    chunk observations: Sana's own ambiguity for a config from before 2026-10-14 without the key."""

    contract = resolve_causal_contract(_load(path))
    assert (contract.actions_per_chunk, contract.video_frame_stride, contract.latent_frames_per_chunk) == (32, 4, 1)
    assert contract.sliding_window_chunks == window and contract.memory_content == memory_content
    assert (contract.multiview, contract.visual_layout, contract.rope) == ("sana_latent", "three_view_strip", "aligned")
    assert (contract.dt_bias_init, contract.a_log_init) == (-5.0, 0.0)
    text = contract.describe()
    assert text.startswith("protocol=episode ") and f"memory_content={memory_content}" in text
    assert " chunk_observation=False " in text
    assert contract.aligned_phase_window == phase and contract.first_chunk == (0 if phase == ALIGNED_PHASE_FIRST_FRAME else 1)
    assert f"aligned_phase_window={phase} first_chunk={contract.first_chunk}" in text


@pytest.mark.parametrize("value", [True, "false", 1])
def test_chunk_observations_are_retired(value):
    """Sana 4bd6eca83: only false or no key; the recipes trained with chunk observations declare true and are refused,
    as is anything that is not the bool false."""

    if value is True:
        cfg = _load(RECIPE_OBSERVATION)
        assert _policy(cfg)["chunk_observation"] is True
    else:
        cfg = copy.deepcopy(_load(RECIPE_NOOBS))
        _policy(cfg)["chunk_observation"] = value
    with pytest.raises(ValueError, match="chunk observations are retired"):
        resolve_causal_contract(cfg)


def test_the_first_frame_checkpoint_emits_chunk_0():
    """The released no-observation checkpoint trains the first-frame window (Sana 04c8acbee): its episodes open on
    chunk 0, which reads the sink."""

    contract = resolve_causal_contract(_load(FIRST_FRAME_NOOBS))
    assert (contract.aligned_phase_window, contract.first_chunk) == (ALIGNED_PHASE_FIRST_FRAME, 0)
    assert contract.memory_content == MEMORY_VIDEO_ONLY
    assert (contract.sliding_window_chunks, contract.multiview, contract.visual_layout) == (24, "sana_latent", "three_view_strip")
    assert "aligned_phase_window=first_frame first_chunk=0" in contract.describe()


def test_a_sana_pixel_canvas_resolves_on_the_episode_protocol():
    """The Pantheon14k sana_pixel N2 recipe (7de7711c2): the one-view canvas, a two-chunk softmax window, the
    first-frame window; data.extra.require_video_target is the data layer's and changes nothing at deploy."""

    cfg = _load(RECIPE_PIXEL_N2)
    assert cfg["data"]["extra"]["require_video_target"] is True
    contract = resolve_causal_contract(cfg)
    assert (contract.multiview, contract.visual_layout) == ("sana_pixel", "sana_pixel_canvas")
    assert (contract.sliding_window_chunks, contract.memory_content, contract.first_chunk) == (2, MEMORY_VIDEO_ONLY, 0)


def test_the_bidirectional_view_restores_the_sft_tier(causal_cfg):
    view = bidirectional_view(causal_cfg)
    assert view["model"]["model"] == THREE_VIEW_POLICY_FACTORY
    assert view["data"]["num_frames"] == 33 and view["data"]["multi_fps"] == {"25": [33]}
    assert causal_cfg["data"]["num_frames"] == 1569  # the source is left untouched
    config = policy_config_from_train_config(view)
    assert (config.depth, config.hidden_size, tuple(config.softmax_layer_indices)) == (32, 2560, (3, 7, 11, 15, 19, 23, 27, 31))
    assert config.rope == "aligned" and config.separate_action_schedule
    assert tuple(config.multiview_spatial_rope_tile_shape) == (8, 16)


@pytest.mark.parametrize("path", [SINKLESS_VA, SINKLESS_VO], ids=["video_and_action", "video_only"])
def test_the_sinkless_pantheon14k_checkpoints_are_refused(path):
    """The real config.yaml of the pantheon14k m48n24 step-5000 checkpoints (the sink-less per-chunk observation
    protocol) is refused by its marker, before any weight is read."""

    with pytest.raises(ValueError, match="window_repeats: this checkpoint trained the sink-less per-chunk observation"):
        resolve_causal_contract(_load(path))


def test_the_retired_chunk0_observation_checkpoint_is_refused():
    """The real config.yaml of the vanilla52k causal checkpoint s7646 (chunk-0-only observation) is refused."""

    with pytest.raises(ValueError, match="window_repeats|obs_in_first_chunk"):
        resolve_causal_contract(_load(RETIRED_CHUNK0))


def _policy(cfg):
    return cfg["model"]["extra"]["chunk_causal_policy"]


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda c: c["model"].__setitem__("model", THREE_VIEW_POLICY_FACTORY), "not the chunk-causal policy class"),
        (lambda c: c["data"]["extra"].__setitem__("window_repeats", [[1, 1], [35, 3]]), "sink-less"),
        (lambda c: _policy(c).__setitem__("obs_in_first_chunk", False), "retired observation layout"),
        (lambda c: _policy(c).__setitem__("memory_content", "video_and_state"), "memory_content"),
        (lambda c: _policy(c).__setitem__("chunk_observation", True), "chunk observations are retired"),
        (lambda c: _policy(c).__setitem__("memory_type", "strided"), "serve it with SANA_WAM_SM"),
        (lambda c: _policy(c).__setitem__("memory_frames", 16), "serve it with SANA_WAM_SM"),
        (lambda c: _policy(c).__setitem__("hierarchical_memory", {"gdn": "dense"}), "SANA_WAM_HM up to XPolicyLab d57a617"),
        (lambda c: _policy(c).__setitem__("memory_uniform", {"sparse_frames": 16}), "SANA_WAM_HM up to XPolicyLab d57a617"),
        (lambda c: _policy(c).__setitem__("memory_split", {"sparse_frames": 16}), "SANA_WAM_HM up to XPolicyLab d57a617"),
        (lambda c: c["model"]["extra"].__setitem__("rope", "independent"), "aligned only"),
        (lambda c: c["model"].__setitem__("use_time_conditioning", True), "use_time_conditioning"),
        (lambda c: (c["data"]["extra"].__setitem__("multiview", "openwam"),
                    c["data"].__setitem__("aspect_ratio_type", "ASPECT_RATIO_OPENWAM_LSHAPE_384_320")),
         "sana_latent strip and the one-view"),
        (lambda c: c["data"].__setitem__("type", "RoboDojoSFTDataset"), "causal chunk-window dataset"),
        (lambda c: c["data"]["extra"].__setitem__("aligned_phase_window", "h0"), "aligned_phase_window"),
        (lambda c: _policy(c).__setitem__("actions_per_chunk", {"25": 24}), "disagree"),
    ],
    ids=["sft-class", "window-repeats", "obs-key", "memory-content", "chunk-observation", "sparse-memory-strided",
         "sparse-memory-fixed", "hierarchical", "memory-uniform", "memory-split", "independent", "time-conditioning", "openwam", "sft-dataset",
         "phase-window", "chunk-mismatch"],
)
def test_unsupported_contracts_are_refused(causal_cfg, edit, match):
    cfg = copy.deepcopy(causal_cfg)
    edit(cfg)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        resolve_causal_contract(cfg)


def test_a_masked_sana_pixel_pad_is_refused():
    cfg = copy.deepcopy(_load(RECIPE_PIXEL_N2))
    cfg["model"]["extra"]["sana_pixel_pad"] = "masked"
    with pytest.raises(NotImplementedError, match="sana_pixel_pad"):
        resolve_causal_contract(cfg)


# -- the memory --------------------------------------------------------------------------------------------------------


class _Cfg(SimpleNamespace):
    def __getattr__(self, name):
        return None


class _Model:
    def __init__(self, kinds):
        self.blocks = [type("B", (), {"attn_type": kind})() for kind in kinds]


def _entry_caches(memory: CausalMemory, value: float) -> list[LayerCache]:
    caches = []
    for kind in memory.kinds:
        cache = LayerCache(commit=True)
        if kind == "GatedDeltaNet":
            cache.gdn_next = (torch.full((1,), value), torch.full((1,), value))
        else:
            cache.softmax_new = (torch.full((1, 1, 1, 2), value, dtype=torch.float64),
                                 torch.full((1, 1, 1, 2), value, dtype=torch.float64))
        caches.append(cache)
    return caches


def _sink(memory: CausalMemory) -> None:
    """Commit the sink: softmax (pre-RoPE k, v) = (-1, -1) as [1, 1, 1, 2]."""

    memory.absorb_sink(_entry_caches(memory, -1.0))


def _add(memory: CausalMemory) -> None:
    """Commit ONE chunk numbered by the memory's count: softmax (pre-RoPE k, v) = (id, id) as [1, 1, 1, 2]."""

    memory.absorb(_entry_caches(memory, float(memory.committed)))


def _fill(memory: CausalMemory, chunks: int) -> None:
    _sink(memory)
    for _ in range(chunks):
        _add(memory)


def _identity_rope(slot=None):
    return torch.ones(1, 1, 1, 1, dtype=torch.complex128)   # [B, 1, T = 1, D / 2 = 1]


@pytest.mark.parametrize(
    "window,chunks,expected",
    [
        (None, 5, [0, 1, 2, 3, 4]),
        (24, 3, [0, 1, 2]),
        (3, 3, [0, 1, 2]),
        (3, 4, [1, 2, 3]),
        (3, 7, [4, 5, 6]),
        (1, 4, [3]),
        (2, 1, [0]),
    ],
)
def test_the_read_is_the_sink_and_the_newest_window_of_committed_chunks(window, chunks, expected):
    memory = CausalMemory(_Model(["GatedDeltaNet", "GatedSoftmaxAttention"]), window)
    _fill(memory, chunks)
    assert memory.read_chunk_ids() == expected
    k, v = memory.layer_caches(commit=False, rope_of=_identity_rope, sink_rope=_identity_rope())[1].softmax_context
    assert k[..., 0].reshape(-1).tolist() == v[..., 0].reshape(-1).tolist() == [-1.0] + [float(c) for c in expected]
    assert memory.gdn[0][0].item() == float(chunks - 1)       # the GDN state of the last commit, never windowed
    assert len(memory.chunks[1]) == len(expected)              # only what can be read again is kept


def test_the_sink_opens_the_memory_once():
    memory = CausalMemory(_Model(["GatedDeltaNet", "GatedSoftmaxAttention"]), 3)
    assert memory.read_plan() == ([], 0)
    caches = memory.layer_caches(commit=False)   # nothing to read before the sink
    assert caches[0].gdn_state is None and caches[1].softmax_context is None
    with pytest.raises(RuntimeError, match="commit the sink first"):
        _add(memory)
    _sink(memory)
    assert memory.has_sink and memory.committed == 0 and memory.gdn[0][0].item() == -1.0
    k, _ = memory.layer_caches(commit=False, sink_rope=_identity_rope())[1].softmax_context   # the sink alone
    assert k[..., 0].reshape(-1).tolist() == [-1.0]
    with pytest.raises(ValueError, match="sink's rope"):
        memory.layer_caches(commit=False)
    with pytest.raises(RuntimeError, match="committed once"):
        _sink(memory)


@pytest.mark.parametrize(
    "window,chunks,ids,origin",
    [
        (3, 0, [], 0),             # the frozen chunk 0's commit reads the sink alone
        (3, 2, [0, 1], 0),
        (3, 3, [0, 1, 2], 0),
        (3, 4, [1, 2, 3], 1),      # chunk 4's read: chunks 1..3 in slots 0..2, chunk 4 itself in slot 3
        (3, 7, [4, 5, 6], 4),
        (None, 6, [0, 1, 2, 3, 4, 5], 0),
    ],
)
def test_every_read_is_numbered_from_zero_at_its_oldest_chunk(window, chunks, ids, origin):
    memory = CausalMemory(_Model(["GatedSoftmaxAttention"]), window)
    _fill(memory, chunks)
    assert memory.read_plan() == (ids, origin)
    asked = []

    def rope_of(slot):
        asked.append(slot)
        return _identity_rope(slot)

    memory.layer_caches(commit=False, rope_of=rope_of, sink_rope=_identity_rope())
    assert asked == list(range(len(ids)))


def test_reads_rotate_the_pre_rope_keys_at_their_slots():
    """The stored keys are pre-RoPE: a read multiplies them by its own phase per slot (the sink by its own), values stay
    untouched."""

    memory = CausalMemory(_Model(["GatedSoftmaxAttention"]), 2)
    _fill(memory, 5)   # the sink, chunks 0..4; the read takes chunks 3, 4 (origin 3) in slots 0, 1

    def phase(angle):
        return torch.polar(torch.ones(1, 1, 1, 1, dtype=torch.float64), torch.full((1, 1, 1, 1), angle, dtype=torch.float64))

    k, v = memory.layer_caches(commit=False, rope_of=lambda slot: phase(1.0 + slot), sink_rope=phase(0.5))[0].softmax_context
    expected = []
    for value, angle in ((-1, 0.5), (3, 1.0), (4, 2.0)):
        z = complex(value, value) * complex(math.cos(angle), math.sin(angle))
        expected.append([z.real, z.imag])
    torch.testing.assert_close(k.reshape(3, 2), torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-12)
    assert v[..., 0].reshape(-1).tolist() == [-1.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="rope_of"):
        memory.layer_caches(commit=False, sink_rope=phase(0.5))
    with pytest.raises(ValueError, match=">= 1 committed chunks"):
        CausalMemory(_Model(["GatedSoftmaxAttention"]), 0)


# -- the session on a tiny model -----------------------------------------------------------------------------------------


def _tiny_model(window=2, memory_content=MEMORY_VIDEO_AND_ACTION, phase=ALIGNED_PHASE_PADDED):
    config = PolicyConfig.from_sana_kwargs(
        depth=4, hidden_size=64, patch_size=(1, 1, 1), num_heads=2, in_channels=4, caption_channels=16,
        model_max_length=7, linear_head_dim=32, softmax_head_dim=32, softmax_ratio=0.5, attn_res_block_size=2,
        use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True,
        config=_Cfg(model=_Cfg(extra={"rope": "aligned"}), scheduler=_Cfg(action_flow_shift=1.0), vae=_Cfg(vae_stride=[8, 32, 32])),
    )
    contract = CausalContract(
        actions_per_chunk=32, video_frame_stride=4, latent_frames_per_chunk=1, sliding_window_chunks=window,
        memory_content=memory_content, model_fps=25.0, tier_num_frames=33,
        multiview="sana_pixel", visual_layout="sana_pixel_canvas", rope="aligned", dt_bias_init=-5.0, a_log_init=0.0,
        reset_legacy_beta=False, aligned_phase_window=phase,
    )
    torch.manual_seed(0)
    model = CausalPolicyModel(config, contract)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    return model.eval()


def _session(model, caption, null, window=2, video_cfg=6.0, steps=3):
    return CausalPolicySession(
        model, caption=caption, unconditional_caption=null, steps=steps, flow_shift=5.0, action_flow_shift=1.0,
        video_cfg_scale=video_cfg, action_cfg_scale=1.0, model_fps=25.0, window=window,
    )


def _captions(seed):
    g = torch.Generator().manual_seed(seed)
    caption = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    null = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    return caption, null, g


def _open(session, sink, frozen, state, mask):
    session.commit_sink(sink)
    session.commit_chunk(frozen, torch.zeros(1, 32, 80), anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask)


@pytest.mark.parametrize("memory_content", [MEMORY_VIDEO_AND_ACTION, MEMORY_VIDEO_ONLY], ids=["video_and_action", "video_only"])
def test_the_session_runs_a_deterministic_episode_on_cpu(memory_content):
    model = _tiny_model(window=2, memory_content=memory_content)
    caption, null, g = _captions(1)
    sink, frozen = torch.randn(1, 4, 1, 3, 4, generator=g), torch.randn(1, 4, 1, 3, 4, generator=g)
    clean = [torch.randn(1, 4, 1, 3, 4, generator=g) for _ in range(5)]
    mask = torch.zeros(1, 32, 80, dtype=torch.bool)
    mask[..., 7:17] = True
    mask[..., 36:46] = True
    state = torch.randn(1, 80, generator=g)

    def run():
        session = _session(model, caption, null)
        _open(session, sink, frozen, state, mask)
        outs = []
        for c in range(1, 5):
            assert session.chunk_idx == c and session.read_chunk_ids() == list(range(max(0, c - 2), c))
            gen = torch.Generator().manual_seed(100 + c)
            action, video = session.generate_chunk(
                anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask, generator=gen,
                gripper_bounds=((-1.0, 1.0), (-1.0, 1.0)),
            )
            assert action.shape == (1, 32, 80) and video.shape == (1, 4, 1, 3, 4)
            assert torch.count_nonzero(action[~mask]) == 0
            assert float(action[..., 16].abs().max()) <= 1.0 and float(action[..., 45].abs().max()) <= 1.0
            outs.append((action, video))
            session.commit_chunk(clean[c], action, anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask)
        assert session.chunk_idx == 5 and session.read_chunk_ids() == [3, 4] and session.softmax_origin() == 3
        return outs

    first, second = run(), run()
    for (a1, v1), (a2, v2) in zip(first, second):
        assert torch.equal(a1, a2) and torch.equal(v1, v2)


def test_every_text_stream_commits_on_its_own_clock():
    """CFG is linear in the two streams, so with one solver step the guided video at s = 2 is 2 * (cond only) - (null
    only) exactly -- past the softmax window too, which holds only when every stream's commit reads on its own clock
    (a commit runs the streams one after the other; a regression found while porting)."""

    model = _tiny_model(window=2).double()
    caption, null, g = _captions(4)
    caption, null = Caption(caption.y.double(), caption.mask), Caption(null.y.double(), null.mask)
    sink, frozen = torch.randn(1, 4, 1, 3, 4, generator=g).double(), torch.randn(1, 4, 1, 3, 4, generator=g).double()
    clean = [torch.randn(1, 4, 1, 3, 4, generator=g).double() for _ in range(6)]
    noise = [torch.randn(1, 4, 1, 3, 4, generator=g).double() for _ in range(6)]
    mask = torch.ones(1, 32, 80, dtype=torch.bool)
    state = torch.randn(1, 80, generator=g).double()

    def episode(first, second, scale):
        session = CausalPolicySession(
            model, caption=first, unconditional_caption=second, steps=1, flow_shift=5.0, action_flow_shift=1.0,
            video_cfg_scale=scale, action_cfg_scale=1.0, model_fps=25.0, window=2, autocast_dtype=None,
        )
        _open(session, sink, frozen, state.float(), mask)
        videos = []
        for c in range(1, 6):
            _, video = session.generate_chunk(
                anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask,
                video_noise=noise[c], action_noise=torch.zeros(1, 32, 80, dtype=torch.float64),
            )
            videos.append(video)
            session.commit_chunk(clean[c], torch.zeros(1, 32, 80, dtype=torch.float64), anchor_state=state,
                                 anchor_state_mask=mask[0, 0], action_mask=mask)
        return videos

    cond, uncond, guided = episode(caption, null, 1.0), episode(null, caption, 1.0), episode(caption, null, 2.0)
    for c, (a, b, d) in enumerate(zip(cond, uncond, guided), start=1):
        torch.testing.assert_close(2 * a - b, d, rtol=1e-9, atol=1e-12, msg=lambda m: f"chunk {c}: {m}")


def test_reset_memory_reopens_the_episode():
    model = _tiny_model(window=2)
    caption, null, g = _captions(2)
    sink, frozen = (torch.randn(1, 4, 1, 3, 4, generator=g) for _ in range(2))
    mask = torch.ones(1, 32, 80, dtype=torch.bool)
    state = torch.randn(1, 80, generator=g)
    kwargs = dict(anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask)

    fresh = _session(model, caption, null)
    _open(fresh, sink, frozen, state, mask)
    reference = fresh.generate_chunk(generator=torch.Generator().manual_seed(5), **kwargs)
    used = _session(model, caption, null)
    _open(used, torch.randn(1, 4, 1, 3, 4, generator=g), torch.randn(1, 4, 1, 3, 4, generator=g), state, mask)
    for _ in range(3):
        used.commit_chunk(torch.randn(1, 4, 1, 3, 4, generator=g), torch.randn(1, 32, 80, generator=g), **kwargs)
    with_memory = used.generate_chunk(generator=torch.Generator().manual_seed(5), **kwargs)
    assert not torch.equal(with_memory[0], reference[0])
    used.reset_memory()
    assert used.chunk_idx == 0 and used.read_chunk_ids() == [] and not used.has_sink
    with pytest.raises(RuntimeError, match="commit_sink first"):
        used.generate_chunk(generator=torch.Generator().manual_seed(5), **kwargs)
    used.commit_sink(sink)
    with pytest.raises(RuntimeError, match="first emission is chunk 1"):   # a padded window's chunk 0 is memory only
        used.generate_chunk(generator=torch.Generator().manual_seed(5), **kwargs)
    used.commit_chunk(frozen, torch.zeros(1, 32, 80), anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask)
    again = used.generate_chunk(generator=torch.Generator().manual_seed(5), **kwargs)
    assert torch.equal(again[0], reference[0]) and torch.equal(again[1], reference[1])


def test_a_session_refuses_misuse():
    model = _tiny_model()
    caption = Caption(torch.zeros(1, 1, 1, 7, 16), torch.ones(1, 1, 7, dtype=torch.int16))
    with pytest.raises(ValueError, match="unconditional caption"):
        CausalPolicySession(
            model, caption=caption, unconditional_caption=None, steps=2, flow_shift=5.0, action_flow_shift=1.0,
            video_cfg_scale=6.0, action_cfg_scale=1.0, model_fps=25.0, window=2,
        )
    session = CausalPolicySession(
        model, caption=caption, unconditional_caption=None, steps=2, flow_shift=5.0, action_flow_shift=1.0,
        video_cfg_scale=1.0, action_cfg_scale=1.0, model_fps=25.0, window=2,
    )
    mask = torch.ones(1, 32, 80, dtype=torch.bool)
    kwargs = dict(anchor_state=torch.zeros(1, 80), anchor_state_mask=mask[0, 0], action_mask=mask)
    with pytest.raises(ValueError, match="sink_latent must be"):
        session.commit_sink(torch.zeros(1, 4, 2, 3, 4))   # two frames: not the first frame alone
    _open(session, torch.zeros(1, 4, 1, 3, 4), torch.zeros(1, 4, 1, 3, 4), torch.zeros(1, 80), mask)
    with pytest.raises(ValueError, match="video_noise must be"):
        session.generate_chunk(video_noise=torch.zeros(1, 4, 2, 3, 4), **kwargs)   # two frames: not one chunk's f_c
    session.generate_chunk(**kwargs)
    with pytest.raises(ValueError, match="latent grid"):
        session.generate_chunk(video_noise=torch.zeros(1, 4, 1, 4, 4), **kwargs)   # the latent grid is fixed per session
    with pytest.raises(ValueError, match="chunk_latent"):
        session.commit_chunk(torch.zeros(1, 4, 2, 3, 4), torch.zeros(1, 32, 80), **kwargs)
    with pytest.raises(ValueError, match="executed_action"):
        session.commit_chunk(torch.zeros(1, 4, 1, 3, 4), torch.zeros(1, 31, 80), **kwargs)


def test_a_first_frame_session_emits_from_chunk_0():
    """aligned_phase_window first_frame: no frozen chunk, chunk 0 is generated at the first frame after the sink, which
    it reads as its observation, and the cycle runs on from there; generating before the sink is refused."""

    model = _tiny_model(window=2, memory_content=MEMORY_VIDEO_ONLY, phase=ALIGNED_PHASE_FIRST_FRAME)
    caption, null, g = _captions(6)
    sink = torch.randn(1, 4, 1, 3, 4, generator=g)
    clean = [torch.randn(1, 4, 1, 3, 4, generator=g) for _ in range(4)]
    mask = torch.ones(1, 32, 80, dtype=torch.bool)
    state = torch.randn(1, 80, generator=g)
    kwargs = dict(anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask)

    def generate(session, c):
        return session.generate_chunk(generator=torch.Generator().manual_seed(40 + c), **kwargs)

    def run():
        session = _session(model, caption, null)
        assert session.first_chunk == 0
        session.commit_sink(sink)
        outs = [generate(session, 0)]            # reads the sink
        for c in range(4):
            session.commit_chunk(clean[c], outs[c][0], **kwargs)
            if c < 3:
                assert session.chunk_idx == c + 1 and session.read_chunk_ids() == list(range(max(0, c - 1), c + 1))
                outs.append(generate(session, c + 1))
        return outs

    first, second = run(), run()
    for (a1, v1), (a2, v2) in zip(first, second):
        assert torch.equal(a1, a2) and torch.equal(v1, v2)
    wrong = _session(model, caption, null)
    with pytest.raises(RuntimeError, match="commit_sink first"):
        generate(wrong, 0)
