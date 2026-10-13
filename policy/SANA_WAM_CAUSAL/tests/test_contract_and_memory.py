"""Contract resolution from a causal training yaml and the session memory's softmax read rule (CPU, no Sana)."""

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
    MEMORY_VIDEO_AND_ACTION,
    MEMORY_VIDEO_ONLY,
    CausalContract,
    bidirectional_view,
    resolve_causal_contract,
)
from sana_wam_causal.model import CausalPolicyModel
from sana_wam_causal.session import Caption, CausalMemory, CausalPolicySession
from sana_wam_min.config import THREE_VIEW_POLICY_FACTORY, policy_config_from_train_config
from sana_wam_min.policy_model.config import PolicyConfig

FIXTURE = os.path.join(_paths.FIXTURES, "config_causal_pantheon14k_m48n24_videoandaction_s5000.yaml")
FIXTURE_VIDEOONLY = os.path.join(_paths.FIXTURES, "config_causal_pantheon14k_m48n24_videoonly_s5000.yaml")
FIXTURE_RETIRED = os.path.join(_paths.FIXTURES, "config_causal_vanilla52k_m48n24_s7646.yaml")


def _load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def causal_cfg():
    return _load(FIXTURE)


@pytest.mark.parametrize(
    "path,memory_content",
    [(FIXTURE, MEMORY_VIDEO_AND_ACTION), (FIXTURE_VIDEOONLY, MEMORY_VIDEO_ONLY)],
    ids=["no-key-video_and_action", "video_only"],
)
def test_the_pantheon14k_recipes_resolve(path, memory_content):
    contract = resolve_causal_contract(_load(path))
    assert (contract.actions_per_chunk, contract.video_frame_stride, contract.latent_frames_per_chunk) == (32, 4, 1)
    assert contract.sliding_window_chunks == 24 and contract.memory_content == memory_content
    assert (contract.multiview, contract.visual_layout, contract.rope) == ("sana_pixel", "sana_pixel_canvas", "aligned")
    assert (contract.dt_bias_init, contract.a_log_init) == (-5.0, 0.0)
    assert contract.describe().startswith("protocol=per_chunk_obs ") and f"memory_content={memory_content}" in contract.describe()


def test_the_bidirectional_view_restores_the_sft_tier(causal_cfg):
    view = bidirectional_view(causal_cfg)
    assert view["model"]["model"] == THREE_VIEW_POLICY_FACTORY
    assert view["data"]["num_frames"] == 33 and view["data"]["multi_fps"] == {"25": [33]}
    assert causal_cfg["data"]["num_frames"] == 1537  # the source is left untouched
    config = policy_config_from_train_config(view)
    assert (config.depth, config.hidden_size, tuple(config.softmax_layer_indices)) == (32, 2560, (3, 7, 11, 15, 19, 23, 27, 31))
    assert config.rope == "aligned" and config.separate_action_schedule


def test_the_retired_chunk0_observation_checkpoint_is_refused():
    """The real config.yaml of the vanilla52k causal checkpoint s7646 (chunk-0-only observation) is refused by name."""

    with pytest.raises(ValueError, match="obs_in_first_chunk=True: this checkpoint trained a retired observation layout"):
        resolve_causal_contract(_load(FIXTURE_RETIRED))


def _policy(cfg):
    return cfg["model"]["extra"]["chunk_causal_policy"]


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda c: c["model"].__setitem__("model", THREE_VIEW_POLICY_FACTORY), "not the chunk-causal policy class"),
        (lambda c: _policy(c).__setitem__("obs_in_first_chunk", False), "retired observation layout"),
        (lambda c: _policy(c).__setitem__("memory_content", "video_and_state"), "memory_content"),
        (lambda c: c["model"]["extra"].__setitem__("rope", "independent"), "aligned only"),
        (lambda c: c["model"].__setitem__("use_time_conditioning", True), "use_time_conditioning"),
        (lambda c: c["model"]["extra"].__setitem__("sana_pixel_pad", "masked"), "sana_pixel_pad"),
        (lambda c: c["data"]["extra"].__setitem__("multiview", "sana_latent"), "sana_pixel canvas"),
        (lambda c: c["data"].__setitem__("type", "RoboDojoSFTDataset"), "causal chunk-window dataset"),
        (lambda c: _policy(c).__setitem__("actions_per_chunk", {"25": 24}), "disagree"),
    ],
    ids=["sft-class", "obs-key", "memory-content", "independent", "time-conditioning", "padmask", "strip", "sft-dataset",
         "chunk-mismatch"],
)
def test_unsupported_contracts_are_refused(causal_cfg, edit, match):
    cfg = copy.deepcopy(causal_cfg)
    edit(cfg)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        resolve_causal_contract(cfg)


class _Cfg(SimpleNamespace):
    def __getattr__(self, name):
        return None


class _Model:
    def __init__(self, kinds):
        self.blocks = [type("B", (), {"attn_type": kind})() for kind in kinds]


def _add(memory: CausalMemory) -> None:
    """Commit ONE chunk numbered by the memory's count: softmax (pre-RoPE k, v) = (id, id) as [1, 1, 1, 2]."""

    from sana_wam_causal.layers import LayerCache

    chunk = memory.committed
    caches = []
    for kind in memory.kinds:
        cache = LayerCache(commit=True)
        if kind == "GatedDeltaNet":
            cache.gdn_next = (torch.full((1,), float(chunk)), torch.full((1,), float(chunk)))
        else:
            cache.softmax_new = (torch.full((1, 1, 1, 2), float(chunk), dtype=torch.float64),
                                 torch.full((1, 1, 1, 2), float(chunk), dtype=torch.float64))
        caches.append(cache)
    memory.absorb(caches)


def _fill(memory: CausalMemory, chunks: int) -> None:
    for _ in range(chunks):
        _add(memory)


def _identity_rope(slot):
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
def test_the_read_is_the_newest_window_of_committed_chunks(window, chunks, expected):
    memory = CausalMemory(_Model(["GatedDeltaNet", "GatedSoftmaxAttention"]), window)
    _fill(memory, chunks)
    assert memory.read_chunk_ids() == expected
    k, v = memory.layer_caches(commit=False, rope_of=_identity_rope)[1].softmax_context
    assert k[..., 0].reshape(-1).tolist() == v[..., 0].reshape(-1).tolist() == [float(c) for c in expected]
    assert memory.gdn[0][0].item() == float(chunks - 1)       # the GDN state of the last commit, never windowed
    assert len(memory.chunks[1]) == len(expected)              # only what can be read again is kept


def test_an_empty_memory_reads_nothing():
    memory = CausalMemory(_Model(["GatedDeltaNet", "GatedSoftmaxAttention"]), 3)
    assert memory.read_plan() == ([], 0)
    caches = memory.layer_caches(commit=False)   # no rope needed: nothing is read
    assert caches[0].gdn_state is None and caches[1].softmax_context is None


@pytest.mark.parametrize(
    "window,chunks,ids,origin",
    [
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

    memory.layer_caches(commit=False, rope_of=rope_of)
    assert asked == list(range(len(ids)))


def test_reads_rotate_the_pre_rope_keys_at_their_slots():
    """The stored keys are pre-RoPE: a read multiplies them by its own phase per slot, values stay untouched."""

    memory = CausalMemory(_Model(["GatedSoftmaxAttention"]), 2)
    _fill(memory, 5)   # chunks 0..4; the read takes chunks 3, 4 (origin 3) in slots 0, 1

    def rope_of(slot):
        angle = 1.0 + slot
        return torch.polar(torch.ones(1, 1, 1, 1, dtype=torch.float64), torch.full((1, 1, 1, 1), angle, dtype=torch.float64))

    k, v = memory.layer_caches(commit=False, rope_of=rope_of)[0].softmax_context
    expected = []
    for chunk, angle in ((3, 1.0), (4, 2.0)):
        z = complex(chunk, chunk) * complex(math.cos(angle), math.sin(angle))
        expected.append([z.real, z.imag])
    torch.testing.assert_close(k.reshape(2, 2), torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-12)
    assert v[..., 0].reshape(-1).tolist() == [3.0, 4.0]
    with pytest.raises(ValueError, match="rope_of"):
        memory.layer_caches(commit=False)
    with pytest.raises(ValueError, match=">= 1 committed chunks"):
        CausalMemory(_Model(["GatedSoftmaxAttention"]), 0)


def _tiny_model(window=2, memory_content=MEMORY_VIDEO_AND_ACTION):
    config = PolicyConfig.from_sana_kwargs(
        depth=4, hidden_size=64, patch_size=(1, 1, 1), num_heads=2, in_channels=4, caption_channels=16,
        model_max_length=7, linear_head_dim=32, softmax_head_dim=32, softmax_ratio=0.5, attn_res_block_size=2,
        use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True,
        config=_Cfg(model=_Cfg(extra={"rope": "aligned"}), scheduler=_Cfg(action_flow_shift=1.0), vae=_Cfg(vae_stride=[8, 32, 32])),
    )
    contract = CausalContract(
        actions_per_chunk=32, video_frame_stride=4, latent_frames_per_chunk=1, sliding_window_chunks=window,
        memory_content=memory_content, model_fps=25.0, tier_num_frames=33, multiview="sana_pixel",
        visual_layout="sana_pixel_canvas", rope="aligned", dt_bias_init=-5.0, a_log_init=0.0, reset_legacy_beta=False,
    )
    torch.manual_seed(0)
    model = CausalPolicyModel(config, contract)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    return model.eval()


def _session(model, caption, null, window=2):
    return CausalPolicySession(
        model, caption=caption, unconditional_caption=null, steps=3, flow_shift=5.0, action_flow_shift=1.0,
        video_cfg_scale=6.0, action_cfg_scale=1.0, model_fps=25.0, window=window,
    )


@pytest.mark.parametrize("memory_content", [MEMORY_VIDEO_AND_ACTION, MEMORY_VIDEO_ONLY])
def test_the_session_runs_a_deterministic_episode_on_cpu(memory_content):
    model = _tiny_model(window=2, memory_content=memory_content)
    g = torch.Generator().manual_seed(1)
    caption = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    null = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    observations = [torch.randn(1, 4, 1, 3, 4, generator=g) for _ in range(4)]
    clean = [torch.randn(1, 4, 1, 3, 4, generator=g) for _ in range(4)]
    mask = torch.zeros(1, 32, 80, dtype=torch.bool)
    mask[..., 7:17] = True
    mask[..., 36:46] = True
    state = torch.randn(1, 80, generator=g)

    def run():
        session = _session(model, caption, null)
        outs = []
        for c in range(4):
            assert session.read_chunk_ids() == list(range(max(0, c - 2), c))
            gen = torch.Generator().manual_seed(100 + c)
            action, video = session.generate_chunk(
                observations[c], anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask, generator=gen,
                gripper_bounds=((-1.0, 1.0), (-1.0, 1.0)),
            )
            assert action.shape == (1, 32, 80) and video.shape == (1, 4, 1, 3, 4)
            assert torch.count_nonzero(action[~mask]) == 0
            assert float(action[..., 16].abs().max()) <= 1.0 and float(action[..., 45].abs().max()) <= 1.0
            outs.append((action, video))
            session.commit_chunk(clean[c], action, anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask)
        assert session.chunk_idx == 4 and session.read_chunk_ids() == [2, 3] and session.softmax_origin() == 2
        return outs

    first, second = run(), run()
    for (a1, v1), (a2, v2) in zip(first, second):
        assert torch.equal(a1, a2) and torch.equal(v1, v2)


def test_reset_memory_makes_the_next_chunk_a_first_chunk():
    model = _tiny_model(window=2)
    g = torch.Generator().manual_seed(2)
    caption = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    null = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    observation = torch.randn(1, 4, 1, 3, 4, generator=g)
    mask = torch.ones(1, 32, 80, dtype=torch.bool)
    state = torch.randn(1, 80, generator=g)
    kwargs = dict(anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask)

    fresh = _session(model, caption, null)
    reference = fresh.generate_chunk(observation, generator=torch.Generator().manual_seed(5), **kwargs)
    used = _session(model, caption, null)
    for c in range(3):
        used.commit_chunk(torch.randn(1, 4, 1, 3, 4, generator=g), torch.randn(1, 32, 80, generator=g), **kwargs)
    with_memory = used.generate_chunk(observation, generator=torch.Generator().manual_seed(5), **kwargs)
    assert not torch.equal(with_memory[0], reference[0])
    used.reset_memory()
    assert used.chunk_idx == 0 and used.read_chunk_ids() == []
    again = used.generate_chunk(observation, generator=torch.Generator().manual_seed(5), **kwargs)
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
    with pytest.raises(ValueError, match="observation_latent must be"):
        session.generate_chunk(torch.zeros(1, 4, 2, 3, 4), **kwargs)   # two frames: not one observation
    session.generate_chunk(torch.zeros(1, 4, 1, 3, 4), **kwargs)
    with pytest.raises(ValueError, match="latent grid"):
        session.generate_chunk(torch.zeros(1, 4, 1, 4, 4), **kwargs)   # the canvas grid is fixed per session
    with pytest.raises(ValueError, match="chunk_latent"):
        session.commit_chunk(torch.zeros(1, 4, 2, 3, 4), torch.zeros(1, 32, 80), **kwargs)
    with pytest.raises(ValueError, match="executed_action"):
        session.commit_chunk(torch.zeros(1, 4, 1, 3, 4), torch.zeros(1, 31, 80), **kwargs)
