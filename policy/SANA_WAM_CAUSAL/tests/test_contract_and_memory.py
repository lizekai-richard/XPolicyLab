"""Contract resolution from a causal training yaml and the session memory's softmax read rules (CPU, no Sana)."""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import yaml

import _paths

from sana_wam_causal.contract import bidirectional_view, resolve_causal_contract
from sana_wam_causal.model import CausalPolicyModel
from sana_wam_causal.session import Caption, CausalMemory, CausalPolicySession
from sana_wam_min.config import THREE_VIEW_POLICY_FACTORY, policy_config_from_train_config
from sana_wam_min.policy_model.config import PolicyConfig

FIXTURE = os.path.join(_paths.FIXTURES, "config_causal_vanilla52k_m48n24_derived.yaml")


@pytest.fixture(scope="module")
def causal_cfg():
    with open(FIXTURE) as handle:
        return yaml.safe_load(handle)


def test_the_m48n24_recipe_resolves(causal_cfg):
    contract = resolve_causal_contract(causal_cfg)
    assert (contract.actions_per_chunk, contract.video_frame_stride, contract.latent_frames_per_chunk) == (32, 4, 1)
    assert contract.sliding_window_chunks == 24 and contract.obs_in_first_chunk
    assert (contract.multiview, contract.visual_layout, contract.rope) == ("sana_pixel", "sana_pixel_canvas", "aligned")
    assert (contract.dt_bias_init, contract.a_log_init) == (-5.0, 0.0)


def test_the_bidirectional_view_restores_the_sft_tier(causal_cfg):
    view = bidirectional_view(causal_cfg)
    assert view["model"]["model"] == THREE_VIEW_POLICY_FACTORY
    assert view["data"]["num_frames"] == 33 and view["data"]["multi_fps"] == {"25": [33]}
    assert causal_cfg["data"]["num_frames"] == 1537  # the source is left untouched
    config = policy_config_from_train_config(view)
    assert (config.depth, config.hidden_size, tuple(config.softmax_layer_indices)) == (32, 2560, (3, 7, 11, 15, 19, 23, 27, 31))
    assert config.rope == "aligned" and config.separate_action_schedule


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda c: c["model"].__setitem__("model", THREE_VIEW_POLICY_FACTORY), "not the chunk-causal policy class"),
        (lambda c: c["model"]["extra"]["chunk_causal_policy"].__setitem__("obs_in_first_chunk", False), "obs_in_first_chunk"),
        (lambda c: c["model"]["extra"].__setitem__("rope", "independent"), "aligned only"),
        (lambda c: c["model"].__setitem__("use_time_conditioning", True), "use_time_conditioning"),
        (lambda c: c["model"]["extra"].__setitem__("sana_pixel_pad", "masked"), "sana_pixel_pad"),
        (lambda c: c["data"]["extra"].__setitem__("multiview", "sana_latent"), "sana_pixel canvas"),
        (lambda c: c["data"].__setitem__("type", "RoboDojoSFTDataset"), "causal chunk-window dataset"),
        (lambda c: c["model"]["extra"]["chunk_causal_policy"].__setitem__("actions_per_chunk", {"25": 24}), "disagree"),
    ],
    ids=["sft-class", "obs-segment", "independent", "time-conditioning", "padmask", "strip", "sft-dataset", "chunk-mismatch"],
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


def _fill(memory: CausalMemory, entries: int) -> None:
    from sana_wam_causal.layers import LayerCache

    for entry in range(entries):
        caches = []
        for kind in memory.kinds:
            cache = LayerCache(commit=True)
            if kind == "GatedDeltaNet":
                cache.gdn_next = (torch.full((1,), float(entry)), torch.full((1,), float(entry)))
            else:
                cache.softmax_new = (torch.full((1, 1, 1, 1), float(entry)), torch.full((1, 1, 1, 1), float(entry)))
            caches.append(cache)
        memory.absorb(caches)


@pytest.mark.parametrize(
    "window,rule,entries,expected",
    [
        (None, "training", 5, [0, 1, 2, 3, 4]),
        (24, "training", 3, [0, 1, 2]),
        (3, "training", 3, [0, 1, 2]),
        (3, "training", 4, [1, 2, 3]),
        (3, "training", 7, [4, 5, 6]),
        (3, "keep_obs", 7, [0, 5, 6]),
        (1, "training", 4, [3]),
    ],
)
def test_read_rules(window, rule, entries, expected):
    memory = CausalMemory(_Model(["GatedDeltaNet", "GatedSoftmaxAttention"]), window, rule)
    _fill(memory, entries)
    assert memory.read_entry_ids() == expected
    k, _v = memory._context(1)
    assert k.reshape(-1).tolist() == [float(e) for e in expected]
    assert memory.gdn[0][0].item() == float(entries - 1)
    if rule == "training" and window is not None and entries > window:
        assert memory.obs[1] is None  # the observation entry is dropped once it can never be read again


def _tiny_model(window=2):
    config = PolicyConfig.from_sana_kwargs(
        depth=4, hidden_size=64, patch_size=(1, 1, 1), num_heads=2, in_channels=4, caption_channels=16,
        model_max_length=7, linear_head_dim=32, softmax_head_dim=32, softmax_ratio=0.5, attn_res_block_size=2,
        use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True,
        config=_Cfg(model=_Cfg(extra={"rope": "aligned"}), scheduler=_Cfg(action_flow_shift=1.0), vae=_Cfg(vae_stride=[8, 32, 32])),
    )
    from sana_wam_causal.contract import CausalContract

    contract = CausalContract(
        actions_per_chunk=32, video_frame_stride=4, latent_frames_per_chunk=1, sliding_window_chunks=window,
        obs_in_first_chunk=True, model_fps=25.0, tier_num_frames=33, multiview="sana_pixel",
        visual_layout="sana_pixel_canvas", rope="aligned", dt_bias_init=-5.0, a_log_init=0.0, reset_legacy_beta=False,
    )
    torch.manual_seed(0)
    model = CausalPolicyModel(config, contract)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    return model.eval()


def test_the_session_runs_a_deterministic_episode_on_cpu():
    model = _tiny_model(window=2)
    g = torch.Generator().manual_seed(1)
    caption = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    null = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    obs = torch.randn(1, 4, 1, 3, 4, generator=g)
    clean = [torch.randn(1, 4, 1, 3, 4, generator=g) for _ in range(4)]
    mask = torch.zeros(1, 32, 80, dtype=torch.bool)
    mask[..., 7:17] = True
    mask[..., 36:46] = True
    state = torch.randn(1, 80, generator=g)

    def run():
        session = CausalPolicySession(
            model, caption=caption, unconditional_caption=null, steps=3, flow_shift=5.0, action_flow_shift=1.0,
            video_cfg_scale=6.0, action_cfg_scale=1.0, model_fps=25.0, window=2,
        )
        session.commit_observation(obs)
        outs = []
        for c in range(4):
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
        assert session.chunk_idx == 4 and session.read_entry_ids() == [3, 4]
        return outs

    first, second = run(), run()
    for (a1, v1), (a2, v2) in zip(first, second):
        assert torch.equal(a1, a2) and torch.equal(v1, v2)


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
    with pytest.raises(RuntimeError, match="commit_observation first"):
        session.generate_chunk(anchor_state=torch.zeros(1, 80), anchor_state_mask=mask[0, 0], action_mask=mask)
    session.commit_observation(torch.zeros(1, 4, 1, 3, 4))
    with pytest.raises(RuntimeError, match="once per episode"):
        session.commit_observation(torch.zeros(1, 4, 1, 3, 4))
    with pytest.raises(ValueError, match="chunk_latent"):
        session.commit_chunk(
            torch.zeros(1, 4, 2, 3, 4), torch.zeros(1, 32, 80), anchor_state=torch.zeros(1, 80),
            anchor_state_mask=mask[0, 0], action_mask=mask,
        )
