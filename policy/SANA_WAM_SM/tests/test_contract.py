"""The sparse-memory contract resolved from the four rwm/sparse-memory recipes (3fa3e9689: strided s = 25 and fixed
m = 16, each with its latent-store twin), the refusals (a hierarchical-memory checkpoint among them), and the shared
front-end's reading of the recipes' bidirectional view."""

from __future__ import annotations

import copy
import os

import pytest
import yaml

import _paths

from sana_wam_sm import sparse_memory as sm
from sana_wam_sm.contract import (
    DEPLOY_CACHED,
    DEPLOY_UNCACHED,
    bidirectional_view,
    is_sparse_memory_config,
    resolve_sm_contract,
)
from sana_wam_min.config import THREE_VIEW_POLICY_FACTORY, policy_config_from_train_config, sft_options_from_train_config

STRIDED = os.path.join(_paths.FIXTURES, "recipe_sm_strided_s25_3fa3e9689.yaml")
STRIDED_CACHED = os.path.join(_paths.FIXTURES, "recipe_sm_strided_s25_cached_3fa3e9689.yaml")
FIXED = os.path.join(_paths.FIXTURES, "recipe_sm_fixed_m16_3fa3e9689.yaml")
FIXED_CACHED = os.path.join(_paths.FIXTURES, "recipe_sm_fixed_m16_cached_3fa3e9689.yaml")
# a hierarchical-memory checkpoint's frozen config (hm_uniform_noobs epoch_3_step_7250, Sana 6564adc38)
HM_CHECKPOINT = os.path.join(_paths.FIXTURES, "config_hm_uniform_noobs_s7250.yaml")


def _load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def _policy(cfg):
    return cfg["model"]["extra"]["chunk_causal_policy"]


@pytest.mark.parametrize(
    "path,memory,trained",
    [
        (STRIDED, {"memory_type": "strided", "memory_stride": 25}, 8),
        (STRIDED_CACHED, {"memory_type": "strided", "memory_stride": 25}, 8),
        (FIXED, {"memory_type": "fixed", "memory_frames": 16}, 2),
        (FIXED_CACHED, {"memory_type": "fixed", "memory_frames": 16}, 2),
    ],
    ids=["strided", "strided-cached", "fixed", "fixed-cached"],
)
def test_the_recipes_resolve(path, memory, trained):
    cfg = _load(path)
    assert is_sparse_memory_config(cfg)
    contract = resolve_sm_contract(cfg)
    assert contract.memory == memory and contract.trained_memory_latents == trained
    assert (contract.actions_per_chunk, contract.video_frame_stride, contract.tier_num_frames) == (32, 4, 33)
    assert (contract.multiview, contract.visual_layout, contract.rope) == ("sana_pixel", "sana_pixel_canvas", "aligned")
    assert contract.resolve_deploy_form(None) == DEPLOY_CACHED and contract.resolve_deploy_form(" Uncached ") == DEPLOY_UNCACHED
    with pytest.raises(ValueError, match="deploy_form"):
        contract.resolve_deploy_form("streaming")
    size = f"memory_frames={memory['memory_frames']}" if "memory_frames" in memory else f"memory_stride={memory['memory_stride']}"
    assert contract.describe().startswith(f"protocol=sparse_memory memory_type={memory['memory_type']} {size} ")
    assert f" trained_memory_latents={trained} " in contract.describe()


def test_the_latent_store_twins_are_their_recipes():
    """The cached recipes differ by the data-side latent store alone: the same contract and network."""

    for plain, cached in ((STRIDED, STRIDED_CACHED), (FIXED, FIXED_CACHED)):
        assert resolve_sm_contract(_load(plain)) == resolve_sm_contract(_load(cached))
        assert policy_config_from_train_config(bidirectional_view(_load(plain))) == policy_config_from_train_config(
            bidirectional_view(_load(cached))
        )


def test_the_bidirectional_view_restores_the_sft_tier():
    cfg = _load(STRIDED)
    view = bidirectional_view(cfg)
    assert view["model"]["model"] == THREE_VIEW_POLICY_FACTORY
    assert view["data"]["num_frames"] == 33 and view["data"]["multi_fps"] == {"25": [33]}
    assert cfg["data"]["multi_fps"] == {"25": [33, 65, 97, 129, 161, 193, 225, 257, 289]}   # the source is left untouched
    config = policy_config_from_train_config(view)
    assert (config.depth, config.hidden_size, tuple(config.softmax_layer_indices)) == (32, 2560, (3, 7, 11, 15, 19, 23, 27, 31))
    assert config.rope == "aligned" and config.separate_action_schedule
    options = sft_options_from_train_config(view)
    assert options["normalization_sha256"].startswith("1fe3b7e7") and options["tier_num_frames"] == 33
    # the fixed recipe trains from another donor at another lr: the same network
    assert policy_config_from_train_config(bidirectional_view(_load(FIXED))) == config


def test_the_keys_the_sparse_memory_never_reads_are_ignored():
    """Sana's sparse-memory forms read neither memory_content nor sliding_window_chunks nor chunk_observation."""

    reference = resolve_sm_contract(_load(FIXED))
    for key, value in (("memory_content", "video_only"), ("sliding_window_chunks", 24), ("chunk_observation", False)):
        cfg = copy.deepcopy(_load(FIXED))
        _policy(cfg)[key] = value
        assert resolve_sm_contract(cfg) == reference


def test_a_hierarchical_memory_checkpoint_is_refused():
    cfg = _load(HM_CHECKPOINT)
    assert not is_sparse_memory_config(cfg)
    with pytest.raises(ValueError, match="SANA_WAM_HM up to XPolicyLab d57a617"):
        resolve_sm_contract(cfg)


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda c: _policy(c).pop("memory_type"), "without memory_type"),
        (lambda c: [_policy(c).pop(k) for k in ("memory_type", "memory_frames")], "SANA_WAM_CAUSAL"),
        (lambda c: _policy(c).__setitem__("memory_type", "uniform"), "fixed or strided"),
        (lambda c: _policy(c).__setitem__("memory_frames", 12), "multiple of 8"),
        (lambda c: _policy(c).__setitem__("memory_frames", 24), "rung 129"),
        (lambda c: _policy(c).__setitem__("hierarchical_memory", {"gdn": "sparse"}), "SANA_WAM_HM up to XPolicyLab d57a617"),
        (lambda c: _policy(c).__setitem__("memory_uniform", {"sparse_frames": 16}), "SANA_WAM_HM up to XPolicyLab d57a617"),
        (lambda c: _policy(c).__setitem__("memory_split", {"sparse_frames": 16}), "SANA_WAM_HM up to XPolicyLab d57a617"),
        (lambda c: _policy(c).__setitem__("actions_per_chunk", {"25": 64}), "disagree|32 rows"),
        (lambda c: c["model"].__setitem__("model", THREE_VIEW_POLICY_FACTORY), "chunk-causal policy class"),
        (lambda c: c["data"].__setitem__("type", "RoboDojoSFTDataset"), "causal dataset"),
        (lambda c: c["data"].__setitem__("multi_fps", {"25": [33, 50]}), "rung is 1"),
        (lambda c: c["model"]["extra"].__setitem__("rope", "independent"), "aligned"),
        (lambda c: c["model"].__setitem__("use_time_conditioning", True), "use_time_conditioning"),
        (lambda c: c["model"]["extra"].__setitem__("sana_pixel_pad", "masked"), "sana_pixel_pad"),
        (lambda c: c["data"].__setitem__("aspect_ratio_type", "ASPECT_RATIO_SANA_PIXEL_2X2_320_480"), "320 x 512"),
        (lambda c: c["vae"].__setitem__("use_causal_encode", False), "use_causal_encode"),
    ],
    ids=[
        "frames-without-type", "no-memory", "memory-type", "frames-not-x8", "frames-off-the-ladder", "hierarchical",
        "memory-uniform", "memory-split", "chunk", "sft-class", "sft-dataset", "bad-rung", "independent",
        "time-conditioning", "padmask", "canvas-320x480", "non-causal-vae",
    ],
)
def test_unsupported_contracts_are_refused(edit, match):
    cfg = copy.deepcopy(_load(FIXED))
    edit(cfg)
    with pytest.raises((ValueError, NotImplementedError, KeyError), match=match):
        resolve_sm_contract(cfg)


def test_a_non_positive_stride_is_refused():
    cfg = copy.deepcopy(_load(STRIDED))
    _policy(cfg)["memory_stride"] = 0
    with pytest.raises(ValueError, match="positive row count"):
        resolve_sm_contract(cfg)


def test_the_causal_adapter_refuses_a_sparse_memory_checkpoint():
    """SANA_WAM_CAUSAL must not serve a sparse-memory checkpoint as its episode protocol."""

    import sys

    causal_dir = os.path.abspath(os.path.join(_paths.ADAPTER_DIR, "..", "SANA_WAM_CAUSAL"))
    if not os.path.isdir(os.path.join(causal_dir, "sana_wam_causal")):
        pytest.skip("no SANA_WAM_CAUSAL sibling")
    sys.path.insert(0, causal_dir)
    try:
        from sana_wam_causal.contract import resolve_causal_contract
    finally:
        sys.path.remove(causal_dir)
    for path in (STRIDED, STRIDED_CACHED, FIXED, FIXED_CACHED):
        with pytest.raises(ValueError, match="SANA_WAM_SM"):
            resolve_causal_contract(_load(path))
    with pytest.raises(ValueError, match="SANA_WAM_HM"):
        resolve_causal_contract(_load(HM_CHECKPOINT))
    assert sm.MEMORY_KEYS == ("memory_type", "memory_frames", "memory_stride")
