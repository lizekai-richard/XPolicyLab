"""The hierarchical-memory contract resolved from the two rwm/sparse-memory recipes (e8dde6b56), the refusals, and the
shared front-end's reading of the recipes' bidirectional view."""

from __future__ import annotations

import copy
import os

import pytest
import yaml

import _paths

from sana_wam_hm import memory as hm
from sana_wam_hm.contract import (
    DEPLOY_FULL,
    DEPLOY_STREAMING,
    bidirectional_view,
    is_hierarchical_memory_config,
    resolve_hm_contract,
)
from sana_wam_min.config import THREE_VIEW_POLICY_FACTORY, policy_config_from_train_config, sft_options_from_train_config

DENSE_SPARSE = os.path.join(_paths.FIXTURES, "recipe_hm_dense_sparse_videoonly_e8dde6b56.yaml")
SPARSE_SPARSE = os.path.join(_paths.FIXTURES, "recipe_hm_sparse_sparse_videoonly_e8dde6b56.yaml")


def _load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def _policy(cfg):
    return cfg["model"]["extra"]["chunk_causal_policy"]


@pytest.mark.parametrize(
    "path,gdn",
    [(DENSE_SPARSE, "dense"), (SPARSE_SPARSE, "sparse")],
    ids=["dense_sparse", "sparse_sparse"],
)
def test_the_recipes_resolve(path, gdn):
    cfg = _load(path)
    assert is_hierarchical_memory_config(cfg)
    contract = resolve_hm_contract(cfg)
    assert (contract.gdn_memory, contract.softmax_memory, contract.sparse_frames) == (gdn, "sparse", 16)
    assert (contract.actions_per_chunk, contract.video_frame_stride, contract.tier_num_frames) == (32, 4, 33)
    assert (contract.multiview, contract.visual_layout, contract.rope) == ("sana_pixel", "sana_pixel_canvas", "aligned")
    assert contract.default_deploy_form() == DEPLOY_STREAMING and contract.resolve_deploy_form(None) == DEPLOY_STREAMING
    assert contract.resolve_deploy_form("full") == DEPLOY_FULL and contract.resolve_deploy_form(" Streaming ") == DEPLOY_STREAMING
    assert contract.describe().startswith("protocol=hierarchical_memory ")
    with pytest.raises(ValueError, match="deploy_form"):
        contract.resolve_deploy_form("sometimes")


def test_the_bidirectional_view_restores_the_sft_tier():
    cfg = _load(DENSE_SPARSE)
    view = bidirectional_view(cfg)
    assert view["model"]["model"] == THREE_VIEW_POLICY_FACTORY
    assert view["data"]["num_frames"] == 33 and view["data"]["multi_fps"] == {"25": [33]}
    assert cfg["data"]["num_frames"] == 1537           # the source is left untouched
    config = policy_config_from_train_config(view)
    assert (config.depth, config.hidden_size, tuple(config.softmax_layer_indices)) == (32, 2560, (3, 7, 11, 15, 19, 23, 27, 31))
    assert config.rope == "aligned" and config.separate_action_schedule
    options = sft_options_from_train_config(view)
    assert options["normalization_sha256"].startswith("1fe3b7e7") and options["tier_num_frames"] == 33


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda c: _policy(c).pop(hm.MEMORY_KEY), "per-chunk observation protocol"),
        (lambda c: _policy(c).__setitem__("memory_content", "video_and_action"), "video_only only"),
        (lambda c: _policy(c).pop("memory_content"), "video_only only"),
        (lambda c: _policy(c).__setitem__("sliding_window_chunks", 24), "no softmax window"),
        (lambda c: _policy(c).__setitem__("obs_in_first_chunk", True), "retired layout"),
        (lambda c: _policy(c)[hm.MEMORY_KEY].__setitem__("gdn", "uniform"), "reads one of"),
        (lambda c: _policy(c)[hm.MEMORY_KEY].__setitem__("sparse_frames", 12), "multiple of 8"),
        (lambda c: _policy(c).__setitem__("actions_per_chunk", {"25": 64}), "disagree|32-row chunk"),
        (lambda c: c["model"].__setitem__("model", THREE_VIEW_POLICY_FACTORY), "chunk-causal policy class"),
        (lambda c: c["data"].__setitem__("type", "RoboDojoSFTDataset"), "causal dataset"),
        (lambda c: c["model"]["extra"].__setitem__("rope", "independent"), "aligned"),
        (lambda c: c["model"].__setitem__("use_time_conditioning", True), "use_time_conditioning"),
        (lambda c: c["model"]["extra"].__setitem__("sana_pixel_pad", "masked"), "sana_pixel_pad"),
        (lambda c: c["data"].__setitem__("aspect_ratio_type", "ASPECT_RATIO_SANA_PIXEL_2X2_320_480"), "320 x 512"),
        (lambda c: c["vae"].__setitem__("use_causal_encode", False), "use_causal_encode"),
    ],
    ids=[
        "no-memory", "video-and-action", "no-memory-content", "softmax-window", "obs-key", "memory-kind",
        "sparse-frames", "chunk", "sft-class", "sft-dataset", "independent", "time-conditioning", "padmask",
        "canvas-320x480", "non-causal-vae",
    ],
)
def test_unsupported_contracts_are_refused(edit, match):
    cfg = copy.deepcopy(_load(DENSE_SPARSE))
    edit(cfg)
    with pytest.raises((ValueError, NotImplementedError, KeyError), match=match):
        resolve_hm_contract(cfg)


def test_the_causal_adapter_refuses_a_hierarchical_memory_checkpoint():
    """SANA_WAM_CAUSAL must not serve an HM checkpoint as the per-chunk observation protocol (it used to accept it)."""

    import sys

    causal_dir = os.path.abspath(os.path.join(_paths.ADAPTER_DIR, "..", "SANA_WAM_CAUSAL"))
    if not os.path.isdir(os.path.join(causal_dir, "sana_wam_causal")):
        pytest.skip("no SANA_WAM_CAUSAL sibling")
    sys.path.insert(0, causal_dir)
    try:
        from sana_wam_causal.contract import resolve_causal_contract
    finally:
        sys.path.remove(causal_dir)
    for path in (DENSE_SPARSE, SPARSE_SPARSE):
        with pytest.raises(ValueError, match="SANA_WAM_HM"):
            resolve_causal_contract(_load(path))
