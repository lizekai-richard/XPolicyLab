"""The hierarchical-memory contract resolved from the two rwm/sparse-memory recipes (e8dde6b56), their noobs twin
(``chunk_observation: false``, 6564adc38) and the pre-rename memory_uniform recipe (5736eebad), the refusals, and the
shared front-end's reading of the recipes' bidirectional view."""

from __future__ import annotations

import copy
import dataclasses
import os

import pytest
import yaml

import _paths

from sana_wam_hm import memory as hm
from sana_wam_hm.contract import (
    DEPLOY_FULL,
    DEPLOY_STREAMING,
    LEGACY_MEMORY_SPLIT,
    LEGACY_MEMORY_UNIFORM,
    bidirectional_view,
    is_hierarchical_memory_config,
    resolve_hm_contract,
)
from sana_wam_min.config import THREE_VIEW_POLICY_FACTORY, policy_config_from_train_config, sft_options_from_train_config

DENSE_SPARSE = os.path.join(_paths.FIXTURES, "recipe_hm_dense_sparse_videoonly_e8dde6b56.yaml")
SPARSE_SPARSE = os.path.join(_paths.FIXTURES, "recipe_hm_sparse_sparse_videoonly_e8dde6b56.yaml")
# the same recipe before Sana 3e7435754 renamed memory_uniform: {sparse_frames: 16} to hierarchical_memory
MEMORY_UNIFORM = os.path.join(_paths.FIXTURES, "recipe_memory_uniform_videoonly_5736eebad.yaml")
# sparse_sparse + chunk_causal_policy.chunk_observation: false (the noisy chunk is the target alone)
NOOBS = os.path.join(_paths.FIXTURES, "recipe_hm_sparse_sparse_videoonly_noobs_6564adc38.yaml")
# the trainer dump of that recipe in the released hm_uniform_noobs checkpoint (epoch_3_step_7250)
NOOBS_CHECKPOINT = os.path.join(_paths.FIXTURES, "config_hm_uniform_noobs_s7250.yaml")


def _load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def _policy(cfg):
    return cfg["model"]["extra"]["chunk_causal_policy"]


@pytest.mark.parametrize(
    "path,gdn,observed",
    [
        (DENSE_SPARSE, "dense", True),
        (SPARSE_SPARSE, "sparse", True),
        (NOOBS, "sparse", False),
        (NOOBS_CHECKPOINT, "sparse", False),
    ],
    ids=["dense_sparse", "sparse_sparse", "sparse_sparse_noobs", "hm_uniform_noobs_checkpoint"],
)
def test_the_recipes_resolve(path, gdn, observed):
    cfg = _load(path)
    assert is_hierarchical_memory_config(cfg)
    contract = resolve_hm_contract(cfg)
    assert (contract.gdn_memory, contract.softmax_memory, contract.sparse_frames) == (gdn, "sparse", 16)
    assert contract.chunk_observation is observed and contract.memory[hm.CHUNK_OBSERVATION] is observed
    assert f" chunk_observation={observed} " in contract.describe()
    assert (contract.actions_per_chunk, contract.video_frame_stride, contract.tier_num_frames) == (32, 4, 33)
    assert (contract.multiview, contract.visual_layout, contract.rope) == ("sana_pixel", "sana_pixel_canvas", "aligned")
    assert contract.default_deploy_form() == DEPLOY_STREAMING and contract.resolve_deploy_form(None) == DEPLOY_STREAMING
    assert contract.resolve_deploy_form("full") == DEPLOY_FULL and contract.resolve_deploy_form(" Streaming ") == DEPLOY_STREAMING
    assert contract.describe().startswith("protocol=hierarchical_memory ")
    with pytest.raises(ValueError, match="deploy_form"):
        contract.resolve_deploy_form("sometimes")


def test_the_pre_rename_memory_uniform_recipe_is_sparse_sparse():
    """memory_uniform (the pin of the memory_uniform checkpoints) resolves to the contract of its renamed twin
    hm_sparse_sparse, the key it was read from aside, and builds the same network."""

    legacy, renamed = _load(MEMORY_UNIFORM), _load(SPARSE_SPARSE)
    assert hm.MEMORY_KEY not in _policy(legacy) and is_hierarchical_memory_config(legacy)
    contract, twin = resolve_hm_contract(legacy), resolve_hm_contract(renamed)
    assert (contract.memory_key, twin.memory_key) == (LEGACY_MEMORY_UNIFORM, hm.MEMORY_KEY)
    assert dataclasses.replace(contract, memory_key=hm.MEMORY_KEY) == twin
    assert contract.streams == (hm.SPARSE,)
    assert " gdn=sparse softmax=sparse sparse_frames=16 memory_key=memory_uniform " in contract.describe()
    assert policy_config_from_train_config(bidirectional_view(legacy)) == policy_config_from_train_config(
        bidirectional_view(renamed)
    )


def test_the_noobs_recipe_is_sparse_sparse_without_the_observation():
    """The _noobs recipe differs from hm_sparse_sparse by chunk_observation: false alone: the same network and memory,
    the noisy chunk the target alone (Sana 6564adc38: [A_0..A_2 | target] + 33 rows, 673 tokens)."""

    noobs, observed = _load(NOOBS), _load(SPARSE_SPARSE)
    assert _policy(noobs)[hm.CHUNK_OBSERVATION] is False and hm.CHUNK_OBSERVATION not in _policy(observed)
    contract, twin = resolve_hm_contract(noobs), resolve_hm_contract(observed)
    assert dataclasses.replace(contract, chunk_observation=True) == twin
    assert contract.streams == (hm.SPARSE,) and contract.memory_key == hm.MEMORY_KEY
    assert hm.memory_layout(hm.FORM_TRAIN, 0, contract.memory)["frames"] == 4
    assert hm.stream_layout(hm.FORM_NOISY, contract.memory)["frames"] == 1
    assert policy_config_from_train_config(bidirectional_view(noobs)) == policy_config_from_train_config(
        bidirectional_view(observed)
    )
    # chunk_observation: true is the absent key
    _policy(observed)[hm.CHUNK_OBSERVATION] = True
    assert resolve_hm_contract(observed) == twin


def test_the_released_noobs_checkpoint_is_its_recipe():
    """The hm_uniform_noobs checkpoint's frozen config (the 6564adc38 trainer dump) resolves to the recipe's contract and
    builds the same network, on the normalization pin 1fe3b7e7."""

    released, recipe = _load(NOOBS_CHECKPOINT), _load(NOOBS)
    assert resolve_hm_contract(released) == resolve_hm_contract(recipe)
    assert policy_config_from_train_config(bidirectional_view(released)) == policy_config_from_train_config(
        bidirectional_view(recipe)
    )
    options = sft_options_from_train_config(bidirectional_view(released))
    assert options["normalization_sha256"].startswith("1fe3b7e7") and options["tier_num_frames"] == 33


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda p: p.__setitem__(hm.CHUNK_OBSERVATION, "false"), "must be true or false"),
        (lambda p: p.__setitem__(hm.CHUNK_OBSERVATION, 0), "must be true or false"),
        (lambda p: p[hm.MEMORY_KEY].__setitem__(hm.CHUNK_OBSERVATION, p.pop(hm.CHUNK_OBSERVATION)),
         "sits in chunk_causal_policy"),
    ],
    ids=["string", "int", "inside-the-block"],
)
def test_a_misdeclared_chunk_observation_is_refused(edit, match):
    cfg = copy.deepcopy(_load(NOOBS))
    edit(_policy(cfg))
    with pytest.raises(ValueError, match=match):
        resolve_hm_contract(cfg)


def test_a_pre_rename_memory_uniform_trained_with_the_observation():
    """No hierarchical-memory trainer before 6564adc38 read chunk_observation: a memory_uniform run's chunk_observation:
    false would claim a layout it never trained; true is what it trained."""

    cfg = copy.deepcopy(_load(MEMORY_UNIFORM))
    _policy(cfg)[hm.CHUNK_OBSERVATION] = True
    assert resolve_hm_contract(cfg).chunk_observation is True
    for value in (False, "false"):
        _policy(cfg)[hm.CHUNK_OBSERVATION] = value
        with pytest.raises(ValueError, match="6564adc38"):
            resolve_hm_contract(cfg)


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda p: p.__setitem__(hm.MEMORY_KEY, {"gdn": "sparse", "softmax": "sparse", "sparse_frames": 16}),
         "one memory design"),
        (lambda p: p.__setitem__(LEGACY_MEMORY_SPLIT, p.pop(LEGACY_MEMORY_UNIFORM)), "memory_split"),
        (lambda p: p[LEGACY_MEMORY_UNIFORM].__setitem__("gdn", "dense"), r"exactly \['sparse_frames'\]"),
        (lambda p: p[LEGACY_MEMORY_UNIFORM].__setitem__("sparse_frames", 12), "multiple of 8"),
    ],
    ids=["both-keys", "memory-split", "uniform-extra-key", "uniform-sparse-frames"],
)
def test_unsupported_pre_rename_memories_are_refused(edit, match):
    cfg = copy.deepcopy(_load(MEMORY_UNIFORM))
    edit(_policy(cfg))
    with pytest.raises((ValueError, NotImplementedError), match=match):
        resolve_hm_contract(cfg)


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
    for path in (DENSE_SPARSE, SPARSE_SPARSE, MEMORY_UNIFORM, NOOBS, NOOBS_CHECKPOINT):
        with pytest.raises(ValueError, match="SANA_WAM_HM"):
            resolve_causal_contract(_load(path))
