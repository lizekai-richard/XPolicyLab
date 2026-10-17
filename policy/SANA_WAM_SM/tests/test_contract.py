"""The sparse-memory contract resolved from the rwm/sparse-memory recipes (343c60b4d: global m = 16, strided m16s16 and
m8s25; cc18ad78f: the mix-v3 strided m8s25 on the sana_pixel canvas and its sana_latent twin on the three views; the final
fixed-16 checkpoint's config.yaml, memory_type global; the 3fa3e9689 fixed m = 16 recipes, whose memory_type is the
global memory's former name), the refusals (the unbounded strided memory before eef7beeab, full
sequence attention, a hierarchical-memory checkpoint among them), and the shared front-end's reading of the recipes'
bidirectional view."""

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

GLOBAL = os.path.join(_paths.FIXTURES, "recipe_sm_global_343c60b4d.yaml")
STRIDED = os.path.join(_paths.FIXTURES, "recipe_sm_strided_m16s16_343c60b4d.yaml")
STRIDED_M8S25 = os.path.join(_paths.FIXTURES, "recipe_sm_strided_m8s25_343c60b4d.yaml")
FULL_ATTENTION = os.path.join(_paths.FIXTURES, "recipe_sm_global_full_attention_343c60b4d.yaml")
# the mix-v3 m8s25 pair at cc18ad78f: the sana_pixel run (cancelled at step 25,970) and the sana_latent run replacing it
PIXEL_M8S25 = os.path.join(_paths.FIXTURES, "recipe_sm_sana_pixel_strided_m8s25_cc18ad78f.yaml")
LATENT_M8S25 = os.path.join(_paths.FIXTURES, "recipe_sm_sana_latent_strided_m8s25_cc18ad78f.yaml")
LATENT_M8S25_CACHED = os.path.join(_paths.FIXTURES, "recipe_sm_sana_latent_strided_m8s25_cached_cc18ad78f.yaml")
# the final fixed-16 checkpoint (zekaili/..._memory_fixed16 @ 1a31e590, epoch_14_step_50000): config.yaml with memory_type global
CHECKPOINT_50K = os.path.join(_paths.FIXTURES, "config_sm_global_m16_s50000.yaml")
# before Sana eef7beeab: the global memory was named fixed (the same rows), and strided was unbounded (no memory_frames)
FIXED = os.path.join(_paths.FIXTURES, "recipe_sm_fixed_m16_3fa3e9689.yaml")
FIXED_CACHED = os.path.join(_paths.FIXTURES, "recipe_sm_fixed_m16_cached_3fa3e9689.yaml")
LEGACY_STRIDED = os.path.join(_paths.FIXTURES, "recipe_sm_strided_s25_3fa3e9689.yaml")
LEGACY_STRIDED_CACHED = os.path.join(_paths.FIXTURES, "recipe_sm_strided_s25_cached_3fa3e9689.yaml")
# a hierarchical-memory checkpoint's frozen config (hm_uniform_noobs epoch_3_step_7250, Sana 6564adc38)
HM_CHECKPOINT = os.path.join(_paths.FIXTURES, "config_hm_uniform_noobs_s7250.yaml")


def _load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def _policy(cfg):
    return cfg["model"]["extra"]["chunk_causal_policy"]


M8S25 = {"memory_type": "strided", "memory_frames": 8, "memory_stride": 25}
CANVAS = ("sana_pixel", "sana_pixel_canvas")
STRIP = ("sana_latent", "three_view_strip")


@pytest.mark.parametrize(
    "path,memory,trained,layout",
    [
        (GLOBAL, {"memory_type": "global", "memory_frames": 16}, 2, CANVAS),
        (STRIDED, {"memory_type": "strided", "memory_frames": 16, "memory_stride": 16}, 2, CANVAS),
        (STRIDED_M8S25, M8S25, 1, CANVAS),
        (PIXEL_M8S25, M8S25, 1, CANVAS),
        (LATENT_M8S25, M8S25, 1, STRIP),
        (LATENT_M8S25_CACHED, M8S25, 1, STRIP),
        (CHECKPOINT_50K, {"memory_type": "global", "memory_frames": 16}, 2, CANVAS),
        (FIXED, {"memory_type": "global", "memory_frames": 16}, 2, CANVAS),
        (FIXED_CACHED, {"memory_type": "global", "memory_frames": 16}, 2, CANVAS),
    ],
    ids=["global", "strided-m16s16", "strided-m8s25", "pixel-m8s25-mixv3", "latent-m8s25", "latent-m8s25-cached",
         "checkpoint-50k", "legacy-fixed", "legacy-fixed-cached"],
)
def test_the_recipes_resolve(path, memory, trained, layout):
    cfg = _load(path)
    assert is_sparse_memory_config(cfg)
    contract = resolve_sm_contract(cfg)
    assert contract.memory == memory and contract.trained_memory_latents == trained
    assert (contract.actions_per_chunk, contract.video_frame_stride, contract.tier_num_frames) == (32, 4, 33)
    assert (contract.multiview, contract.visual_layout, contract.rope) == (*layout, "aligned")
    assert contract.resolve_deploy_form(None) == DEPLOY_CACHED and contract.resolve_deploy_form(" Uncached ") == DEPLOY_UNCACHED
    with pytest.raises(ValueError, match="deploy_form"):
        contract.resolve_deploy_form("streaming")
    size = f"memory_frames={memory['memory_frames']}" + (f" memory_stride={memory['memory_stride']}" if "memory_stride" in memory else "")
    assert contract.describe().startswith(f"protocol=sparse_memory memory_type={memory['memory_type']} {size} ")
    assert f" trained_memory_latents={trained} " in contract.describe()


def test_the_latent_store_twins_are_their_recipes():
    """The cached recipes differ by the data-side latent store alone: the same contract and network."""

    for plain, cached in ((FIXED, FIXED_CACHED), (LATENT_M8S25, LATENT_M8S25_CACHED)):
        assert resolve_sm_contract(_load(plain)) == resolve_sm_contract(_load(cached))
        assert policy_config_from_train_config(bidirectional_view(_load(plain))) == policy_config_from_train_config(
            bidirectional_view(_load(cached))
        )


def test_the_bidirectional_view_restores_the_sft_tier():
    cfg = _load(LEGACY_STRIDED)
    view = bidirectional_view(cfg)
    assert view["model"]["model"] == THREE_VIEW_POLICY_FACTORY
    assert view["data"]["num_frames"] == 33 and view["data"]["multi_fps"] == {"25": [33]}
    assert cfg["data"]["multi_fps"] == {"25": [33, 65, 97, 129, 161, 193, 225, 257, 289]}   # the source is left untouched
    config = policy_config_from_train_config(view)
    assert (config.depth, config.hidden_size, tuple(config.softmax_layer_indices)) == (32, 2560, (3, 7, 11, 15, 19, 23, 27, 31))
    assert config.rope == "aligned" and config.separate_action_schedule
    options = sft_options_from_train_config(view)
    assert options["normalization_sha256"].startswith("1fe3b7e7") and options["tier_num_frames"] == 33
    # the other recipes train from another donor at another lr (and the 50k checkpoint past its epochs), the sana_latent
    # one on the three views: the same network, the strip's semantic 2x2 tiles the (8, 16) of every robot_sft config
    for path in (FIXED, GLOBAL, STRIDED, STRIDED_M8S25, CHECKPOINT_50K, PIXEL_M8S25, LATENT_M8S25):
        assert policy_config_from_train_config(bidirectional_view(_load(path))) == config
    assert config.multiview_spatial_rope_layout == "semantic_2x2" and tuple(config.multiview_spatial_rope_tile_shape) == (8, 16)


def test_the_strip_ignores_the_canvas_keys():
    """The canvas-only checks (the 320 x 512 bucket, sana_pixel_pad) do not reach the sana_latent views, and the strip's
    aspect bucket is the robot table's."""

    cfg = _load(LATENT_M8S25)
    assert cfg["data"]["aspect_ratio_type"] == "ASPECT_RATIO_VIDEO_320_ROBOT" and cfg["model"]["image_size"] == 320
    assert resolve_sm_contract(cfg).visual_layout == "three_view_strip"
    edited = copy.deepcopy(cfg)
    edited["data"]["extra"]["multiview"] = "sana_pixel"
    with pytest.raises((ValueError, NotImplementedError), match="sana_pixel needs data.aspect_ratio_type"):
        resolve_sm_contract(edited)


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
        (lambda c: _policy(c).__setitem__("memory_type", "uniform"), "global or strided"),
        (lambda c: _policy(c).__setitem__("memory_stride", 25), "belongs to a strided memory"),
        (lambda c: _policy(c).__setitem__("sequence_attention", "dense"), "causal or full"),
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
        "frames-without-type", "no-memory", "memory-type", "global-with-stride", "sequence-attention", "frames-not-x8", "frames-off-the-ladder", "hierarchical",
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
    with pytest.raises(ValueError, match="positive chunk_causal_policy.memory_stride"):
        resolve_sm_contract(cfg)


@pytest.mark.parametrize("path", [LEGACY_STRIDED, LEGACY_STRIDED_CACHED], ids=["plain", "cached"])
def test_the_unbounded_strided_memory_is_refused(path):
    """The strided memory before Sana eef7beeab (every s rows back to the sink, no memory_frames) is retired upstream; its
    configs would otherwise read as the bounded strided memory with a missing frame count."""

    with pytest.raises(ValueError, match="unbounded strided memory"):
        resolve_sm_contract(_load(path))


def test_full_sequence_attention_is_refused():
    """sequence_attention full (Sana b76502831) runs the whole sequence as one chunk: no mirror here, and the causal forms
    would serve it silently."""

    cfg = _load(FULL_ATTENTION)
    assert _policy(cfg)["sequence_attention"] == "full"
    with pytest.raises(NotImplementedError, match="sequence_attention: full"):
        resolve_sm_contract(cfg)
    reference = resolve_sm_contract(_load(GLOBAL))
    _policy(cfg)["sequence_attention"] = "causal"
    assert resolve_sm_contract(cfg) == reference


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
    for path in (GLOBAL, STRIDED, STRIDED_M8S25, PIXEL_M8S25, LATENT_M8S25, CHECKPOINT_50K, FIXED, FIXED_CACHED,
                 LEGACY_STRIDED, LEGACY_STRIDED_CACHED):
        with pytest.raises(ValueError, match="SANA_WAM_SM"):
            resolve_causal_contract(_load(path))
    with pytest.raises(ValueError, match="SANA_WAM_HM"):
        resolve_causal_contract(_load(HM_CHECKPOINT))
    assert sm.MEMORY_KEYS == ("memory_type", "memory_frames", "memory_stride")


def test_the_action_mode_sentence_can_be_the_checkpoints_own():
    """Sana db382ee90 changed the absolute-EEF sentence for every later run with no config key to tell: deploy.yml's
    action_mode_text puts the checkpoint's own sentence (its README) on the front-end and drops the cached prompts."""

    from collections import OrderedDict
    from types import SimpleNamespace

    from sana_wam_sm.runtime import apply_action_mode_text

    table = "robot_base_eef; end-effector targets are absolute future poses in the robot base frame, and gripper targets are absolute future targets"
    frontend = SimpleNamespace(action_mode_text=table, _prompt_cache=OrderedDict(old=1))
    assert apply_action_mode_text(frontend, None) == "table" and frontend.action_mode_text == table
    assert frontend._prompt_cache
    short = "robot_base_eef; end-effector targets are absolute future poses expressed in the robot base frame"
    assert apply_action_mode_text(frontend, f"  {short}. ") == "deploy"
    assert frontend.action_mode_text == short and not frontend._prompt_cache
    with pytest.raises(ValueError, match="empty"):
        apply_action_mode_text(frontend, " . ")
