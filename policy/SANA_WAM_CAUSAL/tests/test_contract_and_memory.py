"""Contract resolution from a causal training yaml and the session memory's softmax read rule, episode protocol (CPU,
no Sana)."""

from __future__ import annotations

import copy
import hashlib
import json
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
    SINK_HORIZON_CHUNKS,
    SINK_HORIZON_TREES,
    WINDOW_START_SINK_TREES,
    check_observation_form,
    check_run_record,
    describe_run_records,
    load_run_records,
    resolve_causal_contract,
    resolve_rope_clock,
)
from sana_wam_causal.checkpoint import load_causal_state_dict
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
# the episode VO run at step 1000 (trained WITH chunk observations on the padded window; its config.yaml declares
# neither key) and the run records of two Hub uploads (RUN_ID.json, trimmed to identity and protocol): that VO
# checkpoint and the released no-observation one (FIRST_FRAME_NOOBS)
OBSERVATION_S1000 = os.path.join(_paths.FIXTURES, "config_causal_pantheon33k_episode_n24_videoonly_s1000.yaml")
# the released mix-v3 dense sana_latent N1-obs checkpoint (step 15000): causal_observation + causal_state, numerator-only GDN
N1_OBS_S15000 = os.path.join(_paths.FIXTURES, "config_causal_mixv3_dense14k_sana_latent_episode_n1_obs_videoonly_s15000.yaml")
# the physical-RoPE mix-v4 capv3 sana_latent N1-obs+state checkpoint (step 10000) and its run record (deploy block)
PHYSROPE_S10000 = os.path.join(_paths.FIXTURES, "config_causal_mixv4_capv3_sana_latent_physrope_episode_n1_obs_state_s10000.yaml")
RECORD_PHYSROPE_S10000 = os.path.join(_paths.FIXTURES, "run_record_physrope_n1_obs_state_s10000.json")
OBSBLOCK_PIXEL_S15000 = os.path.join(
    _paths.FIXTURES, "config_causal_mixv4_capv3_obsblock_sana_pixel_physrope_episode_n1_obs_state_s15000.yaml"
)
RECORD_OBSBLOCK_PIXEL_S15000 = os.path.join(_paths.FIXTURES, "run_record_obsblock_sana_pixel_physrope_n1_obs_state_s15000.json")
# the sana_latent obs-block N1-obs+state checkpoint trained without AttnRes (step 10000, Sana a5ae8cf8f); its Hub repo ships
# no run record, the local one names the README's training tree
NOATTNRES_S10000 = os.path.join(
    _paths.FIXTURES, "config_causal_mixv4_capv3_obsblock_sana_latent_physrope_n1_obs_state_noattnres_s10000.yaml"
)
RECORD_LOCAL_NOATTNRES_S10000 = os.path.join(_paths.FIXTURES, "run_record_local_obsblock_sana_latent_noattnres_s10000.json")
# the sana_latent no-AttnRes N1 recipe of Sana 7ed0d28bf, on the 0c88c11ab clock (the physical clock only, the softmax
# sink at chunk max(0, c - 7)): it no longer declares physical_rope
RECIPE_HORIZON_NOATTNRES = os.path.join(
    _paths.FIXTURES, "recipe_causal_mixv4_capv3_obsblock_sana_latent_noattnres_episode_n1_obs_state_7ed0d28bf.yaml"
)
RECORD_S1000 = os.path.join(_paths.FIXTURES, "run_record_episode_n24_videoonly_s1000.json")
RECORD_NOOBS_S3324 = os.path.join(_paths.FIXTURES, "run_record_episode_n24_videoonly_noobs_s3324.json")
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


def _record(path):
    with open(path) as handle:
        return json.load(handle)


def test_a_run_record_of_chunk_observations_is_refused():
    """The VO run's config resolves on its own (the pre-2026-10-14 ambiguity: no key, read as the single form); its run
    record names chunk observations and refuses it, pointing at the adapter that serves it."""

    contract = resolve_causal_contract(_load(OBSERVATION_S1000))
    assert (contract.aligned_phase_window, contract.memory_content) == (ALIGNED_PHASE_PADDED, MEMORY_VIDEO_ONLY)
    with pytest.raises(ValueError, match=r"RUN_ID.json records chunk_observation=True .*215385e9c.*d57a617"):
        check_run_record(_record(RECORD_S1000), contract, source="RUN_ID.json")


def test_the_released_no_observation_record_is_accepted():
    check_run_record(_record(RECORD_NOOBS_S3324), resolve_causal_contract(_load(FIRST_FRAME_NOOBS)))


@pytest.mark.parametrize("value,expected", [(None, True), (True, True), (False, False)], ids=["absent", "true", "false"])
def test_the_gdn_output_normalizer_switch_resolves(value, expected):
    """Sana c414e9b32: model.memory_gdn_output_normalizer, absent = true (every config before the key divides); the
    ready line names it."""

    cfg = copy.deepcopy(_load(RECIPE_PIXEL_N2))
    assert "memory_gdn_output_normalizer" not in cfg["model"]
    if value is not None:
        cfg["model"]["memory_gdn_output_normalizer"] = value
    contract = resolve_causal_contract(cfg)
    assert contract.gdn_output_normalizer is expected
    assert contract.describe().endswith(f"first_chunk=0 gdn_output_normalizer={expected} use_attn_res=True sink_horizon=None")


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_a_gdn_output_normalizer_that_is_not_a_boolean_is_refused(value):
    """Sana's bool() would serve the string "false" as true: only a yaml boolean is accepted."""

    cfg = copy.deepcopy(_load(RECIPE_PIXEL_N2))
    cfg["model"]["memory_gdn_output_normalizer"] = value
    with pytest.raises(ValueError, match="memory_gdn_output_normalizer must be true or false"):
        resolve_causal_contract(cfg)


def test_the_released_n1_obs_checkpoint_resolves_with_its_observation_and_state():
    """The real config.yaml of the mix-v3 dense sana_latent N1-obs checkpoint (step 15000): every noisy chunk reads its
    own observation latent with its state row (Sana 764ee2113 / afc9874b6), the GDN numerator alone (c414e9b32); its
    training-only speed switches (gdn_fused_two_stream, attn_res_fast_attend, attn_res_compiled_sum) change nothing."""

    contract = resolve_causal_contract(_load(N1_OBS_S15000))
    assert (contract.causal_observation, contract.causal_state, contract.gdn_output_normalizer) == (True, True, False)
    assert (contract.sliding_window_chunks, contract.memory_content, contract.aligned_phase_window) == (
        1, MEMORY_VIDEO_ONLY, ALIGNED_PHASE_FIRST_FRAME,
    )
    assert (contract.multiview, contract.visual_layout) == ("sana_latent", "three_view_strip")
    text = contract.describe()
    assert " chunk_observation=False causal_observation=True causal_state=True multiview=sana_latent " in text
    assert text.endswith("first_chunk=0 gdn_output_normalizer=False use_attn_res=True sink_horizon=None")


@pytest.mark.parametrize(
    "observation,state,expected",
    [(None, None, (False, False)), (True, None, (True, False)), (True, True, (True, True)), (False, False, (False, False))],
    ids=["absent", "observation", "observation+state", "both-false"],
)
def test_the_causal_observation_switches_resolve(observation, state, expected):
    cfg = copy.deepcopy(_load(RECIPE_PIXEL_N2))
    for key, value in (("causal_observation", observation), ("causal_state", state)):
        if value is not None:
            _policy(cfg)[key] = value
    contract = resolve_causal_contract(cfg)
    assert (contract.causal_observation, contract.causal_state) == expected
    assert f"causal_observation={expected[0]} causal_state={expected[1]} " in contract.describe()


@pytest.mark.parametrize(
    "edit,match",
    [
        ({"causal_state": True}, "set causal_observation: true"),
        ({"causal_observation": "true"}, "causal_observation='true': must be true or false"),
        ({"causal_observation": True, "causal_state": 1}, "causal_state=1: must be true or false"),
        ({"physical_rope": "true"}, "physical_rope='true': must be true or false"),
    ],
    ids=["state-without-observation", "observation-string", "state-int", "physical-rope-string"],
)
def test_the_causal_observation_switches_are_refused_outside_their_contract(edit, match):
    cfg = copy.deepcopy(_load(RECIPE_PIXEL_N2))
    _policy(cfg).update(edit)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        resolve_causal_contract(cfg)


def test_the_physical_rope_checkpoint_resolves_and_its_record_is_accepted():
    """The real config.yaml of the physical-RoPE mix-v4 capv3 sana_latent N1-obs+state checkpoint (Sana 6e91eaf2a): the
    physical clock with the observation latent and its state row, numerator-only GDN, bf16 softmax attention; its run
    record's deploy block names the same switches."""

    contract = resolve_causal_contract(_load(PHYSROPE_S10000))
    assert (contract.physical_rope, contract.causal_observation, contract.causal_state) == (True, True, True)
    assert (contract.gdn_output_normalizer, contract.sliding_window_chunks, contract.memory_content) == (False, 1, MEMORY_VIDEO_ONLY)
    assert " rope=aligned physical_rope=True aligned_phase_window=first_frame " in contract.describe()
    check_run_record(_record(RECORD_PHYSROPE_S10000), contract, source="RUN_ID.json")
    plain = resolve_causal_contract(_load(N1_OBS_S15000))
    assert plain.physical_rope is False and " physical_rope=False " in plain.describe()


def test_the_obs_block_sana_pixel_checkpoint_is_served_as_the_branch_form(tmp_path):
    """The real config.yaml and RUN_ID.json of the sana_pixel obs-block N1 checkpoint (step 15000, trained on Sana
    c6d008749 / ff004392a, after fd5261ff6): the one-view canvas on the physical clock with the observation latent and
    its state row, numerator-only GDN; its record names no source_commit but its training segments, which make it the
    branch form; the ready line names the newest segment."""

    contract = resolve_causal_contract(_load(OBSBLOCK_PIXEL_S15000))
    assert (contract.physical_rope, contract.causal_observation, contract.causal_state) == (True, True, True)
    assert (contract.gdn_output_normalizer, contract.sliding_window_chunks, contract.memory_content) == (False, 1, MEMORY_VIDEO_ONLY)
    assert " multiview=sana_pixel visual_layout=sana_pixel_canvas " in contract.describe()
    record = _record(RECORD_OBSBLOCK_PIXEL_S15000)
    check_run_record(record, contract, source="RUN_ID.json")
    records = [(tmp_path / "RUN_ID.json", record)]
    assert check_observation_form(records, contract) == "branch(fd5261ff6; trained on ff004392a)"
    config = tmp_path / "config.yaml"
    config.write_bytes(open(OBSBLOCK_PIXEL_S15000, "rb").read())
    assert "step=15000 source_commit=ff004392a config_sha256=unrecorded)" in describe_run_records(records, config)


def test_a_checkpoint_of_the_joint_observation_read_is_refused(tmp_path):
    """No config key tells the GDN observation forms apart: the physical-RoPE sana_latent N1-obs+state record names
    6e91eaf2a .. 35b423b3b (before fd5261ff6), so it is refused with the adapter that serves it; an observation
    checkpoint without a record (the mix-v3 N1-obs s15000 directory) is refused unless it is a donor directory; a
    checkpoint without causal_observation is not checked."""

    contract = resolve_causal_contract(_load(PHYSROPE_S10000))
    records = [(tmp_path / "RUN_ID.json", _record(RECORD_PHYSROPE_S10000))]
    with pytest.raises(ValueError, match=r"before Sana fd5261ff6.*e35fcde"):
        check_observation_form(records, contract)
    plain = resolve_causal_contract(_load(N1_OBS_S15000))
    with pytest.raises(ValueError, match="name no known training tree"):
        check_observation_form([], plain)
    assert check_observation_form([], plain, allow_unrecorded=True).startswith("branch(unrecorded")
    mixed = [(tmp_path / "RUN_ID.json", {"scheme_history": [{"code": "6e91eaf2a"}, {"code": "c6d008749 (git archive)"}]})]
    assert check_observation_form(mixed, contract) == "branch(fd5261ff6; trained on c6d008749)"
    assert check_observation_form([], resolve_causal_contract(_load(FIRST_FRAME_NOOBS))) is None


@pytest.mark.parametrize("value,expected", [(None, True), (True, True), (False, False)], ids=["absent", "true", "false"])
def test_the_attn_res_switch_resolves_and_the_view_keeps_the_mirror_config(value, expected):
    """model.use_attn_res (Sana 53234167c on the causal policy): absent -> true, a yaml boolean otherwise; the contract
    carries it, and the bidirectional view pins it true so the shared policy config still builds."""

    cfg = copy.deepcopy(_load(OBSBLOCK_PIXEL_S15000))
    if value is None:
        cfg["model"].pop("use_attn_res", None)
    else:
        cfg["model"]["use_attn_res"] = value
    contract = resolve_causal_contract(cfg)
    assert contract.use_attn_res is expected and contract.describe().endswith(f" use_attn_res={expected} sink_horizon=None")
    view = bidirectional_view(cfg)
    assert view["model"]["use_attn_res"] is True and cfg["model"].get("use_attn_res", None) is value
    policy_config_from_train_config(view)


def test_the_sana_latent_checkpoint_without_attn_res_resolves_and_its_local_record_is_read(tmp_path):
    """The real config.yaml of the sana_latent obs-block N1-obs+state checkpoint trained without AttnRes (step 10000, Sana
    a5ae8cf8f): the plain residual stack on the physical clock with the observation latent and its state row,
    numerator-only GDN; RUN_ID.local.json (the README's training tree) is read next to the config and makes it the
    branch form; the model it builds has no AttnRes."""

    contract = resolve_causal_contract(_load(NOATTNRES_S10000))
    assert contract.use_attn_res is False
    assert (contract.physical_rope, contract.causal_observation, contract.causal_state) == (True, True, True)
    assert (contract.gdn_output_normalizer, contract.sliding_window_chunks, contract.memory_content) == (False, 1, MEMORY_VIDEO_ONLY)
    assert (contract.multiview, contract.visual_layout, contract.aligned_phase_window) == (
        "sana_latent", "three_view_strip", ALIGNED_PHASE_FIRST_FRAME)
    (tmp_path / "config.yaml").write_bytes(open(NOATTNRES_S10000, "rb").read())
    (tmp_path / "RUN_ID.local.json").write_bytes(open(RECORD_LOCAL_NOATTNRES_S10000, "rb").read())
    records = load_run_records(tmp_path)
    assert [path.name for path, _ in records] == ["RUN_ID.local.json"]
    for source, record in records:
        check_run_record(record, contract, source=str(source))
    assert resolve_rope_clock(records, contract) == (contract, "physical+sink_window_start(trained on a5ae8cf8f)")
    assert check_observation_form(records, contract) == "branch(fd5261ff6; trained on a5ae8cf8f)"
    assert "step=10000 source_commit=a5ae8cf8f" in describe_run_records(records, tmp_path / "config.yaml")
    view = bidirectional_view(_load(NOATTNRES_S10000))
    model = CausalPolicyModel(policy_config_from_train_config(view), contract)
    assert model.attn_res is None and not any(key.startswith("attn_res.") for key in model.state_dict())


def test_an_attn_res_switch_that_is_not_a_boolean_is_refused():
    cfg = copy.deepcopy(_load(OBSBLOCK_PIXEL_S15000))
    cfg["model"]["use_attn_res"] = "false"
    with pytest.raises(ValueError, match="use_attn_res must be true or false"):
        resolve_causal_contract(cfg)


def test_the_attn_res_form_of_the_weights_must_match_the_contract():
    """A mirror without AttnRes models no attn_res.* tensor: the strict load refuses an AttnRes checkpoint into it and
    a checkpoint without AttnRes into an AttnRes mirror (a config / weights mismatch never serves)."""

    with_attn_res, plain = _tiny_model(), _tiny_model(attn_res=False)
    assert plain.attn_res is None and not any(key.startswith("attn_res.") for key in plain.state_dict())
    load_causal_state_dict(plain, dict(_tiny_model(attn_res=False).state_dict()))
    with pytest.raises(RuntimeError, match="does not model.*attn_res"):
        load_causal_state_dict(plain, dict(with_attn_res.state_dict()))
    with pytest.raises(RuntimeError, match="misses policy tensors.*attn_res"):
        load_causal_state_dict(with_attn_res, dict(plain.state_dict()))


def test_the_trees_after_the_attn_res_switch_keep_the_window_start_sink(tmp_path):
    """53234167c .. 9a1b3e33d descend from fd5261ff6 (the observation branch) and read the softmax sink at the reader's
    window start, on the clock physical_rope declares: the contract is served as the config resolves it."""

    contract = resolve_causal_contract(_load(OBSBLOCK_PIXEL_S15000))
    for tree in ("53234167c", "a5ae8cf8f", "020582851", "9a1b3e33d"):
        records = [(tmp_path / "RUN_ID.json", {"scheme_history": [{"code": "ff004392a"}, {"code": tree}]})]
        assert check_observation_form(records, contract) == f"branch(fd5261ff6; trained on {tree})"
        assert resolve_rope_clock(records, contract) == (contract, f"physical+sink_window_start(trained on {tree})")
    slots = resolve_causal_contract(_load(FIRST_FRAME_NOOBS))
    records = [(tmp_path / "RUN_ID.json", {"source_commit": "b297028fe"})]
    assert resolve_rope_clock(records, slots) == (slots, "latent_slots+sink_window_start(trained on b297028fe)")
    assert not set(SINK_HORIZON_TREES) & set(WINDOW_START_SINK_TREES)


@pytest.mark.parametrize("tree", SINK_HORIZON_TREES)
def test_a_checkpoint_of_the_sink_horizon_trees_runs_the_physical_clock_with_the_sink_at_chunk_c_minus_7(tree, tmp_path):
    """From Sana 0c88c11ab on the physical clock is the only one and the recipes drop physical_rope (the 7ed0d28bf
    sana_latent no-AttnRes N1 recipe declares none, so its config alone resolves to the latent-slot clock): a run record
    naming such a tree serves the physical clock with the softmax sink at chunk max(0, c - 7), the GDN observation
    branch, and a record that names the clock is checked against the completed contract."""

    config = resolve_causal_contract(_load(RECIPE_HORIZON_NOATTNRES))
    assert (config.physical_rope, config.sink_horizon, config.use_attn_res) == (False, None, False)
    records = [(tmp_path / "RUN_ID.json", {"source_commit": f"{tree}0", "deploy": {"physical_rope": True}})]
    contract, label = resolve_rope_clock(records, config)
    assert (contract.physical_rope, contract.sink_horizon) == (True, SINK_HORIZON_CHUNKS) == (True, 7)
    assert contract == CausalContract(**{**config.__dict__, "physical_rope": True, "sink_horizon": 7})
    assert label == f"physical+sink_horizon7(0c88c11ab; trained on {tree}0)"
    assert contract.describe().endswith(" use_attn_res=False sink_horizon=7")
    check_run_record(records[0][1], contract)
    with pytest.raises(ValueError, match="records deploy.physical_rope=True"):
        check_run_record(records[0][1], config)
    assert check_observation_form(records, contract) == f"branch(fd5261ff6; trained on {tree}0)"
    # a run resumed onto the horizon tree: the newest segment's rule, the earlier one named
    resumed = [(tmp_path / "RUN_ID.json", {"scheme_history": [{"code": "9a1b3e33d"}, {"code": tree}]})]
    assert resolve_rope_clock(resumed, config)[1].endswith(f"trained on {tree}; earlier segments on 9a1b3e33d)")
    view = bidirectional_view(_load(RECIPE_HORIZON_NOATTNRES))
    model = CausalPolicyModel(policy_config_from_train_config(view), contract)
    assert (model.physical_rope, model.sink_horizon, model.target_frame_offset()) == (True, 7, 0.5625)


@pytest.mark.parametrize("window,ok", [(None, False), (1, True), (7, True), (8, False)])
def test_a_sink_horizon_checkpoint_needs_a_softmax_window_within_the_horizon(window, ok, tmp_path):
    """Sana refuses sliding_window_chunks above the sink horizon, and a reader past it without a window."""

    cfg = copy.deepcopy(_load(RECIPE_HORIZON_NOATTNRES))
    _policy(cfg)["sliding_window_chunks"] = window
    records = [(tmp_path / "RUN_ID.json", {"source_commit": "7ed0d28bf"})]
    if ok:
        assert resolve_rope_clock(records, resolve_causal_contract(cfg))[0].sliding_window_chunks == window
    else:
        with pytest.raises(ValueError, match=f"sliding_window_chunks={window} on a checkpoint trained on Sana 7ed0d28bf"):
            resolve_rope_clock(records, resolve_causal_contract(cfg))
    view = bidirectional_view(cfg)
    contract = CausalContract(**{**resolve_causal_contract(cfg).__dict__, "physical_rope": True, "sink_horizon": 7})
    if not ok:
        with pytest.raises(ValueError, match="softmax window of at most 7 chunks"):
            CausalPolicyModel(policy_config_from_train_config(view), contract)


def test_a_checkpoint_whose_records_name_no_known_tree_is_refused(tmp_path):
    """No config key tells the sink rules apart: records naming only unknown trees are refused, and so is a checkpoint
    without a record unless it is a donor directory (served on the window-start rule); a donor directory whose record
    names an unknown tree is refused too."""

    contract = resolve_causal_contract(_load(RECIPE_HORIZON_NOATTNRES))
    unknown = [(tmp_path / "RUN_ID.json", {"source_commit": "0123abcde", "scheme_history": [{"code": "fedcba987"}]})]
    for records, donor in ((unknown, False), (unknown, True), ([], False)):
        with pytest.raises(ValueError, match="name no known training tree.*SINK_HORIZON_TREES.*RUN_ID.local.json"):
            resolve_rope_clock(records, contract, allow_unrecorded=donor)
    assert resolve_rope_clock([], contract, allow_unrecorded=True) == (
        contract, "latent_slots+sink_window_start(unrecorded: donor directory)")


@pytest.mark.parametrize(
    "key,value",
    [("physical_rope", False), ("causal_state", False), ("sliding_window_chunks", 2), ("memory_gdn_output_normalizer", True)],
)
def test_a_run_records_deploy_block_contradicting_the_config_is_refused(key, value):
    contract = resolve_causal_contract(_load(PHYSROPE_S10000))
    record = _record(RECORD_PHYSROPE_S10000)
    record["deploy"] = {**record["deploy"], key: value}
    with pytest.raises(ValueError, match=f"records deploy.{key}="):
        check_run_record(record, contract)


def test_a_run_record_contradicting_the_observation_switches_is_refused():
    contract = resolve_causal_contract(_load(N1_OBS_S15000))
    check_run_record({"causal_observation": True, "causal_state": True, "memory_gdn_output_normalizer": False}, contract)
    for key in ("causal_observation", "causal_state"):
        with pytest.raises(ValueError, match=f"records {key}=False"):
            check_run_record({key: False}, contract)


@pytest.mark.parametrize(
    "key,value",
    [("aligned_phase_window", "padded"), ("memory_content", "video_and_action"), ("chunk_observation", "false"),
     ("memory_gdn_output_normalizer", False)],
)
def test_a_run_record_contradicting_the_config_is_refused(key, value):
    contract = resolve_causal_contract(_load(FIRST_FRAME_NOOBS))
    with pytest.raises(ValueError, match=f"records {key}="):
        check_run_record({**_record(RECORD_NOOBS_S3324), key: value}, contract)


def test_a_record_without_protocol_keys_checks_nothing():
    check_run_record({"step": 3324}, resolve_causal_contract(_load(FIRST_FRAME_NOOBS)))


def test_the_run_records_are_read_next_to_the_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_bytes(open(OBSERVATION_S1000, "rb").read())
    assert load_run_records(tmp_path) == [] and describe_run_records([], config) == "none"
    record = _record(RECORD_S1000)
    (tmp_path / "RUN_ID.json").write_text(json.dumps(record))
    (tmp_path / "upload_manifest.json").write_text(json.dumps({**record, "files": []}))
    records = load_run_records(tmp_path)
    assert [path.name for path, _ in records] == ["RUN_ID.json", "upload_manifest.json"]
    line = describe_run_records(records, config)
    assert line.startswith("RUN_ID.json+upload_manifest.json(chunk_observation=True aligned_phase_window=padded ")
    assert "step=1000 source_commit=215385e9c" in line
    assert line.endswith("config_sha256=differs)")   # the fixture carries a provenance header
    sha = hashlib.sha256(config.read_bytes()).hexdigest()
    assert describe_run_records([(p, {**r, "config_sha256": sha}) for p, r in records], config).endswith("=match)")
    (tmp_path / "upload_manifest.json").write_text("{not json")
    with pytest.raises(ValueError, match="cannot read the run record"):
        load_run_records(tmp_path)


def test_the_runtime_refuses_the_record_before_any_weight(tmp_path):
    """``CausalRuntime.from_paths`` reads the record next to the config before the normalization and the weights (the
    directory holds neither, so any later step would fail differently)."""

    from sana_wam_causal.runtime import CausalRuntime

    (tmp_path / "config.yaml").write_bytes(open(OBSERVATION_S1000, "rb").read())
    (tmp_path / "RUN_ID.json").write_text(open(RECORD_S1000).read())
    with pytest.raises(ValueError, match="records chunk_observation=True"):
        CausalRuntime.from_paths(str(tmp_path), text_encoder_path="unused", vae_path="unused", device="cpu")


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
    def __init__(self, kinds, sink_horizon=None):
        self.blocks = [type("B", (), {"attn_type": kind})() for kind in kinds]
        self.sink_horizon = sink_horizon


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


@pytest.mark.parametrize(
    "window,chunks,ids,origin",
    [
        (1, 0, [], 0),             # chunks 0..7 read the sink in place
        (1, 1, [0], 0),
        (1, 7, [6], 0),            # chunk 7's read: chunk 6 in slot 6, chunk 7 itself in slot 7
        (1, 8, [7], 1),            # past the horizon the sink is read at chunk c - 7: chunk 8 reads it at chunk 1
        (1, 30, [29], 23),
        (2, 12, [10, 11], 5),
        (7, 7, [0, 1, 2, 3, 4, 5, 6], 0),
        (7, 12, [5, 6, 7, 8, 9, 10, 11], 5),   # W = H: the horizon is the window's start
    ],
)
def test_a_sink_horizon_read_is_numbered_from_the_chunk_its_sink_is_read_at(window, chunks, ids, origin):
    """Sana 0c88c11ab: a reader in chunk c reads the sink at the instant chunk max(0, c - 7) starts; the read is
    numbered from that chunk, its window in slots c - W - O .. c - 1 - O and the reader in slot c - O <= 7."""

    memory = CausalMemory(_Model(["GatedSoftmaxAttention"], sink_horizon=7), window)
    _fill(memory, chunks)
    assert memory.read_plan() == (ids, origin) and origin == max(0, chunks - 7)
    asked = []

    def rope_of(slot):
        asked.append(slot)
        return _identity_rope(slot)

    memory.layer_caches(commit=False, rope_of=rope_of, sink_rope=_identity_rope())
    assert asked == [chunk - origin for chunk in ids] and chunks - origin <= 7


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


def _tiny_model(window=2, memory_content=MEMORY_VIDEO_AND_ACTION, phase=ALIGNED_PHASE_PADDED, attn_res=True):
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
        reset_legacy_beta=False, aligned_phase_window=phase, use_attn_res=attn_res,
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
