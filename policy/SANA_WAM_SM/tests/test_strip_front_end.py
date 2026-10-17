"""The sana_latent front-end against Sana's sparse-memory deploy (``SANA_SM_REPO``, live): a control row's three camera
frames stretched to their 256 x 320 buckets and quantized to uint8 (``SMRuntime.frames``) are Sana's ``layout_frames``,
and the shared front-end's G = V + 1 token-group prompts carry the run's Action Mode sentence (deploy.yml
``action_mode_text``) as Sana renders them for the sana_latent m8s25 recipe (``render_token_group_rows``, cc18ad78f:
the absolute-EEF sentence of db382ee90)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import _paths
import _live

from sana_wam_sm.contract import bidirectional_view
from sana_wam_sm.runtime import SMRuntime
from sana_wam_min.robodojo_io import ROBODOJO_VIEW_ORDER, VIEW_SLOT_IDS
from sana_wam_min.text import render_token_group_rows

LATENT_M8S25 = os.path.join(_paths.FIXTURES, "recipe_sm_sana_latent_strided_m8s25_cc18ad78f.yaml")
# Sana db382ee90: the absolute-EEF Action Mode sentence of every run trained after 2026-10-17 12:02 PDT
SHORT_SENTENCE = "robot_base_eef; end-effector targets are absolute future poses expressed in the robot base frame"
INSTRUCTION = "stack the blocks"


def _raw():
    with open(LATENT_M8S25) as handle:
        return yaml.safe_load(handle)


def _namespace(value):
    return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def _live_contract():
    _live.live_session()
    from dev.rwm.deploy import contract

    raw = _raw()
    return raw, contract, contract.resolve_policy_input_contract(raw)


def _strip_runtime() -> SMRuntime:
    runtime = SMRuntime.__new__(SMRuntime)
    view = bidirectional_view(_raw())
    runtime.frontend = SimpleNamespace(
        view_order=ROBODOJO_VIEW_ORDER, view_slot_ids=VIEW_SLOT_IDS, image_size=int(view["model"]["image_size"]),
        view_resize="stretch",
    )
    runtime.contract = SimpleNamespace(visual_layout="three_view_strip")
    return runtime


def test_the_views_and_their_slots_are_the_recipes():
    _raw_cfg, _contract, live = _live_contract()
    assert live.multiview == "sana_latent" and live.view_keys == tuple(ROBODOJO_VIEW_ORDER)
    runtime = _strip_runtime()
    assert runtime.strip and runtime.view_slot_ids == tuple(live.view_slot_ids) == (0, 2, 3)


@pytest.mark.parametrize("frame_hw", [(480, 640), (240, 320)])
def test_a_rows_frames_are_sanas_layout_frames(frame_hw):
    raw, _contract, live = _live_contract()
    from dev.rwm.deploy.common import resolve_aspect_ratios
    from dev.rwm.deploy.causal.sparse_memory_session import layout_frames

    rng = np.random.default_rng(sum(frame_hw))
    cameras = [rng.integers(0, 256, (*frame_hw, 3), dtype=np.uint8) for _ in ROBODOJO_VIEW_ORDER]
    reference = layout_frames(
        [torch.from_numpy(frame).permute(2, 0, 1) for frame in cameras], live, resolve_aspect_ratios(_namespace(raw))
    )
    ours = _strip_runtime().frames(cameras)
    assert ours.dtype == torch.uint8 and tuple(ours.shape) == (3, 3, 256, 320)
    assert torch.equal(ours, reference)


@pytest.mark.parametrize("include_instruction", [True, False], ids=["conditional", "unconditional"])
def test_the_token_group_prompts_are_sanas(include_instruction):
    raw, contract, live = _live_contract()
    reference = contract.render_token_group_rows(raw, live, INSTRUCTION, include_instruction=include_instruction)
    ours = render_token_group_rows(
        INSTRUCTION, include_instruction, action_mode_text=SHORT_SENTENCE, view_order=ROBODOJO_VIEW_ORDER
    )
    assert len(ours) == len(ROBODOJO_VIEW_ORDER) + 1 and ours == reference
    # the table sentence of the shared front-end (the runs before db382ee90) is not this run's
    assert render_token_group_rows(INSTRUCTION, include_instruction, view_order=ROBODOJO_VIEW_ORDER) != reference
