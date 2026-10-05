"""The shared canvas prompt: byte parity with Sana's token-group renderer and the canvas dataset's descriptor."""

from __future__ import annotations

import os
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
import _tiny_mot  # noqa: E402,F401  (sys.path)

from sana_mot_min.prompt import (  # noqa: E402
    ACTION_MODE_TEXT,
    COMPOSITE_VIEW_TEXT,
    canvas_prompt_payload,
    render_canvas_prompt_rows,
    render_multiview_prompt_rows,
    unconditional_rows_from_conditional,
)
from sana_wam_min.text import ROBODOJO_EMBODIMENT  # noqa: E402

SANA_MOT_REPO = os.environ.get("SANA_MOT_REPO", os.path.expanduser("~/zekail/Sana_mot"))
INSTRUCTION = "Stack the three bowls together."


def test_single_shared_row_layout():
    rows = render_canvas_prompt_rows(INSTRUCTION)
    assert len(rows) == 1
    lines = rows[0].split("\n")
    assert lines[0] == f"Embodiment Type: {ROBODOJO_EMBODIMENT}."
    assert lines[1] == f"Action Mode: {ACTION_MODE_TEXT['absolute']}."
    assert lines[2] == f"Observation View: {COMPOSITE_VIEW_TEXT}."
    assert lines[3] == f"Instruction: {INSTRUCTION.rstrip('.')}."
    uncond = render_canvas_prompt_rows(INSTRUCTION, include_instruction=False)
    assert uncond == unconditional_rows_from_conditional(rows) and "Instruction" not in uncond[0]
    payload = canvas_prompt_payload(INSTRUCTION, "anchor_delta")
    assert "relative to the first state" in payload["conditional"][0] and payload["unconditional"][0] == "\n".join(payload["conditional"][0].split("\n")[:-1])
    with pytest.raises(ValueError):
        render_canvas_prompt_rows(INSTRUCTION, joint_target_mode="delta")


@pytest.mark.parametrize("joint_target_mode", ["absolute", "anchor_delta"])
@pytest.mark.parametrize("include_instruction", [True, False])
def test_byte_parity_with_sana_token_group_prompts(joint_target_mode, include_instruction):
    if SANA_MOT_REPO not in sys.path and os.path.isdir(SANA_MOT_REPO):
        sys.path.insert(0, SANA_MOT_REPO)
    live = pytest.importorskip("dev.rwm.diffusion.data.token_group_prompts", reason="Sana rwm/mot checkout not importable")
    reference = live.build_token_group_prompts(
        embodiment=ROBODOJO_EMBODIMENT,
        action_mode="joint_only",
        joint_target_mode=joint_target_mode,
        instruction=INSTRUCTION,
        views=(live.PromptDescriptor("openwam_composite", COMPOSITE_VIEW_TEXT),),
        include_instruction=include_instruction,
    )
    ours = render_canvas_prompt_rows(INSTRUCTION, include_instruction=include_instruction, joint_target_mode=joint_target_mode)
    assert ours == (reference[0],)


def test_composite_descriptor_and_embodiment_match_the_canvas_dataset():
    if SANA_MOT_REPO not in sys.path and os.path.isdir(SANA_MOT_REPO):
        sys.path.insert(0, SANA_MOT_REPO)
    dataset = pytest.importorskip(
        "dev.rwm.diffusion.data.datasets.robodojo_openwam_canvas_sft_data", reason="Sana rwm/mot checkout (and h5py) not importable"
    )
    specs = pytest.importorskip("dev.rwm.diffusion.data.videoaction_dataset_specs")
    # the checkout's descriptor is the current one (two_rows since 3ea1f9af9); COMPOSITE_VIEW_TEXT is the pre-2b4a4dc8d one
    from sana_wam_min.openwam_canvas import OPENWAM_PROMPT_TEXTS

    assert dataset._OPENWAM_COMPOSITE_DESCRIPTOR.text in OPENWAM_PROMPT_TEXTS.values()
    assert OPENWAM_PROMPT_TEXTS["composite_view"] == COMPOSITE_VIEW_TEXT
    assert specs.get_videoaction_dataset_spec("RoboDojo-ARX-X5").embodiment_prompt == ROBODOJO_EMBODIMENT


def test_multiview_rows_layout():
    rows = render_multiview_prompt_rows(INSTRUCTION)
    assert len(rows) == 4
    assert [row.split("\n")[2] for row in rows[:3]] == [
        "Observation View: overhead external camera.",
        "Observation View: left wrist-mounted camera.",
        "Observation View: right wrist-mounted camera.",
    ]
    assert "Observation View" not in rows[3] and rows[3].endswith(f"Instruction: {INSTRUCTION.rstrip('.')}.")
    assert render_multiview_prompt_rows(INSTRUCTION, include_instruction=False) == unconditional_rows_from_conditional(rows)


@pytest.mark.parametrize("joint_target_mode", ["absolute", "anchor_delta"])
@pytest.mark.parametrize("include_instruction", [True, False])
def test_multiview_byte_parity_with_sana(joint_target_mode, include_instruction):
    if SANA_MOT_REPO not in sys.path and os.path.isdir(SANA_MOT_REPO):
        sys.path.insert(0, SANA_MOT_REPO)
    live = pytest.importorskip("dev.rwm.diffusion.data.token_group_prompts", reason="Sana rwm/mot checkout not importable")
    robodojo = pytest.importorskip("dev.rwm.diffusion.data.data_format.robodojo", reason="Sana rwm/mot checkout not importable")
    fmt = robodojo.RoboDojoArxX5DataFormat
    views = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    reference = live.build_token_group_prompts(
        embodiment=ROBODOJO_EMBODIMENT,
        action_mode="joint_only",
        joint_target_mode=joint_target_mode,
        instruction=INSTRUCTION,
        views=tuple(fmt.view_spec(view).descriptor for view in views),
        include_instruction=include_instruction,
    )
    ours = render_multiview_prompt_rows(INSTRUCTION, include_instruction=include_instruction, joint_target_mode=joint_target_mode)
    assert ours == tuple(reference)
    assert tuple(fmt.view_spec(view).spatial_slot for view in views) == (0, 2, 3)
