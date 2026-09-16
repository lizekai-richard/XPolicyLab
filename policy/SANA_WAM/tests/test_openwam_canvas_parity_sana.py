"""Parity of the canvas front-end against Sana's own rwm/openwam code (skipped when that checkout is not importable):
the compositor pixel for pixel, the canvas pixel transform, the composite descriptor, and every Action Mode sentence /
the shared prompt row against ``build_token_group_prompts``. Point ``SANA_OPENWAM_REPO`` at a checkout of Sana
``rwm/openwam`` (default ~/zekail/Sana_openwam)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

_SANA_WAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SANA_WAM_DIR not in sys.path:
    sys.path.insert(0, _SANA_WAM_DIR)
SANA_REPO = os.environ.get("SANA_OPENWAM_REPO", os.path.expanduser("~/zekail/Sana_openwam"))
if SANA_REPO not in sys.path and os.path.isdir(SANA_REPO):
    sys.path.insert(0, SANA_REPO)

layout_ref = pytest.importorskip(
    "dev.rwm.diffusion.data.openwam_multiview_layout", reason="Sana rwm/openwam checkout not importable (set SANA_OPENWAM_REPO)"
)
prompts_ref = pytest.importorskip("dev.rwm.diffusion.data.token_group_prompts")

from sana_wam_min import config as wam_config  # noqa: E402
from sana_wam_min import openwam_canvas as canvas  # noqa: E402
from sana_wam_min import text  # noqa: E402


def _frames(seed: int, shapes) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        camera: rng.integers(0, 256, size=(*hw, 3), dtype=np.uint8)
        for camera, hw in zip(canvas.OPENWAM_CAMERA_LAYOUT, shapes, strict=True)
    }


def test_layout_constants_match_sana():
    assert tuple(layout_ref.OPENWAM_CAMERA_LAYOUT) == canvas.OPENWAM_CAMERA_LAYOUT
    assert (layout_ref.OPENWAM_CANVAS_HEIGHT, layout_ref.OPENWAM_CANVAS_WIDTH) == (384, 320)
    assert layout_ref.OPENWAM_TOP_HEIGHT_RATIO == canvas.OPENWAM_TOP_HEIGHT_RATIO
    assert layout_ref.OPENWAM_LAYOUT_ID == wam_config.OPENWAM_CANVAS_LAYOUT_ID


@pytest.mark.parametrize("seed,shapes", [(0, ((480, 640),) * 3), (1, ((480, 640), (240, 320), (300, 300))), (2, ((97, 131),) * 3)])
def test_compositor_matches_sana_pixel_for_pixel(seed, shapes):
    frames = _frames(seed, shapes)
    reference = layout_ref.assemble_openwam_canvas(
        {k: Image.fromarray(v) for k, v in frames.items()},
        layout_ref.OPENWAM_CAMERA_LAYOUT,
        layout_ref.OPENWAM_CANVAS_HEIGHT,
        layout_ref.OPENWAM_CANVAS_WIDTH,
    )
    assert np.array_equal(canvas.assemble_openwam_canvas(frames), np.asarray(reference))


def _dataset_module_constants() -> dict:
    """String constants of the canvas dataset module, read from its source so no h5py is needed to import it."""

    import ast

    path = os.path.join(SANA_REPO, "dev", "rwm", "diffusion", "data", "datasets", "robodojo_openwam_canvas_sft_data.py")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    constants = {}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name, value = node.targets[0].id, node.value
        if isinstance(value, ast.Constant):
            constants[name] = value.value
        elif isinstance(value, ast.Call) and getattr(value.func, "id", None) == "PromptDescriptor":
            constants[name] = tuple(argument.value for argument in value.args)
    return constants


def test_canvas_transform_and_descriptor_match_the_dataset_module():
    transforms_ref = pytest.importorskip("diffusion.data.transforms")
    from torchvision import transforms as T

    reference_transform = T.Compose([transforms_ref.ToTensorVideo(), T.Normalize([0.5] * 3, [0.5] * 3, inplace=True)])
    out = canvas.assemble_openwam_canvas(_frames(3, ((480, 640),) * 3))
    clip = torch.from_numpy(out)[None].permute(0, 3, 1, 2)  # the dataset's ``canvas.permute(0, 3, 1, 2)`` for F = 1
    torch.testing.assert_close(canvas.canvas_to_model_tensor(out), reference_transform(clip)[0], rtol=0, atol=0)
    constants = _dataset_module_constants()
    assert constants["_OPENWAM_COMPOSITE_DESCRIPTOR"] == ("openwam_composite", canvas.OPENWAM_COMPOSITE_VIEW_TEXT)
    assert constants["_OPENWAM_VIEW_KEY"] == canvas.OPENWAM_VIEW_KEY
    assert constants["_ENCODE_MODE_JOINT_RGB_CANVAS"] == wam_config.OPENWAM_CANVAS_ENCODE_MODE


def test_every_action_mode_sentence_and_the_shared_row_match_build_token_group_prompts():
    composite = (prompts_ref.PromptDescriptor("openwam_composite", canvas.OPENWAM_COMPOSITE_VIEW_TEXT),)
    checked = 0
    for joint_mode in ("anchor_delta", "absolute"):
        for eef_mode in ("anchor_delta", "absolute"):
            for action_mode in ("qwen_canonical", "robot_base_eef", "joint_only"):
                common = dict(
                    embodiment=text.ROBODOJO_EMBODIMENT, action_mode=action_mode, joint_target_mode=joint_mode,
                    eef_target_mode=eef_mode, instruction="Pick up the cup.", views=composite,
                )
                try:
                    reference = prompts_ref.build_token_group_prompts(**common, include_instruction=True)
                except ValueError:
                    with pytest.raises(ValueError):
                        text.action_mode_text(action_mode, joint_mode, eef_mode)
                    continue
                sentence = text.action_mode_text(action_mode, joint_mode, eef_mode)
                assert canvas.render_canvas_prompt_rows("Pick up the cup.", action_mode_text=sentence) == (reference[0],)
                unconditional = prompts_ref.build_token_group_prompts(**common, include_instruction=False)
                assert canvas.render_canvas_prompt_rows(
                    "Pick up the cup.", action_mode_text=sentence, include_instruction=False
                ) == (unconditional[0],)
                checked += 1
    assert checked == 10  # 12 combinations minus the two undefined qwen_canonical / absolute-EEF ones


def test_three_view_rows_match_build_token_group_prompts_for_both_joint_target_modes():
    views = (prompts_ref.OVERHEAD_EXTERNAL, prompts_ref.LEFT_WRIST, prompts_ref.RIGHT_WRIST)
    for joint_mode in ("anchor_delta", "absolute"):
        for action_mode in ("joint_only", "robot_base_eef"):
            reference = prompts_ref.build_token_group_prompts(
                embodiment=text.ROBODOJO_EMBODIMENT, action_mode=action_mode, joint_target_mode=joint_mode,
                eef_target_mode="anchor_delta", instruction="Make toast.", views=views, include_instruction=True,
            )
            rows = text.render_token_group_rows("Make toast.", action_mode_text=text.action_mode_text(action_mode, joint_mode))
            assert rows == reference, (joint_mode, action_mode)
