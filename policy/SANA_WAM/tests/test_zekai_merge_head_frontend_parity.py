"""Front-end and language parity with rwm/zekai-merge at the 2026-09-23 contracts (``SANA_ZEKAI_MERGE_REPO``, default
the worktree ~/zekail/Sana_b13415841; later trees moved the private descriptor constants): the sana_pixel canvas pixel
for pixel against Sana's compositor fed by Sana's own clip
transforms, the canvas descriptors of every era (read from the dataset modules' source at the commits that introduced
them, so no h5py is needed), the canvas rows at G = 1 and G = 2 and the EEF-only Action Mode sentences against
``build_token_group_prompts``, the multiview mode names and the RoPE-mode reader. Run on its own like the other live
parity files (the first file of a session owns the ``dev`` package)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from _tiny_policy import ADAPTER_DIR  # noqa: E402,F401  (sys.path)

SANA_REPO = os.environ.get("SANA_ZEKAI_MERGE_REPO", os.path.expanduser("~/zekail/Sana_b13415841"))
if SANA_REPO not in sys.path and os.path.isdir(SANA_REPO):
    sys.path.insert(0, SANA_REPO)

pixel_layout_ref = pytest.importorskip(
    "dev.rwm.diffusion.data.sana_pixel_multiview_layout",
    reason="Sana rwm/zekai-merge (>= b71474ad1) not importable (set SANA_ZEKAI_MERGE_REPO)",
)
if not os.path.abspath(pixel_layout_ref.__file__).startswith(os.path.abspath(SANA_REPO)):
    pytest.skip("the dev package belongs to another Sana checkout; run this file on its own", allow_module_level=True)
prompts_ref = pytest.importorskip("dev.rwm.diffusion.data.token_group_prompts")
multiview_ref = pytest.importorskip("dev.rwm.diffusion.data.multiview")
transforms_ref = pytest.importorskip("diffusion.data.transforms")
wan_mrope_ref = pytest.importorskip("dev.rwm.diffusion.model.layers.wan_mrope")

from torchvision import transforms as T  # noqa: E402

from sana_wam_min import config as wam_config  # noqa: E402
from sana_wam_min import openwam_canvas as openwam  # noqa: E402
from sana_wam_min import sana_pixel_canvas as pixel  # noqa: E402
from sana_wam_min import text  # noqa: E402

DATASETS = os.path.join("dev", "rwm", "diffusion", "data", "datasets")
OPENWAM_DATASET = os.path.join(DATASETS, "robodojo_openwam_canvas_sft_data.py")
PIXEL_DATASET = os.path.join(DATASETS, "robodojo_sana_pixel_canvas_sft_data.py")


def _constants(source: str) -> dict:
    constants = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name, value = node.targets[0].id, node.value
        if isinstance(value, ast.Constant):
            constants[name] = value.value
        elif isinstance(value, ast.Call) and getattr(value.func, "id", None) == "PromptDescriptor":
            constants[name] = tuple(ast.literal_eval(argument) for argument in value.args)
    return constants


def _source_at(commit: str, path: str) -> str:
    if commit == "HEAD":
        with open(os.path.join(SANA_REPO, path), encoding="utf-8") as handle:
            return handle.read()
    try:
        return subprocess.run(
            ["git", "-C", SANA_REPO, "show", f"{commit}:{path}"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"cannot read {path} at {commit}: {error}")


# -- sana_pixel pixels ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["stretch", "crop"])
@pytest.mark.parametrize("aspect", ["ASPECT_RATIO_SANA_PIXEL_2X2_320_480", "ASPECT_RATIO_SANA_PIXEL_2X2_320_512"])
@pytest.mark.parametrize("seed, shapes", [(0, ((480, 640),) * 3), (1, ((480, 640), (240, 320), (300, 400))), (2, ((97, 131),) * 3)])
def test_sana_pixel_canvas_matches_the_training_path_bit_for_bit(seed, shapes, aspect, mode):
    if not hasattr(pixel_layout_ref, "SANA_PIXEL_CANVASES"):
        if aspect.endswith("320_512"):
            pytest.skip("the Sana checkout predates the 320x512 canvas (88a22ba0c)")
        canvas_hw = (320, 480)
    else:
        canvas_hw = pixel_layout_ref.sana_pixel_canvas_hw(aspect)
    assert pixel.sana_pixel_canvas_hw(aspect) == canvas_hw
    rng = np.random.default_rng(seed)
    frames = [rng.integers(0, 256, size=(*hw, 3), dtype=np.uint8) for hw in shapes]
    # every view at the canvas bucket through ToTensorVideo -> resize -> Normalize (F = 1): StretchResize since
    # 1061b16f0 (the SFT loader, and the adapter default); ResizeCrop = the base loader of every earlier tree (legacy)
    if mode == "stretch":
        try:
            from dev.rwm.diffusion.data.view_resize import StretchResize as resize_ref
        except ImportError:
            pytest.skip("the Sana checkout predates the view stretch (1061b16f0)")
    else:
        resize_ref = transforms_ref.ResizeCrop
    clips = [
        T.Compose([transforms_ref.ToTensorVideo(), resize_ref(canvas_hw), T.Normalize([0.5] * 3, [0.5] * 3, inplace=True)])(
            torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).contiguous()
        )
        for frame in frames
    ]
    reference = pixel_layout_ref.assemble_sana_pixel_canvas(clips, (0, 2, 3), *canvas_hw)
    ours = pixel.sana_pixel_canvas_from_frames(frames, (0, 2, 3), canvas_hw, **({} if mode == "stretch" else {"view_resize": mode}))
    assert ours.shape == reference[0].shape == (3, *canvas_hw)
    torch.testing.assert_close(ours, reference[0], rtol=0, atol=0)


def test_sana_pixel_constants_and_the_view_contract():
    assert (pixel_layout_ref.SANA_PIXEL_CANVAS_HEIGHT, pixel_layout_ref.SANA_PIXEL_CANVAS_WIDTH) == (320, 480)
    assert pixel_layout_ref.SANA_PIXEL_FILL_VALUE == pixel.SANA_PIXEL_FILL_VALUE == -1.0
    assert pixel_layout_ref.SANA_PIXEL_LAYOUT_ID == pixel.SANA_PIXEL_LAYOUT_ID
    assert pixel_layout_ref.sana_pixel_tile_shape() == pixel.sana_pixel_tile_shape() == (160, 240)
    constants = _constants(_source_at("HEAD", PIXEL_DATASET))
    assert constants["_SANA_PIXEL_VIEW_KEY"] == pixel.SANA_PIXEL_VIEW_KEY
    assert constants["_SANA_PIXEL_COMPOSITE_DESCRIPTOR"] == ("sana_pixel_composite", pixel.SANA_PIXEL_TILING_TEXT)
    old = _constants(_source_at("b71474ad1", PIXEL_DATASET))
    assert old["_SANA_PIXEL_COMPOSITE_DESCRIPTOR"] == ("sana_pixel_composite", pixel.SANA_PIXEL_COMPOSITE_VIEW_TEXT)


def test_openwam_descriptors_of_every_era():
    for commit, key in (("b31137206", "composite_view"), ("2a10d0d75", "l_shape"), ("92b9e64d1", "two_rows"), ("HEAD", "two_rows")):
        constants = _constants(_source_at(commit, OPENWAM_DATASET))
        assert constants["_OPENWAM_COMPOSITE_DESCRIPTOR"] == ("openwam_composite", openwam.OPENWAM_PROMPT_TEXTS[key]), commit
        assert constants["_OPENWAM_VIEW_KEY"] == openwam.OPENWAM_VIEW_KEY


# -- language ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize(
    "view_text",
    [
        openwam.OPENWAM_COMPOSITE_VIEW_TEXT,
        openwam.OPENWAM_L_SHAPE_TEXT,
        openwam.OPENWAM_TWO_ROWS_TEXT,
        pixel.SANA_PIXEL_COMPOSITE_VIEW_TEXT,
        pixel.SANA_PIXEL_TILING_TEXT,
    ],
)
def test_canvas_rows_match_build_token_group_prompts(groups, view_text):
    descriptor = (prompts_ref.PromptDescriptor("composite", view_text),)
    for action_mode, joint_mode, eef_mode, eef_only in (
        ("joint_only", "absolute", "absolute", False),
        ("joint_only", "anchor_delta", "anchor_delta", False),
        ("robot_base_eef", "absolute", "absolute", True),
        ("robot_base_eef", "anchor_delta", "anchor_delta", True),
    ):
        common = dict(
            embodiment=text.ROBODOJO_EMBODIMENT, action_mode=action_mode, joint_target_mode=joint_mode,
            eef_target_mode=eef_mode, instruction="Stack the bowls.", views=descriptor,
        )
        sentence = text.action_mode_text(action_mode, joint_mode, eef_mode, eef_only=eef_only)
        for include in (True, False):
            reference = prompts_ref.build_token_group_prompts(**common, include_instruction=include)
            expected = reference[:1] if groups == 1 else reference
            rows = openwam.render_canvas_prompt_rows(
                "Stack the bowls.", action_mode_text=sentence, include_instruction=include, view_text=view_text, groups=groups
            )
            assert rows == tuple(expected), (action_mode, joint_mode, eef_mode, include)


def test_every_head_action_mode_sentence_matches_the_eef_only_contract():
    views = (prompts_ref.OVERHEAD_EXTERNAL, prompts_ref.LEFT_WRIST, prompts_ref.RIGHT_WRIST)
    checked = 0
    for joint_mode in ("anchor_delta", "absolute"):
        for eef_mode in ("anchor_delta", "absolute"):
            for action_mode in ("joint_only", "robot_base_eef"):
                reference = prompts_ref.build_token_group_prompts(
                    embodiment=text.ROBODOJO_EMBODIMENT, action_mode=action_mode, joint_target_mode=joint_mode,
                    eef_target_mode=eef_mode, instruction="Make toast.", views=views, include_instruction=True,
                )
                sentence = text.action_mode_text(action_mode, joint_mode, eef_mode, eef_only=True)
                assert text.render_token_group_rows("Make toast.", action_mode_text=sentence) == reference
                checked += 1
    assert checked == 8


def test_multiview_names_and_the_rope_reader_match_the_live_modules():
    assert multiview_ref.MULTIVIEW_MODES == wam_config.MULTIVIEW_MODES
    assert multiview_ref.DEFAULT_MULTIVIEW == wam_config.MULTIVIEW_SANA_LATENT
    assert wan_mrope_ref.ROPE_MODES == wam_config.ROPE_MODES
    from types import SimpleNamespace

    for extra in ({}, {"rope": "aligned"}, {"rope": "Independent "}):
        live = wan_mrope_ref.rope_mode_from_config(SimpleNamespace(model=SimpleNamespace(extra=dict(extra))))
        assert live == wam_config.rope_mode_from_train_config({"model": {"extra": dict(extra)}})
