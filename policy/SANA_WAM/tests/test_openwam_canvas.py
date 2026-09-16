"""CPU tests for sana_wam_min.openwam_canvas (compositor, pixel transform, shared prompt row) and the
training-yaml resolution of the rwm/openwam canvas line (visual layout, action / target modes, prompt sentences)."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

# policy/SANA_WAM (plain ``sana_wam_min`` imports) and the XPolicyLab parent (``XPolicyLab.policy.SANA_WAM`` imports).
ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(ADAPTER_DIR, "..", "..", ".."))
for _p in (ADAPTER_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sana_wam_min import config as wam_config  # noqa: E402
from sana_wam_min import openwam_canvas as canvas  # noqa: E402
from sana_wam_min import text  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
THREE_VIEW_YAML = os.path.join(FIXTURES, "config.yaml")
CANVAS_YAML = os.path.join(FIXTURES, "config_openwam_canvas.yaml")
# OpenWAM's own compositor as vendored by the OpenWAM adapter (policy/OpenWAM/OpenWAM); the parity test skips without it.
OPENWAM_MULTIVIEW = os.path.join(
    ADAPTER_DIR, "..", "OpenWAM", "OpenWAM", "openwam", "dataloader", "transforms", "multiview.py"
)
ABSOLUTE_JOINT_ONLY = "joint_only; joint and gripper targets are absolute future targets and no end-effector action is supervised"


def _frames(seed: int = 0, shapes=((480, 640), (480, 640), (480, 640))) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        camera: rng.integers(0, 256, size=(*hw, 3), dtype=np.uint8)
        for camera, hw in zip(canvas.OPENWAM_CAMERA_LAYOUT, shapes, strict=True)
    }


# -- compositor ---------------------------------------------------------------------------------------------------


def test_slot_geometry_matches_the_plan():
    assert canvas.canvas_slot_boxes() == {
        "cam_head": (0, 0, 256, 320),
        "cam_left_wrist": (256, 0, 128, 160),
        "cam_right_wrist": (256, 160, 128, 160),
    }
    assert canvas.expected_canvas_latent_hw(32) == (12, 10)
    assert canvas.OPENWAM_VIEW_SLOT_IDS == (0,) and canvas.OPENWAM_VIEW_KEY == "openwam_canvas"
    assert canvas.OPENWAM_CAMERA_LAYOUT == ("cam_head", "cam_left_wrist", "cam_right_wrist")


def test_assemble_places_each_camera_in_its_slot_without_gaps():
    colors = {"cam_head": (200, 10, 30), "cam_left_wrist": (5, 180, 60), "cam_right_wrist": (40, 50, 250)}
    shapes = {"cam_head": (480, 640), "cam_left_wrist": (240, 320), "cam_right_wrist": (97, 131)}
    frames = {camera: np.full((*shapes[camera], 3), colors[camera], dtype=np.uint8) for camera in colors}
    out = canvas.assemble_openwam_canvas(frames)
    assert out.shape == (384, 320, 3) and out.dtype == np.uint8
    for camera, (top, left, height, width) in canvas.canvas_slot_boxes().items():
        region = out[top : top + height, left : left + width]
        # a constant image stays constant under the BILINEAR stretch, so every slot is exactly its camera's colour
        assert (region == np.array(colors[camera], dtype=np.uint8)).all(), camera
    assert out[0, 0].tolist() == [200, 10, 30]  # RGB order preserved: the head's red lands in channel 0
    with pytest.raises(KeyError, match="cam_right_wrist"):
        canvas.assemble_openwam_canvas({k: v for k, v in frames.items() if k != "cam_right_wrist"})
    with pytest.raises(ValueError, match="uint8"):
        canvas.assemble_openwam_canvas({**frames, "cam_head": frames["cam_head"].astype(np.float32)})
    with pytest.raises(ValueError, match="exactly 3 cameras"):
        canvas.assemble_openwam_canvas(frames, camera_layout=("cam_head", "cam_left_wrist"))


def test_assemble_accepts_pil_and_array_inputs_identically():
    frames = _frames(1)
    a = canvas.assemble_openwam_canvas(frames)
    b = canvas.assemble_openwam_canvas({k: Image.fromarray(v) for k, v in frames.items()})
    assert np.array_equal(a, b)
    assert not np.array_equal(a, canvas.assemble_openwam_canvas(_frames(2)))


@pytest.mark.skipif(not os.path.exists(OPENWAM_MULTIVIEW), reason="OpenWAM reference compositor not vendored here")
def test_assemble_is_pixel_identical_to_openwam_assemble_multiview_layout():
    spec = importlib.util.spec_from_file_location("openwam_multiview_reference", OPENWAM_MULTIVIEW)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for seed, shapes in ((0, ((480, 640),) * 3), (1, ((480, 640), (240, 320), (300, 300))), (2, ((96, 128),) * 3)):
        frames = _frames(seed, shapes)
        reference = module.assemble_multiview_layout(
            {k: Image.fromarray(v) for k, v in frames.items()}, list(canvas.OPENWAM_CAMERA_LAYOUT), 384, 320
        )
        assert np.array_equal(canvas.assemble_openwam_canvas(frames), np.asarray(reference)), seed


def test_canvas_to_model_tensor_is_the_training_transform_without_resampling():
    out = canvas.assemble_openwam_canvas(_frames(2))
    tensor = canvas.canvas_to_model_tensor(out)
    assert tensor.shape == (3, 384, 320) and tensor.dtype == torch.float32
    expected = (torch.from_numpy(out).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
    torch.testing.assert_close(tensor, expected, rtol=0, atol=1e-6)
    assert float(tensor.min()) >= -1.0 and float(tensor.max()) <= 1.0
    assert tensor[0, 0, 0].item() == pytest.approx(out[0, 0, 0] / 127.5 - 1.0, abs=1e-6)
    with pytest.raises(ValueError, match="uint8"):
        canvas.canvas_to_model_tensor(out[:256])
    with pytest.raises(ValueError, match="uint8"):
        canvas.canvas_to_model_tensor(out.astype(np.float32))


# -- shared prompt row ----------------------------------------------------------------------------------------------


def test_render_canvas_prompt_rows_is_one_shared_row_with_the_composite_view():
    sentence = text.action_mode_text("joint_only", "absolute", "anchor_delta")
    assert sentence == ABSOLUTE_JOINT_ONLY
    (row,) = canvas.render_canvas_prompt_rows("Stack the bowls.", action_mode_text=sentence)
    assert row == (
        "Embodiment Type: dual-arm RoboDojo ARX-X5 robot with parallel grippers.\n"
        f"Action Mode: {ABSOLUTE_JOINT_ONLY}.\n"
        "Observation View: a composite view combining the head camera above the left and right wrist cameras.\n"
        "Instruction: Stack the bowls."
    )
    (uncond,) = canvas.render_canvas_prompt_rows("Stack the bowls.", action_mode_text=sentence, include_instruction=False)
    assert uncond == "\n".join(row.split("\n")[:-1])
    assert text.unconditional_rows_from_conditional((row,)) == (uncond,)
    with pytest.raises(TypeError):
        canvas.render_canvas_prompt_rows("Stack the bowls.")  # the Action Mode sentence is never defaulted


def test_action_mode_sentences_cover_every_line():
    assert text.action_mode_text() == text.ACTION_MODE_TEXT
    assert text.action_mode_text("joint_only", "absolute") == ABSOLUTE_JOINT_ONLY
    assert text.action_mode_text("joint_only", "anchor_delta", "absolute") == text.ACTION_MODE_TEXT
    assert text.action_mode_text("joint_only", "absolute", "absolute") == ABSOLUTE_JOINT_ONLY
    assert text.action_mode_text("robot_base_eef", "anchor_delta", "anchor_delta") == (
        "robot_base_eef; end-effector motion is relative to the first state and expressed in the robot base frame, "
        "joint motion is relative to the first state, and gripper targets are absolute future targets"
    )
    assert text.action_mode_text("robot_base_eef", "absolute", "absolute") == (
        "robot_base_eef; end-effector targets are absolute future poses in the robot base frame, and joint and "
        "gripper targets are absolute future targets"
    )
    with pytest.raises(ValueError, match="unsupported action mode"):
        text.action_mode_text("qwen_canonical", "anchor_delta", "absolute")
    with pytest.raises(ValueError, match="unsupported target modes"):
        text.action_mode_text("joint_only", "velocity")


# -- training-yaml resolution -------------------------------------------------------------------------------------


def _load(path: str) -> dict:
    return wam_config.load_train_config(path)


def test_resolve_visual_layout_from_the_two_fixtures():
    assert wam_config.resolve_visual_layout(_load(THREE_VIEW_YAML)) == "three_view_strip"
    assert wam_config.resolve_visual_layout(_load(CANVAS_YAML)) == "openwam_canvas"
    assert wam_config.VISUAL_LAYOUTS == ("three_view_strip", "openwam_canvas")


def test_resolve_visual_layout_refuses_inconsistent_declarations():
    cfg = _load(CANVAS_YAML)
    bad = copy.deepcopy(cfg)
    bad["data"]["type"] = "RoboDojoSFTDataset"
    with pytest.raises(ValueError, match="needs data.type"):
        wam_config.resolve_visual_layout(bad)
    bad = copy.deepcopy(cfg)
    bad["data"]["extra"]["openwam_canvas"]["layout"] = "openwam_lshape_rgb_v2"
    with pytest.raises(ValueError, match="layout"):
        wam_config.resolve_visual_layout(bad)
    bad = copy.deepcopy(cfg)
    bad["data"]["extra"]["openwam_canvas"]["encode_mode"] = "per_view"
    with pytest.raises(ValueError, match="encode_mode"):
        wam_config.resolve_visual_layout(bad)
    bad = copy.deepcopy(cfg)
    bad["data"]["aspect_ratio_type"] = "ASPECT_RATIO_VIDEO_320_ROBOT"
    with pytest.raises(ValueError, match="ASPECT_RATIO_OPENWAM_LSHAPE_384_320"):
        wam_config.resolve_visual_layout(bad)
    three = _load(THREE_VIEW_YAML)
    bad = copy.deepcopy(three)
    bad["data"]["type"] = "RoboDojoOpenWAMCanvasSFTDataset"
    with pytest.raises(ValueError, match="three-view policy"):
        wam_config.resolve_visual_layout(bad)
    bad = copy.deepcopy(three)
    bad["model"]["model"] = "SomeOtherPolicy_5B"
    with pytest.raises(ValueError, match="unsupported model.model"):
        wam_config.resolve_visual_layout(bad)


def test_canvas_policy_config_resolves_shared_prompt_and_the_state_flag():
    cfg = _load(CANVAS_YAML)
    policy_cfg = wam_config.policy_config_from_train_config(cfg)
    assert isinstance(policy_cfg, PolicyConfig)
    assert policy_cfg.shared_prompt is True and policy_cfg.state_as_cross_attention is False
    assert policy_cfg.input_size == 10 and policy_cfg.multiview_spatial_rope_layout == "local_reset"
    assert policy_cfg.depth == 32 and policy_cfg.softmax_layer_indices == (3, 7, 11, 15, 19, 23, 27, 31)
    assert wam_config.state_as_cross_attention_from_train_config(cfg) is False
    opt_in = copy.deepcopy(cfg)
    opt_in["model"]["extra"]["state_as_cross_attention"] = True
    assert wam_config.policy_config_from_train_config(opt_in).state_as_cross_attention is True
    assert wam_config.state_as_cross_attention_from_train_config(opt_in) is True
    three = _load(THREE_VIEW_YAML)
    three_cfg = wam_config.policy_config_from_train_config(three)
    assert three_cfg.shared_prompt is False and three_cfg.state_as_cross_attention is False
    assert three_cfg.multiview_spatial_rope_layout == "semantic_2x2"
    three["model"]["extra"]["state_as_cross_attention"] = True
    with pytest.raises(ValueError, match="only defined for the OpenWAM canvas"):
        wam_config.state_as_cross_attention_from_train_config(three)
    assert wam_config.sampling_defaults_from_train_config(cfg) == {"steps": 50, "flow_shift": 3.5, "cfg_scale": 1.0}


def test_policy_config_tolerates_the_source_yaml_of_the_canvas_line():
    """The source yaml (no resolved ModelConfig defaults) must resolve exactly like the dumped one."""

    cfg = _load(CANVAS_YAML)
    resolved = wam_config.policy_config_from_train_config(cfg)
    for key in (
        "multiview_spatial_rope_layout", "multiview_spatial_rope_tile_shape", "use_dual_attn_res_routing",
        "cross_attn_image_embeds", "rope_fhw_dim", "softmax_layer_indices",
    ):
        cfg["model"].pop(key, None)
    cfg["text_encoder"].pop("caption_channels", None)
    assert wam_config.policy_config_from_train_config(cfg) == resolved
    assert resolved.caption_channels == 2304 and resolved.multiview_spatial_rope_tile_shape == (15, 30)


def test_action_and_target_modes_from_train_config():
    cfg = _load(CANVAS_YAML)
    assert wam_config.action_mode_from_train_config(cfg) == "joint_only"
    assert wam_config.joint_target_mode_from_train_config(cfg) == "absolute"
    assert wam_config.eef_target_mode_from_train_config(cfg) == "anchor_delta"  # absent -> the pre-command-channel default
    three = _load(THREE_VIEW_YAML)
    assert wam_config.action_mode_from_train_config(three) == "joint_only"
    assert wam_config.joint_target_mode_from_train_config(three) == "anchor_delta"
    eef = copy.deepcopy(three)
    eef["data"]["extra"]["action_mode_sample_ratio"] = [0.0, 1.0, 0.0]
    assert wam_config.action_mode_from_train_config(eef) == "robot_base_eef"
    eef["data"]["extra"]["eef_target_mode"] = "absolute"
    assert wam_config.eef_target_mode_from_train_config(eef) == "absolute"
    eef["data"]["extra"]["eef_target_mode"] = "velocity"
    with pytest.raises(ValueError, match="eef_target_mode"):
        wam_config.eef_target_mode_from_train_config(eef)
    for ratio, message in (
        ([1.0, 0.0, 0.0], "qwen_canonical"),
        ([0.0, 0.5, 0.5], "names 2 action modes"),
        ([0.0, 1.0], "unexpected"),
    ):
        bad = copy.deepcopy(three)
        bad["data"]["extra"]["action_mode_sample_ratio"] = ratio
        with pytest.raises(ValueError, match=message):
            wam_config.action_mode_from_train_config(bad)
    del three["data"]["extra"]["action_mode_sample_ratio"]
    assert wam_config.action_mode_from_train_config(three) == "joint_only"
    bad = copy.deepcopy(cfg)
    bad["data"]["extra"]["joint_target_mode"] = "velocity"
    with pytest.raises(ValueError, match="joint_target_mode"):
        wam_config.joint_target_mode_from_train_config(bad)
