"""The rwm/mot contracts of 2026-09-20 .. 23 resolved from real yamls -- the tip's recipes (fixtures copied verbatim from
71ac93f43), the pinned canvas campaign 19045548 (8e4d6d019) and the earlier NSC / local runs -- and the session behaviour
they drive: the video layout, the RoPE regime, the canvas text contract, the sana_pixel front-end, the EEF-only prompt."""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest
import torch
import yaml

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from _tiny_mot import CONFIG_CANVAS, CONFIG_F33FPS8, CONFIG_MULTIVIEW, CONFIG_NSC_F25, FIXTURES  # noqa: E402
from test_session_cpu import (  # noqa: E402
    CAP_L,
    STEPS,
    StubModel,
    StubTextEncoder,
    StubTokenizer,
    fake_frames,
    fake_state,
    stub_vae,
)

from sana_mot_min.config import (  # noqa: E402
    load_train_config,
    mot_config_from_train_config,
    resolve_mot_rope,
    resolve_mot_text_contract,
    resolve_video_layout,
)
from sana_mot_min.session import PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256, MoTInferenceSession  # noqa: E402
from sana_wam_min.openwam_canvas import OPENWAM_COMPOSITE_VIEW_TEXT, OPENWAM_TWO_ROWS_TEXT  # noqa: E402
from sana_wam_min.robodojo_io import state80_from_obs  # noqa: E402
from sana_wam_min.robot80 import load_normalization  # noqa: E402
from sana_wam_min.sana_pixel_canvas import SANA_PIXEL_TILING_TEXT  # noqa: E402
from sana_wam_min.text import action_mode_text  # noqa: E402

HEAD = os.path.join(FIXTURES, "head_71ac93f43")
EEFABS_PIXEL = os.path.join(HEAD, "sft_robodojo_mot_eefabs_f33fps8_sana_pixel.yaml")
JOINTABS_OPENWAM = os.path.join(HEAD, "sft_robodojo_mot_jointabs_f33fps8_openwam.yaml")
JOINTABS_PIXEL = os.path.join(HEAD, "sft_robodojo_mot_jointabs_f33fps8_sana_pixel.yaml")
JOINTABS_STRIP = os.path.join(HEAD, "sft_robodojo_mot_jointabs_f25_320px.yaml")
PIN_8E4D6D019 = os.path.join(FIXTURES, "config_pin_8e4d6d019_openwam_f33fps8.yaml")
# the frozen config of logits/sft_robodojo_mot_eefabs_sana_pixel_320x480_sanavideo_independent_f33fps8 (rwm/mot e54ec7d97,
# after b010eb9a5: one caption embedder per expert again, ONE canvas row projected by both), copied verbatim
E54_PIXEL = os.path.join(FIXTURES, "config_mot_eefabs_sana_pixel_e54ec7d97.yaml")
# logits/sft_robodojo_mot_eefabs_sana_pixel_320x512_pretrained_aligned_f33fps8 (b010eb9a5 + overlay): the 320x512 canvas
PIXEL_320X512 = os.path.join(FIXTURES, "config_mot_eefabs_sana_pixel_320x512_b010eb9a5.yaml")
SEPARATE, SHARED, LEGACY = "context_embedder", "shared_caption_embedder", "legacy_action_mlp"


def cfg(path: str) -> dict:
    return load_train_config(path)


def test_video_layouts_of_every_era():
    assert resolve_video_layout(cfg(EEFABS_PIXEL), SHARED) == "sana_pixel_canvas"
    assert resolve_video_layout(cfg(JOINTABS_OPENWAM), SHARED) == "openwam_canvas"
    assert resolve_video_layout(cfg(JOINTABS_STRIP), SEPARATE) == "multiview"
    assert resolve_video_layout(cfg(PIN_8E4D6D019), SEPARATE) == "openwam_canvas"     # model.extra.video_layout era
    assert resolve_video_layout(cfg(CONFIG_NSC_F25), SEPARATE) == "openwam_canvas"    # data.type era
    assert resolve_video_layout(cfg(CONFIG_MULTIVIEW), SEPARATE) == "multiview"
    renamed = cfg(JOINTABS_STRIP)
    renamed["data"]["extra"]["multiview"] = "sana"                                   # the 2026-09-21 spelling
    assert resolve_video_layout(renamed, SEPARATE) == "multiview"
    bad = cfg(JOINTABS_OPENWAM)
    bad["data"]["type"] = "RoboDojoSanaPixelCanvasSFTDataset"
    with pytest.raises(ValueError, match="data.extra.multiview is 'openwam'"):
        resolve_video_layout(bad, SHARED)


def test_rope_regimes_of_every_era():
    assert resolve_mot_rope(cfg(EEFABS_PIXEL), SHARED)[0] == "independent"            # declared
    assert resolve_mot_rope(cfg(JOINTABS_STRIP), SEPARATE)[0] == "independent"        # declared
    # campaign 19045548 (8e4d6d019) and the NSC runs predate 42aee4fa9: physical clock, strided state+actions from 0
    assert resolve_mot_rope(cfg(PIN_8E4D6D019), SEPARATE)[0] == "legacy"
    assert resolve_mot_rope(cfg(CONFIG_NSC_F25), SEPARATE)[0] == "legacy"
    assert resolve_mot_rope(cfg(CONFIG_F33FPS8), SEPARATE)[0] == "legacy"
    assert resolve_mot_rope(cfg(CONFIG_CANVAS), LEGACY)[0] == "legacy"
    undeclared = cfg(EEFABS_PIXEL)
    del undeclared["model"]["extra"]["rope"]
    assert resolve_mot_rope(undeclared, SHARED)[0] == "independent"                   # 42aee4fa9 .. 71ac93f43
    assert resolve_mot_rope(cfg(PIN_8E4D6D019), SEPARATE, "independent")[0] == "independent"
    with pytest.raises(ValueError, match="contradicts"):
        resolve_mot_rope(cfg(EEFABS_PIXEL), SHARED, "aligned")
    with pytest.raises(ValueError, match="predates the RoPE modes"):
        resolve_mot_rope(cfg(CONFIG_CANVAS), LEGACY, "independent")


def test_canvas_text_contracts_of_every_era():
    assert resolve_mot_text_contract(cfg(JOINTABS_STRIP), SEPARATE)[:2] == (4, None)
    assert resolve_mot_text_contract(cfg(JOINTABS_OPENWAM), SHARED)[:2] == (1, "two_rows")
    assert resolve_mot_text_contract(cfg(EEFABS_PIXEL), SHARED)[:2] == (1, "tiling")
    # the pinned campaign: ONE composite-view row read by both experts' embedders
    assert resolve_mot_text_contract(cfg(PIN_8E4D6D019), SEPARATE)[:2] == (1, "composite_view")
    assert resolve_mot_text_contract(cfg(CONFIG_NSC_F25), SEPARATE)[:2] == (1, "composite_view")
    # separate embedders with data.extra.multiview = the 2026-09-21..22 G = 2 payload
    mid = cfg(JOINTABS_OPENWAM)
    del mid["model"]["extra"]["rope"]
    assert resolve_mot_text_contract(mid, SEPARATE)[:2] == (2, "composite_view")
    # shared embedder before the rope key: the l_shape sentence (4e67e1e1d .. 3ea1f9af9)
    assert resolve_mot_text_contract(mid, SHARED)[:2] == (1, "l_shape")
    assert resolve_mot_text_contract(mid, SHARED, None, "two_rows")[:2] == (1, "two_rows")
    with pytest.raises(ValueError, match="ONE prompt row"):
        resolve_mot_text_contract(cfg(JOINTABS_OPENWAM), SHARED, 2)


def test_the_b010eb9a5_canvas_reads_one_row_through_both_embedders():
    """Separate caption embedders + data.extra.multiview + a declared model.extra.rope can only be rwm/mot b010eb9a5 on
    (the canvas modes had the shared embedder from 4e67e1e1d until then, and the rope key arrived with 71ac93f43)."""

    assert resolve_mot_text_contract(cfg(E54_PIXEL), SEPARATE)[:2] == (1, "tiling")
    openwam = cfg(E54_PIXEL)
    openwam["data"]["extra"]["multiview"] = "openwam"
    assert resolve_mot_text_contract(openwam, SEPARATE)[:2] == (1, "two_rows")
    undeclared = cfg(E54_PIXEL)
    del undeclared["model"]["extra"]["rope"]
    assert resolve_mot_text_contract(undeclared, SEPARATE)[:2] == (2, "composite_view")     # a26807821 .. 2b4a4dc8d
    mot = mot_config_from_train_config(cfg(E54_PIXEL), SEPARATE)
    assert (mot.video_layout, mot.rope, mot.canvas_text_groups, mot.shared_caption_embedder) == (
        "sana_pixel_canvas", "independent", 1, False
    )
    session = _session(E54_PIXEL, SEPARATE)
    assert (session.video_layout, session.text_groups, session.canvas_prompt) == ("sana_pixel_canvas", 1, "tiling")
    assert session.action_mode == "robot_base_eef" and session.robot_base_eef_layout == "eef_only"
    _predict(session)
    assert session.model.calls[0]["y_shape"][1] == 1
    (row,) = session.tokenizer.calls[0]
    assert f"Observation View: {SANA_PIXEL_TILING_TEXT}." in row


def test_the_320x512_sana_pixel_checkpoint_gets_its_wide_canvas():
    """Before 2026-09-26 the MoT resolver ignored data.aspect_ratio_type and served every sana_pixel yaml a 320x480 canvas."""

    assert resolve_video_layout(cfg(PIXEL_320X512), SEPARATE) == "sana_pixel_canvas"
    assert resolve_mot_text_contract(cfg(PIXEL_320X512), SEPARATE)[:2] == (1, "tiling")
    session = _session(PIXEL_320X512, SEPARATE)
    assert session.sana_pixel_canvas_hw == (320, 512)
    _predict(session)
    call = session.model.calls[0]
    assert call["data_info"]["view_latent_shape"].tolist() == [[[10, 16]]]
    encoded = session.vae.causal_encoder.inputs[0]
    assert tuple(encoded.shape[-2:]) == (320, 512) and torch.all(encoded[..., :160, 256:] == -1.0)
    bad = cfg(PIXEL_320X512)
    bad["data"]["aspect_ratio_type"] = "ASPECT_RATIO_SANA_PIXEL_2X2_320_448"
    with pytest.raises(ValueError, match="aspect_ratio_type"):
        resolve_video_layout(bad, SEPARATE)


def test_mot_config_of_the_tip_recipes():
    mot = mot_config_from_train_config(cfg(EEFABS_PIXEL), SHARED)
    assert (mot.video_layout, mot.rope, mot.canvas_text_groups, mot.shared_caption_embedder) == (
        "sana_pixel_canvas", "independent", 1, True
    )
    assert mot.multiview_spatial_rope_layout == "semantic_2x2" and mot.multiview_spatial_rope_tile_shape == (15, 30)
    strip = mot_config_from_train_config(cfg(JOINTABS_STRIP), SEPARATE)
    assert (strip.video_layout, strip.rope, strip.shared_caption_embedder) == ("multiview", "independent", False)
    pinned = mot_config_from_train_config(cfg(PIN_8E4D6D019), SEPARATE)
    assert (pinned.video_layout, pinned.rope, pinned.canvas_text_groups) == ("openwam_canvas", "legacy", 1)
    with pytest.raises(ValueError, match="canvas modes only"):
        mot_config_from_train_config(cfg(JOINTABS_STRIP), SHARED)


def _session(path: str, context_layout: str, **knobs) -> MoTInferenceSession:
    train_cfg = copy.deepcopy(cfg(path))
    train_cfg["text_encoder"]["model_max_length"] = CAP_L
    # the packaged artifact is the f25 absolute one: serve the recipe on a 25-row window
    train_cfg["data"]["num_frames"] = 25
    train_cfg["data"]["multi_fps"] = {"25": [25]}
    extra = train_cfg["data"]["extra"]
    options = extra["robot_sft"] if "robot_sft" in extra else extra["robotwin_sft"]   # renamed on 0928354f8
    options["tier_num_frames"] = 25
    if options.get("video_fps") is not None:
        options["video_fps"] = 8
    return MoTInferenceSession(
        model=StubModel(resolve_video_layout(train_cfg, context_layout)),
        vae=stub_vae(),
        tokenizer=StubTokenizer(),
        text_encoder=StubTextEncoder(),
        normalization=load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256),
        train_config=train_cfg,
        device="cpu",
        steps=STEPS,
        cfg_scale=1.0,
        flow_shift=3.5,
        checkpoint_path="stub",
        video_layout=resolve_video_layout(train_cfg, context_layout),
        context_layout=context_layout,
        **knobs,
    )


def _predict(session: MoTInferenceSession, state_mask=None):
    state80, mask80 = state80_from_obs(fake_state())
    if state_mask is not None:
        mask80 = state_mask
    return session.predict(fake_frames(), state80, mask80, "Stack the bowls.", torch.Generator().manual_seed(1))


def test_a_sana_pixel_session_encodes_one_320x480_canvas_with_the_tiling_row():
    session = _session(JOINTABS_PIXEL, SHARED)
    assert (session.video_layout, session.text_groups, session.canvas_prompt) == ("sana_pixel_canvas", 1, "tiling")
    result = _predict(session)
    call = session.model.calls[0]
    assert call["data_info"]["view_count"].tolist() == [1] and call["data_info"]["view_keys"] == ["sana_pixel_canvas"]
    assert call["data_info"]["view_latent_shape"].tolist() == [[[10, 15]]] and call["y_shape"][1] == 1
    assert call["data_info"]["video_frame_stride"].tolist() == [3]                   # 25 rows at video_fps 8
    assert result.video_latent.shape[-2:] == (10, 15) and result.video_latent.shape[2] == 2
    encoded = session.vae.causal_encoder.inputs[0]
    assert tuple(encoded.shape[-2:]) == (320, 480) and torch.all(encoded[..., :160, 240:] == -1.0)
    (row,) = session.tokenizer.calls[0]
    assert f"Observation View: {SANA_PIXEL_TILING_TEXT}." in row


def test_the_openwam_row_follows_the_era():
    tip = _session(JOINTABS_OPENWAM, SHARED)
    _predict(tip)
    assert f"Observation View: {OPENWAM_TWO_ROWS_TEXT}." in tip.tokenizer.calls[0][0]
    pinned = _session(PIN_8E4D6D019, SEPARATE)
    _predict(pinned)
    assert len(pinned.tokenizer.calls[0]) == 1 and f"Observation View: {OPENWAM_COMPOSITE_VIEW_TEXT}." in pinned.tokenizer.calls[0][0]


def test_the_eef_only_line_renders_the_eef_only_sentence():
    session = _session(EEFABS_PIXEL, SHARED)
    assert session.action_mode == "robot_base_eef" and session.robot_base_eef_layout == "eef_only"
    assert session.action_mode_text == action_mode_text("robot_base_eef", "absolute", "absolute", eef_only=True)
    mask = np.zeros(80, dtype=bool)
    mask[list(range(7, 17)) + list(range(36, 46))] = True
    _predict(session, state_mask=mask)
    (row,) = session.tokenizer.calls[0]
    assert "Action Mode: robot_base_eef; end-effector targets are absolute future poses in the robot base frame, and gripper targets" in row
    assert "joint" not in row.split("\n")[1]


def test_a_mot_recipe_with_its_own_action_schedule():
    """rwm/mot b865ad732: a recipe declaring scheduler.action_flow_shift / inference_action_flow_shift samples the action
    stream on its own schedule, so the MoT lockstep lets the action rows share a t of their own."""

    from sana_mot_min.mot_model.model import _assert_lockstep
    from sana_wam_min.session import resolve_action_flow_shift

    recipe = cfg(E54_PIXEL)
    assert not mot_config_from_train_config(recipe, SEPARATE).separate_action_schedule and resolve_action_flow_shift(recipe) is None
    recipe.setdefault("scheduler", {}).update(action_flow_shift=1.0, inference_action_flow_shift=1.0)
    assert mot_config_from_train_config(recipe, SEPARATE).separate_action_schedule and resolve_action_flow_shift(recipe) == 1.0
    t = torch.tensor([[0.0, 700.0]])
    rows = {"action_timestep": torch.full((1, 8), 120.0)}
    _assert_lockstep(t, rows, 2, separate_action_schedule=True)
    with pytest.raises(ValueError, match="lockstep"):
        _assert_lockstep(t, rows, 2)


def test_the_mot_strip_tile_follows_the_training_tree():
    """rwm/mot 66ded97a6 (= zekai-merge 6da565230): the strip's semantic tile (15, 30) -> (8, 16), no yaml key."""

    strip = cfg(JOINTABS_STRIP)
    assert mot_config_from_train_config(strip, SEPARATE).multiview_spatial_rope_tile_shape == (15, 30)
    renamed = copy.deepcopy(strip)
    options = renamed["data"]["extra"].pop("robotwin_sft")
    renamed["data"]["extra"]["robot_sft"] = options            # a config written for a tree at or after 0928354f8
    assert mot_config_from_train_config(renamed, SEPARATE).multiview_spatial_rope_tile_shape == (8, 16)
    renamed["model"].setdefault("extra", {})["multiview_spatial_rope_tile_shape"] = [15, 30]
    assert mot_config_from_train_config(renamed, SEPARATE).multiview_spatial_rope_tile_shape == (15, 30)


# logits/sft_robodojo_eefabs_sana_pixel_320x512_mot49500_aligned_f33fps8 (rwm/mot 0928354f8 + overlay d93766d1), copied
# verbatim: SFT from the MoT pretrain e9s49500, 320x512 canvas, rope aligned, robot_sft, video / action flow shifts 5 / 1
MOT49500 = os.path.join(FIXTURES, "config_mot_eefabs_sana_pixel_320x512_mot49500_0928354f8.yaml")


def test_the_mot49500_recipe_resolves_its_contract():
    from sana_wam_min.session import resolve_action_flow_shift

    recipe = cfg(MOT49500)
    mot = mot_config_from_train_config(recipe, SEPARATE)
    assert (mot.video_layout, mot.rope, mot.canvas_text_groups, mot.separate_action_schedule) == (
        "sana_pixel_canvas", "aligned", 1, True
    )
    assert resolve_mot_text_contract(recipe, SEPARATE)[:2] == (1, "tiling")
    assert resolve_action_flow_shift(recipe) == 1.0 and resolve_action_flow_shift(recipe, 2.0) == 2.0
    session = _session(MOT49500, SEPARATE)
    assert session.sana_pixel_canvas_hw == (320, 512) and session.action_flow_shift == 1.0
    _predict(session)
    assert session.model.calls[0]["data_info"]["view_latent_shape"].tolist() == [[[10, 16]]]


# the openwam twin of the same MoT pretrain (logits/sft_robodojo_eefabs_openwam_320x384_mot49500_aligned_f33fps8, same frozen
# revision + overlay): the OpenWAM L-shape canvas, width 320 x height 384 (tensor 384x320), latent [128, 2, 12, 10] per its README
MOT49500_OPENWAM = os.path.join(FIXTURES, "config_mot_eefabs_openwam_320x384_mot49500_0928354f8.yaml")


def test_the_mot49500_openwam_recipe_resolves_its_contract():
    from sana_wam_min.session import resolve_action_flow_shift

    recipe = cfg(MOT49500_OPENWAM)
    mot = mot_config_from_train_config(recipe, SEPARATE)
    assert (mot.video_layout, mot.rope, mot.canvas_text_groups, mot.separate_action_schedule) == (
        "openwam_canvas", "aligned", 1, True
    )
    assert resolve_mot_text_contract(recipe, SEPARATE)[:2] == (1, "two_rows")
    assert resolve_action_flow_shift(recipe) == 1.0
    session = _session(MOT49500_OPENWAM, SEPARATE)
    assert session.action_flow_shift == 1.0
    _predict(session)
    call = session.model.calls[0]
    assert call["data_info"]["view_keys"] == ["openwam_canvas"] and call["data_info"]["view_count"].tolist() == [1]
    assert call["data_info"]["view_latent_shape"].tolist() == [[[12, 10]]]


# the sana_latent twin (logits/sft_robodojo_eefabs_sana_latent_256x320_mot49500_aligned_f33fps8, same revision + overlay): three
# 256x320 views encoded separately and packed as an 8x30 strip with the semantic 2x2 tile (8, 16), G = V + 1 = 4 text rows;
# no data.extra.robot_sft.view_resize key: trained on the original resize-and-centre-crop, so served with view_resize crop
MOT49500_LATENT = os.path.join(FIXTURES, "config_mot_eefabs_sana_latent_256x320_mot49500_0928354f8.yaml")


def test_the_mot49500_sana_latent_recipe_resolves_its_contract():
    from sana_wam_min.session import resolve_action_flow_shift

    recipe = cfg(MOT49500_LATENT)
    assert "view_resize" not in recipe["data"]["extra"]["robot_sft"]
    mot = mot_config_from_train_config(recipe, SEPARATE)
    assert (mot.video_layout, mot.rope, mot.separate_action_schedule, mot.multiview_spatial_rope_tile_shape) == (
        "multiview", "aligned", True, (8, 16)
    )
    assert resolve_mot_text_contract(recipe, SEPARATE)[:2] == (4, None)
    assert resolve_action_flow_shift(recipe) == 1.0
    session = _session(MOT49500_LATENT, SEPARATE)
    assert session.action_flow_shift == 1.0
    _predict(session)
    call = session.model.calls[0]
    assert call["data_info"]["view_count"].tolist() == [3]
    assert call["data_info"]["view_latent_shape"].tolist() == [[[8, 10], [8, 10], [8, 10]]]
    assert len(session.tokenizer.calls[0]) == 4


def test_view_resize_reaches_the_mot_encoder(tmp_path):
    """rwm/mot 034e55dca / f06615b2f stretch every SFT view whole to its bucket: the sana_pixel canvas and the strip's
    per-view clips the MoT session encodes follow view_resize (default stretch; these pre-034e55dca checkpoints are served
    with the legacy crop); the transforms themselves are byte-checked against the live tree in
    SANA_WAM/tests/test_view_resize_live_parity.py."""

    from sana_wam_min.pixels import frame_to_model_tensor, target_size_hw
    from sana_wam_min.sana_pixel_canvas import sana_pixel_canvas_from_frames

    frames = [np.asarray(frame) for frame in fake_frames()]
    encoded = {}
    for mode in ("crop", "stretch"):
        knobs = {} if mode == "stretch" else {"view_resize": mode}
        session = _session(MOT49500, SEPARATE, **knobs)
        assert (session.view_resize, session.view_resize_source) == (mode, "default" if mode == "stretch" else "deploy")
        _predict(session)
        expected = sana_pixel_canvas_from_frames(frames, session.view_slot_ids, session.sana_pixel_canvas_hw, view_resize=mode)
        assert torch.equal(session.vae.causal_encoder.inputs[0][0, :, 0], expected)
        encoded[mode] = session.vae.causal_encoder.inputs[0]
        strip = _session(MOT49500_LATENT, SEPARATE, **knobs)
        _predict(strip)
        for i, frame in enumerate(frames):
            target = target_size_hw(strip.image_size, frame_hw=(int(frame.shape[0]), int(frame.shape[1])))
            assert torch.equal(strip.vae.causal_encoder.inputs[i][0, :, 0], frame_to_model_tensor(frame, target, mode))
    assert not torch.equal(encoded["crop"], encoded["stretch"])

    masked = cfg(MOT49500)
    masked["model"].setdefault("extra", {})["sana_pixel_pad"] = "masked"
    path = tmp_path / "masked.yaml"
    path.write_text(yaml.safe_dump(masked))
    with pytest.raises(NotImplementedError, match="sana_pixel_pad"):
        _session(str(path), SEPARATE)
