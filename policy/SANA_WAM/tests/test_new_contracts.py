"""The 2026-09-20 .. 23 training contracts of rwm/zekai-merge, resolved from real recipe yamls (fixtures copied verbatim
from b13415841) and from the yamls of the eras before them: the multiview mode (strip / openwam canvas / sana_pixel
canvas), the RoPE contract, the canvas text contract, the EEF-only robot_base_eef meaning, the spatial-RoPE default, and
the session / adapter behaviour they drive (the sana_pixel front-end, strided canvases, EEF-only masks)."""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch
import yaml

from _tiny_policy import ROBOT_DIM  # noqa: E402  (puts policy/SANA_WAM on sys.path)
from test_session_cpu import (  # noqa: E402
    CAP_L,
    LATENT_C,
    STEPS,
    StubModel,
    StubTextEncoder,
    StubTokenizer,
    fake_frames,
    fake_state,
)

from sana_wam_min import config as wam_config  # noqa: E402
from sana_wam_min import text  # noqa: E402
from sana_wam_min.openwam_canvas import OPENWAM_COMPOSITE_VIEW_TEXT, OPENWAM_L_SHAPE_TEXT, OPENWAM_TWO_ROWS_TEXT  # noqa: E402
from sana_wam_min.robodojo_io import state80_from_obs  # noqa: E402
from sana_wam_min.robot80 import load_normalization  # noqa: E402
from sana_wam_min.sana_pixel_canvas import (  # noqa: E402
    SANA_PIXEL_TILING_TEXT,
    assemble_sana_pixel_canvas,
    sana_pixel_canvas_from_frames,
)
from sana_wam_min.session import PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256, PolicyInferenceSession  # noqa: E402
from sana_wam_min.vae import VaeBundle  # noqa: E402

HEAD = os.path.join(os.path.dirname(__file__), "fixtures", "head_b13415841")
STRIP_DENSE = "sft_robodojo_pretrain157k_jointdelta_f25_320px.yaml"
STRIP_FPS8 = "sft_robodojo_pretrain157k_jointdelta_f33fps8_320px.yaml"
OPENWAM_FPS8 = "sft_robodojo_sanavideo_jointabs_f33fps8_openwam.yaml"
PIXEL_DENSE = "sft_robodojo_sanavideo_jointabs_f33_sana_pixel.yaml"
PIXEL_EEF_FPS8 = "sft_robodojo_sanavideo_eefabs_f33fps8_sana_pixel.yaml"
EEF_DELTA = "sft_robodojo_pretrain157k_eefdelta_f33_320px.yaml"
LEGACY_THREE_VIEW = os.path.join(os.path.dirname(__file__), "fixtures", "config.yaml")
LEGACY_CANVAS = os.path.join(os.path.dirname(__file__), "fixtures", "config_openwam_canvas.yaml")


def head(name: str) -> dict:
    with open(os.path.join(HEAD, name)) as handle:
        return yaml.safe_load(handle)


def legacy(path: str) -> dict:
    return wam_config.load_train_config(path)


def without_rope(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["model"]["extra"].pop("rope", None)
    return cfg


# -- the multiview mode ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, layout",
    [
        (STRIP_DENSE, "three_view_strip"),
        (STRIP_FPS8, "three_view_strip"),
        (OPENWAM_FPS8, "openwam_canvas"),
        (PIXEL_DENSE, "sana_pixel_canvas"),
        (PIXEL_EEF_FPS8, "sana_pixel_canvas"),
    ],
)
def test_head_recipes_resolve_their_front_end(name, layout):
    cfg = head(name)
    assert wam_config.resolve_visual_layout(cfg) == layout
    assert wam_config.is_openwam_canvas_policy_class(cfg) is False
    policy = wam_config.policy_config_from_train_config(cfg)
    # the spatial-RoPE keys are gone from these yamls: V > 1 is the fixed semantic 2x2 (15, 30) tiling
    assert policy.multiview_spatial_rope_layout == "semantic_2x2" and policy.multiview_spatial_rope_tile_shape == (15, 30)
    assert policy.shared_prompt is False and policy.state_as_cross_attention is False


def test_legacy_yamls_keep_their_front_end():
    assert wam_config.resolve_visual_layout(legacy(LEGACY_THREE_VIEW)) == "three_view_strip"
    canvas = legacy(LEGACY_CANVAS)
    assert wam_config.resolve_visual_layout(canvas) == "openwam_canvas"
    assert wam_config.is_openwam_canvas_policy_class(canvas) is True
    assert wam_config.policy_config_from_train_config(canvas).shared_prompt is True


def test_multiview_spellings_of_every_era():
    cfg = head(STRIP_DENSE)
    for value, layout in (("sana", "three_view_strip"), ("sana_latent", "three_view_strip"), ("SANA_LATENT", "three_view_strip")):
        cfg["data"]["extra"]["multiview"] = value
        assert wam_config.resolve_visual_layout(cfg) == layout
    cfg["data"]["extra"]["multiview"] = "sana_pixelated"
    with pytest.raises(ValueError, match="unsupported data.extra.multiview"):
        wam_config.resolve_visual_layout(cfg)
    # f8eaad39a .. b31137206 spelled the knob data.extra.video_layout, and the canvas still used the canvas class then
    canvas = legacy(LEGACY_CANVAS)
    canvas["data"]["type"] = "RoboDojoSFTDataset"
    with pytest.raises(ValueError, match="needs data.type"):
        wam_config.resolve_visual_layout(canvas)
    canvas["data"]["extra"]["video_layout"] = "openwam"
    assert wam_config.resolve_visual_layout(canvas) == "openwam_canvas"
    pixel = head(PIXEL_DENSE)
    pixel["data"]["aspect_ratio_type"] = "ASPECT_RATIO_VIDEO_320_ROBOT"
    with pytest.raises(ValueError, match="ASPECT_RATIO_SANA_PIXEL_2X2_320_480"):
        wam_config.resolve_visual_layout(pixel)


# -- the RoPE contract -----------------------------------------------------------------------------------------------


def test_rope_contract_of_every_era():
    # declared (2026-09-23): the yaml's key, actions from 1 under independent
    assert wam_config.resolve_rope_contract(head(STRIP_DENSE))[:2] == ("aligned", 1)
    assert wam_config.resolve_rope_contract(head(STRIP_FPS8))[:2] == ("independent", 1)
    assert wam_config.resolve_rope_contract(head(OPENWAM_FPS8))[:2] == ("independent", 1)
    assert wam_config.resolve_rope_contract(head(PIXEL_DENSE))[:2] == ("aligned", 1)
    # undeclared: the pre-mode tables of the run's era -- no multiview key = rwm/strided_video / NSC f33fps8 (first
    # action at 0), a multiview key = the 2026-09-21..23 zekai-merge runs (first action at 1, campaign 19085339)
    assert wam_config.resolve_rope_contract(without_rope(head(STRIP_FPS8)))[:2] == (None, 0)
    assert wam_config.resolve_rope_contract(without_rope(head(OPENWAM_FPS8)))[:2] == (None, 1)
    assert wam_config.resolve_rope_contract(legacy(LEGACY_THREE_VIEW))[:2] == (None, 0)
    # overrides
    assert wam_config.resolve_rope_contract(without_rope(head(OPENWAM_FPS8)), "independent_from0")[:2] == (None, 0)
    assert wam_config.resolve_rope_contract(without_rope(head(STRIP_FPS8)), "independent")[:2] == ("independent", 1)
    with pytest.raises(ValueError, match="contradicts"):
        wam_config.resolve_rope_contract(head(STRIP_FPS8), "aligned")
    with pytest.raises(ValueError, match="rope_mode must be one of"):
        wam_config.resolve_rope_contract(head(STRIP_FPS8), "physical")
    bad = head(STRIP_FPS8)
    bad["model"]["extra"]["rope"] = "physical"
    with pytest.raises(ValueError, match="model.extra.rope"):
        wam_config.rope_mode_from_train_config(bad)
    policy = wam_config.policy_config_from_train_config(without_rope(head(OPENWAM_FPS8)))
    assert (policy.rope, policy.legacy_strided_action_origin) == (None, 1)
    assert wam_config.policy_config_from_train_config(head(STRIP_FPS8)).rope == "independent"


def test_video_frame_stride_of_the_head_recipes():
    assert wam_config.video_frame_stride_from_train_config(head(STRIP_DENSE)) == 1
    assert wam_config.video_frame_stride_from_train_config(head(PIXEL_DENSE)) == 1
    for name in (STRIP_FPS8, OPENWAM_FPS8, PIXEL_EEF_FPS8):
        assert wam_config.video_frame_stride_from_train_config(head(name)) == 4


# -- the canvas text contract ----------------------------------------------------------------------------------------


def test_canvas_text_contract_of_every_era():
    assert wam_config.resolve_canvas_text_contract(head(STRIP_DENSE))[:2] == (4, None)
    assert wam_config.resolve_canvas_text_contract(head(OPENWAM_FPS8))[:2] == (1, "two_rows")
    assert wam_config.resolve_canvas_text_contract(head(PIXEL_EEF_FPS8))[:2] == (1, "tiling")
    assert wam_config.resolve_canvas_text_contract(legacy(LEGACY_CANVAS))[:2] == (1, "composite_view")
    # campaign 19085339 (a5c7909ae): openwam on the one class, no rope key -> the G = 2 composite-view payload
    assert wam_config.resolve_canvas_text_contract(without_rope(head(OPENWAM_FPS8)))[:2] == (2, "composite_view")
    assert wam_config.resolve_canvas_text_contract(without_rope(head(PIXEL_DENSE)))[:2] == (1, "tiling")
    # overrides (the 2a10d0d75 .. 92b9e64d1 window: G = 1 with the l_shape sentence)
    assert wam_config.resolve_canvas_text_contract(without_rope(head(OPENWAM_FPS8)), 1, "l_shape")[:2] == (1, "l_shape")
    assert wam_config.resolve_canvas_text_contract(head(OPENWAM_FPS8), "2", "auto")[:2] == (2, "two_rows")
    with pytest.raises(ValueError, match="canvas_prompt"):
        wam_config.resolve_canvas_text_contract(head(PIXEL_DENSE), None, "two_rows")
    with pytest.raises(ValueError, match="canvas modes only"):
        wam_config.resolve_canvas_text_contract(head(STRIP_DENSE), 1)
    with pytest.raises(ValueError, match="ONE shared prompt"):
        wam_config.resolve_canvas_text_contract(legacy(LEGACY_CANVAS), 2)


# -- robot_base_eef: 32 slots or EEF-only ----------------------------------------------------------------------------


def test_robot_base_eef_layout_markers():
    old = legacy(LEGACY_THREE_VIEW)
    old["data"]["extra"]["action_mode_sample_ratio"] = [0.0, 1.0, 0.0]
    assert wam_config.robot_base_eef_layout_from_train_config(old) == "full"
    norm = load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256)
    assert norm.gripper_normalization is None and norm.rotation_normalization is None
    assert wam_config.robot_base_eef_layout_from_train_config(old, norm) == "full"
    # the 2026-09-20 forced scheme stamps its artifacts
    stamped = copy.copy(norm)
    object.__setattr__(stamped, "gripper_normalization", "statistics")
    assert wam_config.robot_base_eef_layout_from_train_config(old, stamped) == "eef_only"
    # HEAD recipes carry model.extra.rope (and the canvases data.extra.multiview)
    assert wam_config.robot_base_eef_layout_from_train_config(head(EEF_DELTA)) == "eef_only"
    assert wam_config.robot_base_eef_layout_from_train_config(head(PIXEL_EEF_FPS8)) == "eef_only"


def test_eef_only_sentences_drop_the_joint_clause():
    for joint_mode in ("anchor_delta", "absolute"):
        for eef_mode in ("anchor_delta", "absolute"):
            full = text.action_mode_text("robot_base_eef", joint_mode, eef_mode)
            only = text.action_mode_text("robot_base_eef", joint_mode, eef_mode, eef_only=True)
            assert "joint" in full and "joint" not in only and only.startswith("robot_base_eef; end-effector")
            # joint_only lines are untouched by the flag
            assert text.action_mode_text("joint_only", joint_mode, eef_mode, eef_only=True) == text.action_mode_text(
                "joint_only", joint_mode, eef_mode
            )


# -- the sana_pixel front-end ----------------------------------------------------------------------------------------


def test_sana_pixel_canvas_tiles_the_semantic_quadrants():
    frames = fake_frames(3)
    canvas = sana_pixel_canvas_from_frames(frames, (0, 2, 3))
    assert canvas.shape == (3, 320, 480) and canvas.dtype == torch.float32
    # the unclaimed top-right quadrant is black (-1 in the normalized domain)
    assert torch.equal(canvas[:, :160, 240:], torch.full((3, 160, 240), -1.0))
    # every claimed quadrant is the view's 320x480 resize-crop halved (a 2x2 average)
    from sana_wam_min.pixels import frame_to_model_tensor

    for frame, (top, left) in zip(frames, ((0, 0), (160, 0), (160, 240)), strict=True):
        full = frame_to_model_tensor(frame, (320, 480))
        halved = full.reshape(3, 160, 2, 240, 2).mean(dim=(2, 4))
        torch.testing.assert_close(canvas[:, top : top + 160, left : left + 240], halved, rtol=0, atol=2e-7)
    with pytest.raises(ValueError, match="unique slots"):
        assemble_sana_pixel_canvas([torch.zeros(1, 3, 2, 2)] * 2, (0, 0))


class _GridEncoder:
    """Stub causal encoder whose latent grid is the input's spatial size / 32 (the real VAE's geometry)."""

    device = torch.device("cpu")
    dtype = torch.float32

    def encode(self, video: torch.Tensor):
        frames = (video.shape[2] - 1) // 8 + 1
        pooled = torch.nn.functional.adaptive_avg_pool3d(video, (frames, video.shape[-2] // 32, video.shape[-1] // 32))
        z = torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1)
        from types import SimpleNamespace

        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))


def _grid_vae() -> VaeBundle:
    return VaeBundle(
        diffusers_vae=None,
        causal_encoder=_GridEncoder(),
        latents_mean=torch.zeros(LATENT_C),
        latents_std=torch.ones(LATENT_C),
        scaling_factor=1.0,
        temporal_compression=8,
        spatial_compression=32,
    )


def _session(cfg: dict, **kwargs) -> PolicyInferenceSession:
    cfg = copy.deepcopy(cfg)
    cfg["text_encoder"]["model_max_length"] = CAP_L
    cfg["data"]["extra"].pop("action_mode_sample_ratio", None)  # the packaged artifact is joint_only / anchor_delta
    cfg["data"]["extra"]["joint_target_mode"] = "anchor_delta"
    cfg["data"]["extra"]["eef_target_mode"] = "anchor_delta"
    cfg["data"]["num_frames"] = 25
    cfg["data"]["multi_fps"] = {"25": [25]}
    extra = cfg["data"]["extra"]
    options = extra["robot_sft"] if "robot_sft" in extra else extra["robotwin_sft"]
    options["tier_num_frames"] = 25
    if options.get("video_fps") is not None:
        options["video_fps"] = 8                     # 25 rows at video_fps 8 = stride 3, 9 frames, 2 latent frames
    return PolicyInferenceSession(
        model=StubModel(),
        vae=_grid_vae(),
        tokenizer=StubTokenizer(),
        text_encoder=StubTextEncoder(),
        normalization=load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256),
        train_config=cfg,
        device="cpu",
        steps=STEPS,
        cfg_scale=1.0,
        flow_shift=3.5,
        checkpoint_path="stub",
        **kwargs,
    )


def _predict(session: PolicyInferenceSession):
    state80, mask80 = state80_from_obs(fake_state())
    return session.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))


class _RecordingGridEncoder(_GridEncoder):
    def __init__(self):
        self.videos = []

    def encode(self, video: torch.Tensor):
        self.videos.append(video.detach().clone())
        return super().encode(video)


def test_view_resize_reaches_the_vae_and_a_masked_pad_needs_the_320x512_canvas():
    """rwm/zekai-merge 1061b16f0 / 77cf81fbf: views are stretched whole to the canvas bucket -- the default -- and a legacy
    checkpoint takes view_resize crop; the session encodes exactly that canvas (the per-view transform itself is
    byte-checked in test_view_resize_live_parity)."""

    import dataclasses

    from sana_wam_min.robodojo_io import VIEW_SLOT_IDS
    from sana_wam_min.sana_pixel_canvas import sana_pixel_canvas_from_frames

    cfg = head(PIXEL_EEF_FPS8)
    seen = {}
    for knobs in ({}, {"view_resize": "crop"}):
        session = _session(cfg, **knobs)
        recorder = _RecordingGridEncoder()
        session.vae = dataclasses.replace(session.vae, causal_encoder=recorder)
        _predict(session)
        seen[session.view_resize] = (session.view_resize_source, recorder.videos[0], session.sana_pixel_canvas_hw)
    assert (seen["stretch"][0], seen["crop"][0]) == ("default", "deploy")
    frames = [np.asarray(frame) for frame in fake_frames()]
    for mode, (_, video, canvas_hw) in seen.items():
        assert torch.equal(video[0, :, 0], sana_pixel_canvas_from_frames(frames, VIEW_SLOT_IDS, canvas_hw, view_resize=mode))
    assert not torch.equal(seen["crop"][1], seen["stretch"][1])

    declared = copy.deepcopy(cfg)
    declared["data"]["extra"]["robotwin_sft"]["view_resize"] = "stretch"     # the 1061b16f0 .. 77cf81fbf yamls
    assert (_session(declared).view_resize, _session(declared).view_resize_source) == ("stretch", "yaml")
    with pytest.raises(ValueError, match="contradicts"):
        _session(declared, view_resize="crop")
    masked = copy.deepcopy(cfg)
    masked["model"].setdefault("extra", {})["sana_pixel_pad"] = "masked"
    with pytest.raises(ValueError, match="320x512"):
        _session(masked)          # this recipe is the 320x480 canvas: a quadrant edge falls inside a latent cell


@pytest.mark.parametrize("strided", [False, True])
def test_a_sana_pixel_session_encodes_one_canvas_with_one_prompt(strided):
    cfg = head(PIXEL_EEF_FPS8 if strided else PIXEL_DENSE)
    session = _session(cfg)
    assert session.visual_layout == "sana_pixel_canvas" and session.text_groups == 1 and session.canvas_prompt == "tiling"
    result = _predict(session)
    seen = session.model.calls[0]
    assert seen["view_count"].tolist() == [1] and seen["view_slot_ids"].tolist() == [0]
    assert seen["view_latent_shape"].tolist() == [[[10, 15]]]
    assert result.video_latent.shape[-2:] == (10, 15)
    assert result.video_latent.shape[2] == (2 if strided else 4)
    assert ("video_frame_stride" in seen) is strided
    prompt = session.tokenizer.prompts[0]
    assert len(prompt) == 1 and f"Observation View: {SANA_PIXEL_TILING_TEXT}." in prompt[0]


def test_the_aspect_ratio_type_picks_the_sana_pixel_canvas():
    """zekai-merge 88a22ba0c: ASPECT_RATIO_SANA_PIXEL_2X2_320_512 = a 320x512 canvas of 160x256 tiles (10x16 latent)."""

    from sana_wam_min.sana_pixel_canvas import sana_pixel_canvas_hw, sana_pixel_tile_shape

    assert sana_pixel_canvas_hw("ASPECT_RATIO_SANA_PIXEL_2X2_320_480") == sana_pixel_canvas_hw(None) == (320, 480)
    assert sana_pixel_canvas_hw("ASPECT_RATIO_SANA_PIXEL_2X2_320_512") == (320, 512)
    assert sana_pixel_tile_shape(320, 512) == (160, 256)
    with pytest.raises(ValueError, match="aspect_ratio_type"):
        sana_pixel_canvas_hw("ASPECT_RATIO_SANA_PIXEL_2X2_320_448")
    wide = head(PIXEL_EEF_FPS8)
    wide["data"]["aspect_ratio_type"] = "ASPECT_RATIO_SANA_PIXEL_2X2_320_512"
    assert wam_config.resolve_visual_layout(wide) == "sana_pixel_canvas"
    assert wam_config.sana_pixel_canvas_hw_from_train_config(wide) == (320, 512)
    wide["data"]["aspect_ratio_type"] = "ASPECT_RATIO_SANA_PIXEL_2X2_320_448"
    with pytest.raises(ValueError, match="aspect_ratio_type"):
        wam_config.resolve_visual_layout(wide)


def test_a_320x512_sana_pixel_session_encodes_the_wide_canvas():
    cfg = head(PIXEL_EEF_FPS8)
    cfg["data"]["aspect_ratio_type"] = "ASPECT_RATIO_SANA_PIXEL_2X2_320_512"
    session = _session(cfg)
    assert session.sana_pixel_canvas_hw == (320, 512)
    result = _predict(session)
    seen = session.model.calls[0]
    assert seen["view_latent_shape"].tolist() == [[[10, 16]]] and result.video_latent.shape[-2:] == (10, 16)


def test_an_openwam_session_on_the_one_policy_class_takes_a_stride_and_its_era_prompt():
    session = _session(head(OPENWAM_FPS8))
    assert session.visual_layout == "openwam_canvas" and session.video_frame_stride == 3
    result = _predict(session)
    seen = session.model.calls[0]
    assert seen["view_latent_shape"].tolist() == [[[12, 10]]] and seen["video_frame_stride"].tolist() == [3]
    assert result.video_latent.shape[2] == 2
    rows = session.tokenizer.prompts[0]
    assert len(rows) == 1 and f"Observation View: {OPENWAM_TWO_ROWS_TEXT}." in rows[0]
    # the same checkpoint family one day earlier: G = 2, the composite-view row + the robot row
    old = _session(without_rope(head(OPENWAM_FPS8)))
    _predict(old)
    rows = old.tokenizer.prompts[0]
    assert len(rows) == 2 and f"Observation View: {OPENWAM_COMPOSITE_VIEW_TEXT}." in rows[0]
    assert "Observation View" not in rows[1] and rows[1].split("\n")[:2] == rows[0].split("\n")[:2]
    forced = _session(without_rope(head(OPENWAM_FPS8)), text_groups=1, canvas_prompt="l_shape")
    _predict(forced)
    assert forced.tokenizer.prompts[0] == [forced.tokenizer.prompts[0][0]]
    assert f"Observation View: {OPENWAM_L_SHAPE_TEXT}." in forced.tokenizer.prompts[0][0]


def test_the_eef_only_session_renders_the_eef_only_sentence():
    cfg = head(EEF_DELTA)
    session = PolicyInferenceSession(
        model=StubModel(), vae=_grid_vae(), tokenizer=StubTokenizer(), text_encoder=StubTextEncoder(),
        normalization=load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256),
        train_config=cfg, device="cpu", steps=STEPS, cfg_scale=1.0, flow_shift=3.5, checkpoint_path="stub",
    )
    assert session.robot_base_eef_layout == "eef_only"
    assert session.action_mode_text == text.action_mode_text("robot_base_eef", "anchor_delta", "anchor_delta", eef_only=True)
    full = PolicyInferenceSession(
        model=StubModel(), vae=_grid_vae(), tokenizer=StubTokenizer(), text_encoder=StubTextEncoder(),
        normalization=load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256),
        train_config=cfg, device="cpu", steps=STEPS, cfg_scale=1.0, flow_shift=3.5, checkpoint_path="stub",
        robot_base_eef_layout="full",
    )
    assert "joint motion is relative to the first state" in full.action_mode_text


def test_the_masked_affine_map_applies_the_new_gripper_statistics():
    """The 2026-09-20 artifacts normalize the grippers (center 0.5, scale 0.5); the generic masked map covers it."""

    from sana_wam_min.robot80 import denormalize_action, normalize_state

    norm = load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256)
    stamped = copy.copy(norm)
    for kind in ("state", "action"):
        mask = np.array(getattr(norm, f"{kind}_normalization_mask80"), copy=True)
        center = np.array(getattr(norm, f"{kind}_center80"), copy=True)
        scale = np.array(getattr(norm, f"{kind}_scale80"), copy=True)
        mask[[16, 45]], center[[16, 45]], scale[[16, 45]] = True, 0.5, 0.5
        object.__setattr__(stamped, f"{kind}_normalization_mask80", mask)
        object.__setattr__(stamped, f"{kind}_center80", center)
        object.__setattr__(stamped, f"{kind}_scale80", scale)
    state80, mask80 = state80_from_obs(fake_state())
    normalized = normalize_state(state80, mask80, stamped)
    torch.testing.assert_close(normalized[[16, 45]], (torch.as_tensor(state80[[16, 45]]) - 0.5) / 0.5)
    action = torch.zeros(2, ROBOT_DIM)
    action[:, 16], action[:, 45] = -1.0, 1.0
    raw = denormalize_action(action, torch.ones(2, ROBOT_DIM, dtype=torch.bool), stamped)
    assert raw[:, 16].tolist() == [0.0, 0.0] and raw[:, 45].tolist() == [1.0, 1.0]


def test_the_short_lived_video_layout_key_dispatched_the_canvas_class():
    """zekai-merge f8eaad39a .. b31137206: data.extra.video_layout openwam + the standard factory name built the rwm/openwam
    canvas class (G = 1 composite_view, state_as_cross_attention allowed); data.extra.multiview openwam builds the one
    policy class (G = 2 until 2a10d0d75)."""

    cfg = without_rope(head(OPENWAM_FPS8))
    del cfg["data"]["extra"]["multiview"]
    del cfg["data"]["extra"]["robotwin_sft"]["video_fps"]
    cfg["data"]["extra"]["video_layout"] = "openwam"
    assert wam_config.resolve_visual_layout(cfg) == "openwam_canvas"
    assert wam_config.is_openwam_canvas_policy_class(cfg) is True
    assert wam_config.resolve_canvas_text_contract(cfg)[:2] == (1, "composite_view")
    assert wam_config.policy_config_from_train_config(cfg).shared_prompt is True
    cfg["model"]["extra"]["state_as_cross_attention"] = True
    assert wam_config.state_as_cross_attention_from_train_config(cfg) is True
    one_class = without_rope(head(OPENWAM_FPS8))
    assert wam_config.is_openwam_canvas_policy_class(one_class) is False
    assert wam_config.resolve_canvas_text_contract(one_class)[:2] == (2, "composite_view")


# -- the separate action flow shift (zekai-merge 4f99eecba; the vanilla34k checkpoints of 2026-09-26) -------------------------
V34K_PIXEL = os.path.join(os.path.dirname(__file__), "fixtures", "config_vanilla34k_sana_pixel_320x512_5295c208d.yaml")
V34K_OPENWAM = os.path.join(os.path.dirname(__file__), "fixtures", "config_vanilla34k_openwam_52f3db1f0.yaml")


def test_the_vanilla34k_recipes_declare_their_own_action_schedule():
    from sana_wam_min.session import resolve_action_flow_shift

    for path, layout in ((V34K_PIXEL, "sana_pixel_canvas"), (V34K_OPENWAM, "openwam_canvas")):
        cfg = wam_config.load_train_config(path)
        assert wam_config.resolve_visual_layout(cfg) == layout
        pc = wam_config.policy_config_from_train_config(cfg, rope="aligned", legacy_origin=1)
        assert pc.separate_action_schedule and pc.rope == "aligned"
        assert resolve_action_flow_shift(cfg) == 1.0 and resolve_action_flow_shift(cfg, 2.5) == 2.5
    assert wam_config.sana_pixel_canvas_hw_from_train_config(wam_config.load_train_config(V34K_PIXEL)) == (320, 512)
    legacy = head(PIXEL_EEF_FPS8)                        # recipes before 4f99eecba share one schedule
    from sana_wam_min.session import resolve_action_flow_shift as resolve
    assert resolve(legacy) is None
    assert not wam_config.policy_config_from_train_config(legacy, rope="independent", legacy_origin=1).separate_action_schedule
    inference_only = head(PIXEL_EEF_FPS8)
    inference_only.setdefault("scheduler", {})["inference_action_flow_shift"] = 2.0
    inference_only["scheduler"]["action_flow_shift"] = 1.0
    assert resolve(inference_only) == 2.0                # inference_action_flow_shift before the training shift


# logits/sft_robodojo_vanilla48k_eefabs_f33fps8_{sana_pixel,openwam}_aligned_videoaug (zekai-merge 7070353ec, 2026-09-28): the
# vanilla34k recipes on the 48k vanilla donor plus train.extra.video_augmentation (training only). 7070353ec predates the
# 1061b16f0 whole-frame stretch, so the sana_pixel views stay centre-cropped; its model net, prompt builder and both canvas
# readers are byte-identical to 5295c208d, and its per-stream-CFG sample_policy matched ours bit for bit (2026-09-28).
V48K_PIXEL = os.path.join(os.path.dirname(__file__), "fixtures", "config_vanilla48k_videoaug_sana_pixel_320x512_7070353ec.yaml")
V48K_OPENWAM = os.path.join(os.path.dirname(__file__), "fixtures", "config_vanilla48k_videoaug_openwam_7070353ec.yaml")
TRAIN_ONLY = ("model.load_from", "data.extra.robot_sft.allow_online_encode", "train.extra.video_augmentation")


def _flat(tree, prefix=""):
    out = {}
    for key, value in (tree or {}).items():
        out.update(_flat(value, f"{prefix}{key}.") if isinstance(value, dict) else {f"{prefix}{key}": value})
    return out


def test_the_vanilla48k_videoaug_recipes_serve_like_their_vanilla34k_twins():
    from sana_wam_min.session import resolve_action_flow_shift

    for new, old, layout in ((V48K_PIXEL, V34K_PIXEL, "sana_pixel_canvas"), (V48K_OPENWAM, V34K_OPENWAM, "openwam_canvas")):
        cfg, twin = wam_config.load_train_config(new), wam_config.load_train_config(old)
        assert cfg["train"]["extra"]["video_augmentation"]["enabled"] is True
        assert "view_resize" not in cfg["data"]["extra"]["robot_sft"]            # 7070353ec: cropped views, no stretch key
        differ = {k for k in set(_flat(cfg)) | set(_flat(twin)) if _flat(cfg).get(k) != _flat(twin).get(k)}
        served = {k for k in differ if not k.startswith(TRAIN_ONLY) and k not in ("name", "work_dir")
                  and "cache_dir" not in k and "wandb" not in k and "tracker" not in k}
        assert served == set(), served
        assert wam_config.resolve_visual_layout(cfg) == layout
        pc = wam_config.policy_config_from_train_config(cfg, rope="aligned", legacy_origin=1)
        assert pc.separate_action_schedule and pc.rope == "aligned"
        assert resolve_action_flow_shift(cfg) == 1.0
    assert wam_config.sana_pixel_canvas_hw_from_train_config(wam_config.load_train_config(V48K_PIXEL)) == (320, 512)


def test_the_mirror_lockstep_follows_the_action_schedule():
    """With scheduler.action_flow_shift declared the action rows share one t of their OWN (the live
    _assert_policy_lockstep of 5e03e8f6f); without it an action t that differs from the video t is refused."""

    import dataclasses

    from _tiny_policy import tiny_inputs, tiny_policy_config
    from sana_wam_min.policy_model.model import PolicyModel

    torch.manual_seed(0)
    separate = PolicyModel(dataclasses.replace(tiny_policy_config(), separate_action_schedule=True)).eval()
    shared = PolicyModel(tiny_policy_config()).eval()
    shared.load_state_dict(separate.state_dict())
    inputs = tiny_inputs(batch=1, views=3, fps=25.0, seed=3)
    inputs["data_info"]["action_timestep"] = torch.full_like(inputs["data_info"]["action_timestep"], 123.0)
    with torch.no_grad():
        out = separate(inputs["x"], inputs["timestep"], inputs["y"], mask=inputs["mask"], data_info=dict(inputs["data_info"]))
        assert torch.isfinite(out["action_pred"]).all()
        with pytest.raises(ValueError, match="lockstep"):
            shared(inputs["x"], inputs["timestep"], inputs["y"], mask=inputs["mask"], data_info=dict(inputs["data_info"]))
        ragged = dict(inputs["data_info"])
        ragged["action_timestep"] = ragged["action_timestep"].clone()
        ragged["action_timestep"][:, 0] = 7.0                  # rows at different t stay refused
        with pytest.raises(ValueError, match="lockstep"):
            separate(inputs["x"], inputs["timestep"], inputs["y"], mask=inputs["mask"], data_info=ragged)


# -- the sana_latent strip tile (rwm/zekai-merge 6da565230) -----------------------------------------------------------

# logits/sft_robodojo_eefabs_sana_latent_256x320_vanilla34k_aligned_f33fps8 (zekai-merge 5309c7c77), copied verbatim: the
# third vanilla34k multiview mode, the sana_latent strip (three 256x320 views encoded on their own), data.extra.robot_sft
VANILLA34K_LATENT = os.path.join(os.path.dirname(__file__), "fixtures", "config_vanilla34k_sana_latent_5309c7c77.yaml")
# a detached worktree of the training tree, for the position-id parity below (skipped when absent)
SANA_5309C7C77 = os.path.expanduser("~/zekail/Sana_5309c7c77")


def test_the_sana_latent_tile_follows_the_training_tree():
    """6da565230 moved the strip's semantic tile (15, 30) -> (8, 16) with no yaml key; robot_sft configs postdate it."""

    latent = legacy(VANILLA34K_LATENT)
    assert wam_config.resolve_visual_layout(latent) == "three_view_strip"
    assert wam_config.strip_spatial_rope_tile_shape_from_train_config(latent) == (8, 16)
    assert wam_config.policy_config_from_train_config(latent).multiview_spatial_rope_tile_shape == (8, 16)
    # the b13415841 strip recipes and the legacy dumps predate the change
    assert wam_config.strip_spatial_rope_tile_shape_from_train_config(head(STRIP_DENSE)) == (15, 30)
    assert wam_config.strip_spatial_rope_tile_shape_from_train_config(legacy(LEGACY_THREE_VIEW)) == (15, 30)
    assert wam_config.policy_config_from_train_config(legacy(LEGACY_THREE_VIEW)).multiview_spatial_rope_tile_shape == (15, 30)
    # a tile the yaml spells out wins
    spelled = copy.deepcopy(latent)
    spelled["model"]["multiview_spatial_rope_tile_shape"] = [15, 30]
    assert wam_config.strip_spatial_rope_tile_shape_from_train_config(spelled) == (15, 30)


@pytest.mark.skipif(not os.path.isdir(SANA_5309C7C77), reason="needs the Sana 5309c7c77 worktree")
def test_the_sana_latent_positions_match_the_training_tree():
    import importlib.util

    from sana_wam_min.policy_model.rope import semantic_2x2_position_ids

    spec = importlib.util.spec_from_file_location(
        "live_multiview_utils", os.path.join(SANA_5309C7C77, "dev/rwm/diffusion/multiview_utils.py")
    )
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    assert tuple(live.MULTIVIEW_SPATIAL_ROPE_TILE_SHAPE) == wam_config.STRIP_SPATIAL_ROPE_TILE_SHAPE
    kwargs = dict(frames=2, view_shapes=[(8, 10)] * 3, view_slot_ids=[0, 2, 3], fps=torch.tensor([8.0]), base_fps=25.0,
                  tile_shape=(8, 16), device=torch.device("cpu"))
    ours, theirs = semantic_2x2_position_ids(**kwargs), live.semantic_2x2_position_ids(**kwargs)
    assert ours.dtype == theirs.dtype and torch.equal(ours, theirs)
    # an 8 x 10 view is centred at x + 3 in its 8 x 16 quadrant
    x = ours[..., 2].reshape(-1) if ours.shape[-1] == 3 else None
    if x is not None:
        assert float(x.min()) == 3.0


# logits/sft_robodojo_eefabs_sana_pixel_320x512_vanilla34k_aligned_f33fps8_fixed_resize[_padmask] (zekai-merge 2ee450e05):
# the vanilla34k sana_pixel recipe with stretched views (declared: the 1061b16f0 .. 77cf81fbf key era), the black quadrant
# unmasked / masked, and the f33 normalization 9f8b98e; everything else the adapter serves is the crop twin's
V34K_STRETCH = os.path.join(os.path.dirname(__file__), "fixtures", "config_vanilla34k_sana_pixel_320x512_stretch_2ee450e05.yaml")
V34K_STRETCH_PADMASK = os.path.join(
    os.path.dirname(__file__), "fixtures", "config_vanilla34k_sana_pixel_320x512_stretch_padmask_2ee450e05.yaml"
)


def test_the_vanilla34k_stretch_pair_serves_like_its_crop_twin_but_resize_pad_and_norm():
    import dataclasses

    from sana_wam_min.session import resolve_action_flow_shift

    twin = wam_config.load_train_config(V34K_PIXEL)
    twin_policy = wam_config.policy_config_from_train_config(twin)
    for path, pad in ((V34K_STRETCH, "unmasked"), (V34K_STRETCH_PADMASK, "masked")):
        cfg = wam_config.load_train_config(path)
        assert wam_config.view_resize_from_train_config(cfg) == ("stretch", "yaml")
        assert wam_config.sana_pixel_pad_from_train_config(cfg) == pad
        assert wam_config.resolve_visual_layout(cfg) == "sana_pixel_canvas"
        assert wam_config.sana_pixel_canvas_hw_from_train_config(cfg) == (320, 512)
        assert wam_config.policy_config_from_train_config(cfg) == dataclasses.replace(twin_policy, sana_pixel_pad=pad)
        assert wam_config.resolve_canvas_text_contract(cfg) == wam_config.resolve_canvas_text_contract(twin)
        assert (wam_config.video_fps_from_train_config(cfg), resolve_action_flow_shift(cfg)) == (8, 1.0)
        assert cfg["data"]["extra"]["robot_sft"]["normalization_sha256"].startswith("9f8b98ed")
        session = _session(cfg)
        assert (session.sana_pixel_pad, session.view_resize, session.view_resize_source) == (pad, "stretch", "yaml")
        assert tuple(session.sana_pixel_canvas_hw) == (320, 512)


# logits/sft_robodojo_eefabs_sana_pixel_320x512_vanilla52k_aligned_f33fps8_fixed_resize[_padmask] (zekai-merge 27ed33fc6,
# donor sana_rwm_pretrained_vanilla_e10s52575): trained after 77cf81fbf retired robot_sft.view_resize, so the yaml carries no
# key and stretch comes from the adapter's default; every other served key is the vanilla34k stretch pair's
V52K_STRETCH = os.path.join(os.path.dirname(__file__), "fixtures", "config_vanilla52k_sana_pixel_320x512_stretch_27ed33fc6.yaml")
V52K_STRETCH_PADMASK = os.path.join(
    os.path.dirname(__file__), "fixtures", "config_vanilla52k_sana_pixel_320x512_stretch_padmask_27ed33fc6.yaml"
)


def _flat(tree, prefix=""):
    if isinstance(tree, dict):
        out = {}
        for key, value in tree.items():
            out.update(_flat(value, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: tree}


def test_the_vanilla52k_stretch_pair_serves_like_the_vanilla34k_pair_with_the_default_stretch():
    from sana_wam_min.session import resolve_action_flow_shift

    for path, twin_path, pad in ((V52K_STRETCH, V34K_STRETCH, "unmasked"), (V52K_STRETCH_PADMASK, V34K_STRETCH_PADMASK, "masked")):
        cfg, twin = wam_config.load_train_config(path), wam_config.load_train_config(twin_path)
        flat, twin_flat = _flat(cfg), _flat(twin)
        differ = {key for key in flat.keys() | twin_flat.keys() if flat.get(key) != twin_flat.get(key)}
        assert {key.rsplit(".", 1)[-1] for key in differ} == {"name", "work_dir", "load_from", "valid_prompt_embed_root", "view_resize"}
        assert "view_resize" not in cfg["data"]["extra"]["robot_sft"] and "e10s52575" in flat["model.load_from"]
        assert wam_config.view_resize_from_train_config(cfg) == ("stretch", "default")
        assert wam_config.sana_pixel_pad_from_train_config(cfg) == pad
        assert wam_config.resolve_visual_layout(cfg) == "sana_pixel_canvas"
        assert wam_config.sana_pixel_canvas_hw_from_train_config(cfg) == (320, 512)
        assert wam_config.policy_config_from_train_config(cfg) == wam_config.policy_config_from_train_config(twin)
        assert wam_config.resolve_canvas_text_contract(cfg) == wam_config.resolve_canvas_text_contract(twin)
        assert (wam_config.video_fps_from_train_config(cfg), resolve_action_flow_shift(cfg)) == (8, 1.0)
        assert cfg["data"]["extra"]["robot_sft"]["normalization_sha256"].startswith("9f8b98ed")
        session = _session(cfg)
        assert (session.sana_pixel_pad, session.view_resize, session.view_resize_source) == (pad, "stretch", "default")
        assert tuple(session.sana_pixel_canvas_hw) == (320, 512)


# logits/sft_robodojo_eefabs_sana_latent_256x320_vanilla34k_aligned_f33fps8_fixed_resize (zekai-merge 87e34d163): the
# vanilla34k sana_latent strip retrained on STRETCHED views (after 77cf81fbf: no view_resize key, the adapter's default), with
# the f33 normalization 1fe3b7e7 (numerically = 9f8b98e); a newer trainer dumps every default, so its yaml is longer
V34K_LATENT_STRETCH = os.path.join(os.path.dirname(__file__), "fixtures", "config_vanilla34k_sana_latent_stretch_87e34d163.yaml")
# logits/sft_robodojo_eefabs_sana_latent_256x320_vanilla48k_aligned_f33fps8_videoaug (zekai-merge 5309c7c77, like the crop
# vanilla34k strip): the 48k vanilla donor and a train-only video augmentation; 5309c7c77 predates 1061b16f0, its views
# were CROPPED (user 2026-10-01: "的确是crop的，48k"; the worker serves the label crop from eval/pre_stretch_labels.txt)
V48K_LATENT_VIDEOAUG = os.path.join(os.path.dirname(__file__), "fixtures", "config_vanilla48k_sana_latent_videoaug_5309c7c77.yaml")


def _assert_serves_like_the_vanilla34k_strip(cfg: dict, twin: dict) -> None:
    from sana_wam_min.session import resolve_action_flow_shift

    assert wam_config.resolve_visual_layout(cfg) == "three_view_strip"
    assert wam_config.strip_spatial_rope_tile_shape_from_train_config(cfg) == (8, 16)
    assert wam_config.policy_config_from_train_config(cfg) == wam_config.policy_config_from_train_config(twin)
    assert wam_config.resolve_canvas_text_contract(cfg) == wam_config.resolve_canvas_text_contract(twin)
    assert (wam_config.video_fps_from_train_config(cfg), resolve_action_flow_shift(cfg)) == (8, 1.0)
    assert wam_config.view_resize_from_train_config(cfg) == ("stretch", "default")


def test_the_vanilla34k_sana_latent_stretch_serves_like_its_crop_twin_but_resize_and_norm():
    cfg, twin = legacy(V34K_LATENT_STRETCH), legacy(VANILLA34K_LATENT)
    _assert_serves_like_the_vanilla34k_strip(cfg, twin)
    assert "view_resize" not in cfg["data"]["extra"]["robot_sft"]
    assert cfg["data"]["extra"]["robot_sft"]["normalization_sha256"].startswith("1fe3b7e7")
    session = _session(cfg)
    assert (session.view_resize, session.view_resize_source) == ("stretch", "default")


def test_the_vanilla48k_sana_latent_videoaug_serves_like_the_crop_vanilla34k_strip():
    cfg, twin = legacy(V48K_LATENT_VIDEOAUG), legacy(VANILLA34K_LATENT)
    flat, twin_flat = _flat(cfg), _flat(twin)
    differ = {key for key in flat.keys() | twin_flat.keys() if flat.get(key) != twin_flat.get(key)}
    training_only = {"name", "work_dir", "model.load_from", "train.valid_prompt_embed_root", "vae.cache_dir", "vae.vae_pretrained",
                     "data.extra.robot_sft.allow_online_encode", "data.extra.robot_sft.vae_latent_cache_dir"}
    assert differ - training_only == {key for key in differ if key.startswith("train.extra.video_augmentation.")}
    assert "e9s48000" in flat["model.load_from"] and flat["train.extra.video_augmentation.enabled"] is True
    _assert_serves_like_the_vanilla34k_strip(cfg, twin)
    assert cfg["data"]["extra"]["robot_sft"]["normalization_sha256"].startswith("9f8b98ed")
    session = _session(cfg, view_resize="crop")         # what the worker passes for a label of eval/pre_stretch_labels.txt
    assert (session.view_resize, session.view_resize_source) == ("crop", "deploy")


# logits/sft_robodojo_eefabs_sana_latent_256x320_pantheon2k_aligned_f33fps8_{fixed_resize,videoaug}: the sana_latent strip on
# the Pantheon stage of the vanilla pretrain (sana_rwm_pretrained_vanilla_pantheon_e1s2000), both on STRETCHED views (27ed33fc6 /
# 611c0d702, after 77cf81fbf): the fixed_resize one off the stretch latent store with norm 9f8b98e, the videoaug one encoded
# online with a train-only video augmentation and norm 1fe3b7e7
PANTHEON2K_LATENT = os.path.join(os.path.dirname(__file__), "fixtures", "config_pantheon2k_sana_latent_stretch_27ed33fc6.yaml")
PANTHEON2K_LATENT_VIDEOAUG = os.path.join(os.path.dirname(__file__), "fixtures", "config_pantheon2k_sana_latent_videoaug_611c0d702.yaml")


@pytest.mark.parametrize("path, norm, augmented", [(PANTHEON2K_LATENT, "9f8b98ed", False), (PANTHEON2K_LATENT_VIDEOAUG, "1fe3b7e7", True)])
def test_the_pantheon2k_sana_latent_pair_serves_like_the_vanilla34k_strip_with_the_default_stretch(path, norm, augmented):
    cfg, twin = legacy(path), legacy(VANILLA34K_LATENT)
    _assert_serves_like_the_vanilla34k_strip(cfg, twin)
    assert "view_resize" not in cfg["data"]["extra"]["robot_sft"]
    assert cfg["data"]["extra"]["robot_sft"]["normalization_sha256"].startswith(norm)
    assert "pantheon_e1s2000" in cfg["model"]["load_from"]
    assert bool(((cfg.get("train") or {}).get("extra") or {}).get("video_augmentation", {}).get("enabled")) is augmented
    session = _session(cfg)
    assert (session.view_resize, session.view_resize_source) == ("stretch", "default")


# logits/sft_robodojo_eefabs_sana_pixel_320x512_vanilla_pantheon14k_aligned_f33fps8 (zekai-merge 27ed33fc6, like the vanilla52k
# pair): the sana_pixel 320x512 canvas, pad unmasked, STRETCHED views (no view_resize key: the adapter's default), norm 9f8b98e,
# from the Sana-pixel Pantheon donor sana_rwm_pretrained_vanilla_sanapixel_pantheon_e4s14173; only the donor and the run's own
# names differ from the vanilla52k unmasked yaml
PANTHEON14K_PIXEL = os.path.join(os.path.dirname(__file__), "fixtures", "config_pantheon14k_sana_pixel_320x512_stretch_27ed33fc6.yaml")


def test_the_pantheon14k_sana_pixel_serves_like_the_vanilla52k_unmasked_stretch():
    cfg, twin = wam_config.load_train_config(PANTHEON14K_PIXEL), wam_config.load_train_config(V52K_STRETCH)
    flat, twin_flat = _flat(cfg), _flat(twin)
    differ = {key for key in flat.keys() | twin_flat.keys() if flat.get(key) != twin_flat.get(key)}
    assert differ == {"name", "work_dir", "model.load_from", "train.valid_prompt_embed_root"}
    assert "sanapixel_pantheon_e4s14173" in flat["model.load_from"]
    assert "view_resize" not in cfg["data"]["extra"]["robot_sft"]
    assert wam_config.view_resize_from_train_config(cfg) == ("stretch", "default")
    assert wam_config.sana_pixel_pad_from_train_config(cfg) == "unmasked"
    assert wam_config.resolve_visual_layout(cfg) == "sana_pixel_canvas"
    assert wam_config.sana_pixel_canvas_hw_from_train_config(cfg) == (320, 512)
    assert wam_config.policy_config_from_train_config(cfg) == wam_config.policy_config_from_train_config(twin)
    assert wam_config.resolve_canvas_text_contract(cfg) == wam_config.resolve_canvas_text_contract(twin)
    assert cfg["data"]["extra"]["robot_sft"]["normalization_sha256"].startswith("9f8b98ed")
    session = _session(cfg)
    assert (session.sana_pixel_pad, session.view_resize, session.view_resize_source) == ("unmasked", "stretch", "default")
    assert tuple(session.sana_pixel_canvas_hw) == (320, 512)


# logits/sft_robodojo_eefabs_sana_latent_256x320_vanilla_pantheon33k_aligned_f33fps8 (zekai-merge 27ed33fc6, the same campaign as
# the pantheon14k pixel): the sana_latent strip on STRETCHED views (no view_resize key: the adapter's default) off the stretch latent
# store, norm 9f8b98e, no augmentation, from the Sana-latent Pantheon donor sana_rwm_pretrained_vanilla_sanalatent_pantheon_e8s33139;
# only the donor and the run's own names differ from the Pantheon2k fixed_resize strip yaml
PANTHEON33K_LATENT = os.path.join(os.path.dirname(__file__), "fixtures", "config_pantheon33k_sana_latent_stretch_27ed33fc6.yaml")


def test_the_pantheon33k_sana_latent_serves_like_the_pantheon2k_stretch_strip():
    cfg, twin = legacy(PANTHEON33K_LATENT), legacy(PANTHEON2K_LATENT)
    flat, twin_flat = _flat(cfg), _flat(twin)
    differ = {key for key in flat.keys() | twin_flat.keys() if flat.get(key) != twin_flat.get(key)}
    assert differ == {"name", "work_dir", "model.load_from", "train.valid_prompt_embed_root"}
    assert "sanalatent_pantheon_e8s33139" in flat["model.load_from"]
    _assert_serves_like_the_vanilla34k_strip(cfg, legacy(VANILLA34K_LATENT))
    assert "view_resize" not in cfg["data"]["extra"]["robot_sft"]
    assert cfg["data"]["extra"]["robot_sft"]["normalization_sha256"].startswith("9f8b98ed")
    assert not ((cfg.get("train") or {}).get("extra") or {}).get("video_augmentation")
    session = _session(cfg)
    assert (session.view_resize, session.view_resize_source) == ("stretch", "default")
