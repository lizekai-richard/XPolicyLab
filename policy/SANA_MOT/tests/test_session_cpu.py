"""CPU tests for sana_mot_min.session: normalization resolution, the two observation contracts, the strided-video contract."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from _tiny_mot import CONFIG_CANVAS, CONFIG_F33FPS8, CONFIG_MULTIVIEW, CONFIG_NSC_F25, CONFIG_NSC_F25_SAC  # noqa: E402

from sana_mot_min import session as session_module  # noqa: E402
from sana_mot_min.canvas import COMPOSITE_VIEW_KEY  # noqa: E402
from sana_mot_min.config import load_train_config, strided_video_fps  # noqa: E402
from sana_mot_min.prompt import render_canvas_prompt_rows, render_multiview_prompt_rows  # noqa: E402
from sana_mot_min.session import (  # noqa: E402
    PACKAGED_NORMALIZATION_PATH,
    TRAINING_NORMALIZATION_SHA256,
    MoTInferenceSession,
    PredictResult,
    load_checked_normalization,
    video_frame_stride,
    resolve_normalization_path,
)
from sana_wam_min.robodojo_io import JOINT_ONLY_ACTIVE_SLOTS, ROBOT80_JOINT_SLOTS_12, state80_from_obs  # noqa: E402
from sana_wam_min.robot80 import denormalize_action, load_normalization, normalize_state  # noqa: E402
from sana_wam_min.session import training_normalization_pin  # noqa: E402
from sana_wam_min.vae import VaeBundle  # noqa: E402

LATENT_C = 4
CAP_C, CAP_L = 16, 12
STEPS = 3


class StubEncoder:
    """Deterministic pixel -> [1, 4, 1, H/32, W/32] latent stand-in for the causal VAE encoder."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.inputs: list[torch.Tensor] = []

    def encode(self, video: torch.Tensor):
        self.inputs.append(video.detach().clone())
        pooled = torch.nn.functional.adaptive_avg_pool3d(video, (1, video.shape[-2] // 32, video.shape[-1] // 32))
        z = torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))


def stub_vae() -> VaeBundle:
    return VaeBundle(
        diffusers_vae=None,
        causal_encoder=StubEncoder(),
        latents_mean=torch.zeros(LATENT_C),
        latents_std=torch.ones(LATENT_C),
        scaling_factor=1.0,
        temporal_compression=8,
        spatial_compression=32,
    )


class StubTokens:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return self


class StubTokenizer:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, prompts, max_length, padding, truncation, return_tensors):
        self.calls.append(list(prompts))
        ids = torch.zeros(len(prompts), max_length, dtype=torch.int64)
        mask = torch.zeros(len(prompts), max_length, dtype=torch.int64)
        for row, prompt in enumerate(prompts):
            codes = [1] + [min(ord(ch), 255) + 2 for ch in prompt][: max_length - 1]
            ids[row, : len(codes)] = torch.tensor(codes)
            mask[row, : len(codes)] = 1
        return StubTokens(ids, mask)


class StubTextEncoder:
    def __call__(self, input_ids, attention_mask=None):
        hidden = (input_ids.to(torch.float32) / 257.0)[..., None].expand(*input_ids.shape, CAP_C)
        return (hidden.contiguous(),)


class StubModel:
    """Records every forward's inputs; constant velocities keep the Euler loop deterministic."""

    def __init__(self, video_layout: str = "multiview"):
        self.video_layout = video_layout
        self.calls: list[dict] = []

    def __call__(self, x, timestep, y, mask=None, data_info=None):
        self.calls.append(
            {
                "x_shape": tuple(x.shape),
                "y_shape": tuple(y.shape),
                "mask_shape": tuple(mask.shape),
                "data_info": {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in data_info.items()},
            }
        )
        assert timestep.shape == (1, 1, x.shape[2])
        return {"x": torch.zeros_like(x), "action_pred": torch.full_like(data_info["action80"], 0.5)}


def make_session(config_path: str = CONFIG_MULTIVIEW, cfg_scale: float = 1.0, **knobs) -> MoTInferenceSession:
    train_cfg = load_train_config(config_path)
    train_cfg["text_encoder"]["model_max_length"] = CAP_L
    layout = (train_cfg["model"].get("extra") or {}).get("video_layout", "multiview")
    return MoTInferenceSession(
        model=StubModel(layout),
        vae=stub_vae(),
        tokenizer=StubTokenizer(),
        text_encoder=StubTextEncoder(),
        normalization=load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256),
        train_config=train_cfg,
        device="cpu",
        steps=STEPS,
        cfg_scale=cfg_scale,
        flow_shift=3.5,
        checkpoint_path="stub",
        **knobs,
    )


def fake_frames(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(3)]


def fake_state() -> dict:
    return {
        "left_arm_joint_state": np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32),
        "left_ee_joint_state": np.array([0.25], dtype=np.float32),
        "right_arm_joint_state": np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6], dtype=np.float32),
        "right_ee_joint_state": np.array([1.0], dtype=np.float32),
    }


def _assert_absolute_rows(result, session, state80):
    """Absolute targets: the raw row is the denormalized model row (grippers clipped) -- NO anchor is added."""

    expected = denormalize_action(result.action80_model, result.action_mask, session.normalization)
    expected[:, [16, 45]] = expected[:, [16, 45]].clamp(0, 1)
    assert torch.allclose(result.action80_raw_absolute, expected, atol=1e-6)
    joints = list(ROBOT80_JOINT_SLOTS_12)
    with_anchor = expected.clone()
    with_anchor[:, joints] += torch.as_tensor(state80)[joints]
    assert not torch.allclose(result.action80_raw_absolute, with_anchor, atol=1e-3)
    inactive = ~result.action_mask
    assert torch.all(result.action80_model[inactive] == 0) and torch.all(result.action80_raw_absolute[inactive] == 0)


def test_no_sana_imports_in_the_package():
    package_dir = os.path.dirname(session_module.__file__)
    for root, _, files in os.walk(package_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            with open(os.path.join(root, name)) as handle:
                for line in handle.read().splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")):
                        assert not any(tok in stripped for tok in ("dev.", "diffusion.", " sana ", "tqdm")), f"{name}: {stripped}"


def test_packaged_normalization_is_the_local_absolute_artifact():
    norm = load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256)
    assert norm.sha256 == TRAINING_NORMALIZATION_SHA256 and norm.joint_target_mode == "absolute"
    assert norm.num_frames == 25 and norm.model_fps == 25
    for path in (CONFIG_MULTIVIEW, CONFIG_CANVAS):
        assert load_train_config(path)["data"]["extra"]["robotwin_sft"]["normalization_sha256"] == TRAINING_NORMALIZATION_SHA256
    active = np.flatnonzero(norm.action_normalization_mask80).tolist()
    assert all(slot in active for slot in ROBOT80_JOINT_SLOTS_12) and 16 not in active and 45 not in active


def test_normalization_resolution_and_cross_checks(tmp_path):
    train_cfg = load_train_config(CONFIG_MULTIVIEW)
    ckpt = tmp_path / "ckpt"
    (ckpt / "model").mkdir(parents=True)
    assert resolve_normalization_path(ckpt, None) == PACKAGED_NORMALIZATION_PATH
    assert load_checked_normalization(ckpt, train_cfg).sha256 == TRAINING_NORMALIZATION_SHA256
    (ckpt / "normalization").mkdir()
    canonical = ckpt / "normalization" / PACKAGED_NORMALIZATION_PATH.name
    shutil.copy(PACKAGED_NORMALIZATION_PATH, canonical)
    assert resolve_normalization_path(ckpt, None) == canonical
    with pytest.raises(ValueError):
        load_checked_normalization(ckpt, train_cfg, expected_sha256="0" * 64)
    from sana_wam_min.session import PACKAGED_NORMALIZATION_PATH as WAM_PATH

    # an anchor-delta artifact serving an absolute line hands its STATE statistics to the action joint slots (Sana's
    # select_joint_target_normalization, applied by every dataset); the reverse direction is refused
    reselected = load_checked_normalization(ckpt, train_cfg, normalization_path=WAM_PATH)
    assert (reselected.joint_target_mode, reselected.artifact_joint_target_mode) == ("absolute", "anchor_delta")
    delta_cfg = copy.deepcopy(train_cfg)
    delta_cfg["data"]["extra"]["joint_target_mode"] = "anchor_delta"
    with pytest.raises(ValueError, match="joint_target_mode"):
        load_checked_normalization(ckpt, delta_cfg)
    # the NSC run pins its own artifact (3cd3ce1d...): the packaged local-cluster copy is refused as a fallback
    nsc = load_train_config(CONFIG_NSC_F25)
    with pytest.raises(ValueError, match="not the one the training yaml pins"):
        load_checked_normalization(tmp_path / "nsc", nsc)
    payload = json.loads(PACKAGED_NORMALIZATION_PATH.read_text())
    payload["num_frames"] = 33
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="num_frames"):
        load_checked_normalization(ckpt, train_cfg, normalization_path=bad)


def test_strided_video_config_derives_the_frame_stride():
    # f33fps8: 33-row windows, 8 video frames after the observation frame -> stride 4 (9 video frames, 2 latent).
    assert video_frame_stride(load_train_config(CONFIG_F33FPS8)) == 4
    assert video_frame_stride(load_train_config(CONFIG_MULTIVIEW)) == 1

    strided = make_session(CONFIG_F33FPS8)
    assert (strided.video_fps, strided.video_frame_stride) == (8, 4)
    dense = make_session(CONFIG_MULTIVIEW)
    assert (dense.video_fps, dense.video_frame_stride) == (None, 1)


def test_a_renamed_sft_block_keeps_the_stride_and_the_normalization_pin(tmp_path):
    """Sana 875b3f659 (2026-09-24) renamed data.extra.robotwin_sft -> robot_sft (same options); configs frozen after it
    use the new name, older ones keep the retired one. Both must give the same stride and yaml normalization pin."""

    old = load_train_config(CONFIG_F33FPS8)
    new = copy.deepcopy(old)
    new["data"]["extra"]["robot_sft"] = new["data"]["extra"].pop("robotwin_sft")
    assert video_frame_stride(new) == video_frame_stride(old) == 4
    assert strided_video_fps(new) == strided_video_fps(old) == 8
    assert training_normalization_pin(new) == training_normalization_pin(old) is not None
    renamed = tmp_path / "config.yaml"
    renamed.write_text(yaml.safe_dump(new))
    session = make_session(str(renamed))
    assert (session.video_fps, session.video_frame_stride) == (8, 4)


def test_strided_predict_uses_the_short_window_and_publishes_the_stride():
    """f33fps8: 33-row windows at stride 4 -> 9 video frames -> 2 latent frames, while the 32 action rows stay dense."""

    session = make_session(CONFIG_F33FPS8)
    frames = fake_frames()
    state80, mask80 = state80_from_obs(fake_state())
    result = session.predict(frames, state80, mask80, "Stack the three bowls together.", torch.Generator().manual_seed(7))

    assert result.receipt["frames"] == 33 and result.receipt["video_frames"] == 9
    assert result.receipt["video_frame_stride"] == 4 and result.receipt["k_actions"] == 32
    assert result.action80_raw_absolute.shape == (32, 80)
    assert result.video_latent.shape == (1, LATENT_C, 2, 1, 3 * 8 * 10)          # 2 latent frames, not the dense 5
    for call in session.model.calls:
        assert call["x_shape"] == (1, LATENT_C, 2, 1, 240)
        assert call["data_info"]["video_frame_stride"].tolist() == [4]
        # the port's own contract: (F - 1) * 8 * stride action rows
        assert call["data_info"]["action80"].shape[1] == (2 - 1) * 8 * 4
    # a dense checkpoint's batch carries no stride key at all
    dense = make_session(CONFIG_MULTIVIEW)
    dense.predict(frames, state80, mask80, "Stack the three bowls together.", torch.Generator().manual_seed(7))
    assert all("video_frame_stride" not in call["data_info"] for call in dense.model.calls)


def test_multiview_predict_end_to_end_with_stubs():
    session = make_session(CONFIG_MULTIVIEW)
    assert session.video_layout == "multiview" and not session.canvas
    frames = fake_frames()
    state80, mask80 = state80_from_obs(fake_state())
    instruction = "Stack the three bowls together."
    result = session.predict(frames, state80, mask80, instruction, torch.Generator().manual_seed(7))

    assert isinstance(result, PredictResult)
    assert result.action80_raw_absolute.shape == (24, 80) and result.action80_model.shape == (24, 80)
    assert result.video_latent.shape == (1, LATENT_C, 4, 1, 3 * 8 * 10)          # the packed strip of 3 x [8, 10]
    assert result.receipt["video_layout"] == "multiview" and result.receipt["view_latent_shapes"] == ((8, 10),) * 3
    assert result.receipt["joint_target_mode"] == "absolute" and result.receipt["k_actions"] == 24
    # every view encoded on its own through the training transform (ResizeCrop to 256x320)
    encoded = session.vae.causal_encoder.inputs
    assert [tuple(v.shape) for v in encoded] == [(1, 3, 1, 256, 320)] * 3
    assert all(v.min() >= -1 and v.max() <= 1 for v in encoded)
    calls = session.model.calls
    assert len(calls) == STEPS
    for call in calls:
        assert call["x_shape"] == (1, LATENT_C, 4, 1, 240)
        assert call["y_shape"] == (1, 4, 1, CAP_L, CAP_C) and call["mask_shape"] == (1, 4, 1, 1, CAP_L)
        info = call["data_info"]
        assert int(info["view_count"]) == 3 and info["view_slot_ids"].tolist() == [0, 2, 3]
        assert info["view_latent_shape"].tolist() == [[[8, 10]] * 3] and info["camera_conditioning_enabled"] is False
        assert info["initial_state_condition_mask80"][0].nonzero().flatten().tolist() == list(JOINT_ONLY_ACTIVE_SLOTS)
    assert session.tokenizer.calls == [list(render_multiview_prompt_rows(instruction))]
    _assert_absolute_rows(result, session, state80)
    session.predict(frames, state80, mask80, instruction, torch.Generator().manual_seed(8))
    assert len(session.tokenizer.calls) == 1                                     # prompt cache


def test_canvas_predict_end_to_end_with_stubs():
    session = make_session(CONFIG_CANVAS)
    assert session.video_layout == "openwam_canvas" and session.canvas
    frames = fake_frames(1)
    state80, mask80 = state80_from_obs(fake_state())
    instruction = "Put the bottle into the bin."
    result = session.predict(frames, state80, mask80, instruction, torch.Generator().manual_seed(9))
    assert result.video_latent.shape == (1, LATENT_C, 4, 12, 10)
    assert result.receipt["view_latent_shapes"] == ((12, 10),)
    encoded = session.vae.causal_encoder.inputs
    assert [tuple(v.shape) for v in encoded] == [(1, 3, 1, 384, 320)]
    for call in session.model.calls:
        assert call["y_shape"] == (1, 1, 1, CAP_L, CAP_C)
        info = call["data_info"]
        assert int(info["view_count"]) == 1 and info["view_keys"] == [COMPOSITE_VIEW_KEY] and "view_slot_ids" not in info
    assert session.tokenizer.calls == [list(render_canvas_prompt_rows(instruction))]
    _assert_absolute_rows(result, session, state80)


def test_layouts_refuse_the_wrong_view_count():
    multiview, canvas = make_session(CONFIG_MULTIVIEW), make_session(CONFIG_CANVAS)
    state = torch.zeros(80)
    action = torch.zeros(1, 24, 80)
    mask = torch.zeros(1, 24, 80, dtype=torch.bool)
    with pytest.raises(ValueError, match="expects 3"):
        multiview.build_data_info(((12, 10),), state, state.bool(), action, mask)
    with pytest.raises(ValueError, match="expects 1"):
        canvas.build_data_info(((8, 10),) * 3, state, state.bool(), action, mask)
    with pytest.raises(ValueError, match="expected 3 views"):
        multiview.encode_observation(fake_frames()[:2], 4)


def test_cfg_encodes_the_unconditional_rows_in_the_same_batch():
    session = make_session(CONFIG_MULTIVIEW, cfg_scale=6.0, action_cfg_scale=1.0)
    assert session.uses_cfg and (session.video_cfg_scale, session.action_cfg_scale) == (6.0, 1.0)
    y, mask, y_uncond, mask_uncond = session.encode_instruction("Stack the bowls.")
    assert y.shape == (1, 4, 1, CAP_L, CAP_C) and y_uncond.shape == y.shape and mask_uncond.shape == mask.shape
    rows = session.tokenizer.calls[-1]
    assert len(rows) == 8 and all("Instruction" not in row for row in rows[4:])
    plain = make_session(CONFIG_CANVAS, cfg_scale=1.0)
    assert plain.encode_instruction("Stack the bowls.")[2] is None and len(plain.tokenizer.calls[-1]) == 1


def test_predict_from_latent_replays_the_sampler():
    session = make_session(CONFIG_MULTIVIEW)
    window = torch.zeros(1, LATENT_C, 4, 1, 240)
    state = torch.zeros(80)
    mask = torch.zeros(1, 24, 80, dtype=torch.bool)
    mask[:, :, :6] = True
    rows = render_multiview_prompt_rows("Stack the bowls.")
    action, video = session.predict_from_latent(
        window, ((8, 10),) * 3, rows, None, state, state.bool(), torch.zeros(1, 24, 80), mask, generator=torch.Generator().manual_seed(1)
    )
    assert action.shape == (1, 24, 80) and video.shape == window.shape
    assert int(session.model.calls[-1]["data_info"]["view_count"]) == 3


def test_the_session_clamps_the_gripper_in_the_model_domain_of_its_artifact(monkeypatch):
    """2026-09-20 artifacts normalize the grippers (center 0.5, scale 0.5): the sampler's clamp must be the model-domain
    image of raw closedness [0, 1], i.e. [-1, 1], not Sana's [0, 1] (which capped every commanded opening at 0.5)."""

    import copy

    from sana_mot_min import session as mot_session

    seen = []
    real = mot_session.sample_policy

    def spy(*args, **kwargs):
        seen.append(kwargs.get("gripper_bounds"))
        return real(*args, **kwargs)

    monkeypatch.setattr(mot_session, "sample_policy", spy)
    state80, mask80 = state80_from_obs(fake_state())
    session = make_session()
    session.predict(fake_frames(), state80, mask80, "Stack the bowls.", torch.Generator().manual_seed(1))
    stamped = copy.copy(session.normalization)
    for kind in ("state", "action"):
        mask = np.array(getattr(stamped, f"{kind}_normalization_mask80"), copy=True)
        center = np.array(getattr(stamped, f"{kind}_center80"), copy=True)
        scale = np.array(getattr(stamped, f"{kind}_scale80"), copy=True)
        mask[[16, 45]], center[[16, 45]], scale[[16, 45]] = True, 0.5, 0.5
        object.__setattr__(stamped, f"{kind}_normalization_mask80", mask)
        object.__setattr__(stamped, f"{kind}_center80", center)
        object.__setattr__(stamped, f"{kind}_scale80", scale)
    new = make_session()
    new.normalization = stamped
    new.predict(fake_frames(), state80, mask80, "Stack the bowls.", torch.Generator().manual_seed(1))
    assert seen == [((0.0, 1.0), (0.0, 1.0)), ((-1.0, 1.0), (-1.0, 1.0))]
