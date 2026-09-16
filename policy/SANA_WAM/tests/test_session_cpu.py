"""CPU tests for sana_wam_min.session: path resolution, data_info assembly, noise order, predict with stubs."""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_SANA_WAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(_SANA_WAM_DIR, "..", "..", ".."))
for _p in (_SANA_WAM_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sana_wam_min import session as session_module  # noqa: E402
from sana_wam_min.config import load_train_config  # noqa: E402
from sana_wam_min.multiview import pack_multiview_latents  # noqa: E402
from sana_wam_min.pixels import frame_to_model_tensor, frames_to_vae_input  # noqa: E402
from sana_wam_min.robodojo_io import JOINT_ONLY_ACTIVE_SLOTS, VIEW_SLOT_IDS, state80_from_obs  # noqa: E402
from sana_wam_min.robot80 import ROBOT80_JOINT_SLICES, denormalize_action, load_normalization, normalize_state  # noqa: E402
from sana_wam_min.session import (  # noqa: E402
    PACKAGED_NORMALIZATION_PATH,
    TRAINING_NORMALIZATION_SHA256,
    PolicyInferenceSession,
    PredictResult,
    find_train_config,
    load_checked_normalization,
    resolve_normalization_path,
    resolve_branch_cfg_scales,
    resolve_sampling_knobs,
)
from sana_wam_min.text import render_token_group_rows  # noqa: E402
from sana_wam_min.vae import VaeBundle, encode_video  # noqa: E402

# The resolved training yaml of the RoboDojo 320px joint-only line, as the trainer dumps it next to a
# checkpoint; the header of tests/fixtures/config.yaml states its provenance.
SNAPSHOT_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "config.yaml")
# Optional: a holdout validation manifest (cluster artifact, not in the repo). Override with
# SANA_WAM_TEST_MANIFEST_GLOB; when nothing matches the test renders the prompt rows itself.
MANIFEST_GLOB = os.environ.get(
    "SANA_WAM_TEST_MANIFEST_GLOB",
    "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/sana_wam_runs/output/"
    "VAL_SFT_RoboDojo_ArxX5_320px_unified_joint_only_holdout_s35000/log_vis/rwm_validation/"
    "policy/joint_only/step_35000/sample_000_RoboDojo-ARX-X5/manifest.json",
)
LATENT_C, LATENT_H, LATENT_W = 4, 2, 3
CAP_C, CAP_L = 16, 12
STEPS = 3


# -- stubs ------------------------------------------------------------------------


class StubEncoder:
    """Deterministic pixel -> [1, 4, 1, 2, 3] latent stand-in for the causal VAE encoder."""

    device = torch.device("cpu")
    dtype = torch.float32

    def encode(self, video: torch.Tensor):
        pooled = torch.nn.functional.adaptive_avg_pool3d(video, (1, LATENT_H, LATENT_W))
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
    """Byte-based tokenizer stand-in; records every call for batch-size assertions."""

    def __init__(self):
        self.calls: list[int] = []
        self.prompts: list[list[str]] = []

    def __call__(self, prompts, max_length, padding, truncation, return_tensors):
        self.calls.append(len(prompts))
        self.prompts.append([str(prompt) for prompt in prompts])
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
    """Records every data_info it sees; constant velocities keep the Euler loop deterministic."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, x, timestep, y, mask=None, data_info=None):
        self.calls.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in data_info.items()})
        assert y.shape[-1] == CAP_C and mask.shape[-1] == CAP_L
        assert timestep.shape == (1, 1, x.shape[2])
        return {"x": torch.zeros_like(x), "action_pred": torch.full_like(data_info["action80"], 0.5)}


def make_session(cfg_scale: float = 1.0, model: StubModel | None = None, **knobs) -> PolicyInferenceSession:
    train_cfg = load_train_config(SNAPSHOT_CONFIG)
    train_cfg["text_encoder"]["model_max_length"] = CAP_L
    return PolicyInferenceSession(
        model=model or StubModel(),
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


# -- path / knob resolution ----------------------------------------------------------


def test_import_through_both_package_roots():
    import XPolicyLab.policy.SANA_WAM.sana_wam_min.session as ns_session

    assert ns_session.PredictResult.__doc__ == PredictResult.__doc__
    with open(session_module.__file__) as f:
        for line in f.read().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert not any(tok in stripped for tok in ("dev.", "diffusion.", "sana.", "tqdm")), stripped


def test_find_train_config_searches_dir_parent_grandparent(tmp_path):
    shutil.copy(SNAPSHOT_CONFIG, tmp_path / "config.yaml")
    ckpt = tmp_path / "checkpoints" / "epoch_6_step_35000"
    (ckpt / "model").mkdir(parents=True)
    assert find_train_config(ckpt) == tmp_path / "config.yaml"
    assert find_train_config(ckpt / "model" / "pytorch_model_fsdp.bin") == tmp_path / "config.yaml"
    shutil.copy(SNAPSHOT_CONFIG, ckpt / "config.yaml")
    assert find_train_config(ckpt) == ckpt / "config.yaml"
    with pytest.raises(FileNotFoundError):
        find_train_config(tmp_path / "elsewhere" / "a" / "b")


def test_load_checked_normalization_refuses_an_artifact_the_yaml_does_not_pin(tmp_path):
    """A checkpoint whose yaml pins another artifact (e.g. the robot_base_eef line, rot6d normalized) must not fall back
    to the packaged joint-only copy silently; an explicit normalization_sha256 override still admits it."""
    import copy

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    train_cfg = load_train_config(SNAPSHOT_CONFIG)
    other = copy.deepcopy(train_cfg)
    other["data"]["extra"]["robotwin_sft"]["normalization_sha256"] = "817630bbd5d644ba7c663703a0dce7bfefb836011aba9abe7aeeb5e3f906aab6"
    with pytest.raises(ValueError, match="not the one the training yaml pins"):
        load_checked_normalization(ckpt, other)
    # an explicit normalization_path is the operator's choice and is not held against the yaml pin
    assert load_checked_normalization(ckpt, other, normalization_path=PACKAGED_NORMALIZATION_PATH).sha256 == TRAINING_NORMALIZATION_SHA256
    assert load_checked_normalization(ckpt, other, None, TRAINING_NORMALIZATION_SHA256).sha256 == TRAINING_NORMALIZATION_SHA256
    assert load_checked_normalization(ckpt, train_cfg).sha256 == TRAINING_NORMALIZATION_SHA256      # pin == packaged copy
    mapped = copy.deepcopy(train_cfg)
    mapped["data"]["extra"]["robotwin_sft"]["normalization_sha256"] = {PACKAGED_NORMALIZATION_PATH.name: TRAINING_NORMALIZATION_SHA256}
    assert load_checked_normalization(ckpt, mapped).sha256 == TRAINING_NORMALIZATION_SHA256



def test_resolve_normalization_path_precedence(tmp_path):
    ckpt = tmp_path / "ckpt"
    (ckpt / "model").mkdir(parents=True)
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}")
    assert resolve_normalization_path(ckpt, None) == PACKAGED_NORMALIZATION_PATH
    assert resolve_normalization_path(ckpt, explicit) == explicit.resolve()
    (ckpt / "normalization").mkdir()
    # Other json files in normalization/ are never picked up.
    (ckpt / "normalization" / "zzz.json").write_text("{}")
    assert resolve_normalization_path(ckpt, None) == PACKAGED_NORMALIZATION_PATH
    shutil.copy(PACKAGED_NORMALIZATION_PATH, ckpt / "normalization" / PACKAGED_NORMALIZATION_PATH.name)
    canonical = ckpt / "normalization" / PACKAGED_NORMALIZATION_PATH.name
    assert resolve_normalization_path(ckpt, None) == canonical
    # The explicit path wins over the checkpoint copy.
    assert resolve_normalization_path(ckpt, explicit) == explicit.resolve()
    assert resolve_normalization_path(ckpt / "model" / "pytorch_model_fsdp.bin", None) == canonical


def test_load_checked_normalization_pins_and_cross_checks(tmp_path):
    train_cfg = load_train_config(SNAPSHOT_CONFIG)
    ckpt = tmp_path / "ckpt"
    (ckpt / "model").mkdir(parents=True)
    # Packaged fallback: pinned by default.
    norm = load_checked_normalization(ckpt, train_cfg)
    assert norm.sha256 == TRAINING_NORMALIZATION_SHA256
    # Checkpoint copy: unpinned unless a pin is given; a given pin is enforced.
    (ckpt / "normalization").mkdir()
    canonical = ckpt / "normalization" / PACKAGED_NORMALIZATION_PATH.name
    shutil.copy(PACKAGED_NORMALIZATION_PATH, canonical)
    assert load_checked_normalization(ckpt, train_cfg).source_path == str(canonical)
    assert load_checked_normalization(ckpt, train_cfg, expected_sha256=TRAINING_NORMALIZATION_SHA256).sha256 == TRAINING_NORMALIZATION_SHA256
    with pytest.raises(ValueError):
        load_checked_normalization(ckpt, train_cfg, expected_sha256="0" * 64)
    # Explicit artifact whose declared mode disagrees with the yaml fails before any model is built.
    payload = json.loads(PACKAGED_NORMALIZATION_PATH.read_text())
    payload["joint_target_mode"] = "absolute"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="joint_target_mode"):
        load_checked_normalization(ckpt, train_cfg, normalization_path=bad)
    payload["joint_target_mode"] = norm.joint_target_mode
    payload["num_frames"] = 33
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="num_frames"):
        load_checked_normalization(ckpt, train_cfg, normalization_path=bad)


def test_packaged_normalization_matches_training_pin():
    norm = load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256)
    assert norm.sha256 == TRAINING_NORMALIZATION_SHA256
    assert norm.joint_target_mode == "anchor_delta" and norm.num_frames == 25 and norm.model_fps == 25
    with pytest.raises(ValueError):
        load_normalization(PACKAGED_NORMALIZATION_PATH, "sha256:" + "0" * 64)


def test_resolve_sampling_knobs_defaults_and_overrides():
    cfg = load_train_config(SNAPSHOT_CONFIG)
    assert resolve_sampling_knobs(cfg) == (50, 1.0, 3.5)
    assert resolve_sampling_knobs(cfg, steps=8, cfg_scale=2.0, flow_shift=3.0) == (8, 2.0, 3.0)
    with pytest.raises(ValueError):
        resolve_sampling_knobs(cfg, cfg_scale=0.5)
    with pytest.raises(ValueError):
        resolve_sampling_knobs(cfg, steps=0)


def test_resolve_branch_cfg_scales_inherit_and_validate():
    assert resolve_branch_cfg_scales(6.0) == (6.0, 6.0)
    assert resolve_branch_cfg_scales(6.0, action_cfg_scale=1.0) == (6.0, 1.0)
    assert resolve_branch_cfg_scales(1.0, video_cfg_scale=6.0) == (6.0, 1.0)
    assert resolve_branch_cfg_scales(1.0, 2.0, 3.0) == (2.0, 3.0)
    for bad in ({"video_cfg_scale": 0.5}, {"action_cfg_scale": 0.0}, {"action_cfg_scale": float("inf")}):
        with pytest.raises(ValueError):
            resolve_branch_cfg_scales(6.0, **bad)


# -- data_info / predict --------------------------------------------------------------


def test_build_data_info_contract():
    session = make_session()
    state80, mask80 = state80_from_obs(fake_state())
    normalized = normalize_state(state80, mask80, session.normalization)
    action = torch.zeros(1, 24, 80)
    action_mask = torch.as_tensor(mask80).reshape(1, 1, 80).expand(1, 24, 80).contiguous()
    info = session.build_data_info(((LATENT_H, LATENT_W),) * 3, normalized, torch.as_tensor(mask80), action, action_mask)
    assert info["rwm_task"] == "policy" and info["camera_conditioning_enabled"] is False
    assert info["view_count"].tolist() == [3] and info["view_count"].dtype == torch.int64
    assert info["view_slot_ids"].tolist() == list(VIEW_SLOT_IDS)
    assert info["view_latent_shape"].shape == (1, 3, 2) and info["view_latent_shape"][0, 1].tolist() == [LATENT_H, LATENT_W]
    assert info["model_fps"].dtype == torch.float32 and float(info["model_fps"]) == 25.0
    assert info["initial_state80"].shape == (1, 80) and info["initial_state80"].dtype == torch.float32
    assert torch.equal(info["initial_state80"][0], normalized)
    assert info["initial_state_condition_mask80"].dtype == torch.bool
    assert info["initial_state_condition_mask80"][0].nonzero().flatten().tolist() == list(JOINT_ONLY_ACTIVE_SLOTS)
    assert info["action80"].shape == (1, 24, 80) and info["action_mask80"].dtype == torch.bool
    assert info["num_views_per_sample"] == 3 and info["sample_batch_size"] == 1
    with pytest.raises(ValueError):
        session.build_data_info(((LATENT_H, LATENT_W),) * 2, normalized, torch.as_tensor(mask80), action, action_mask)


def test_predict_end_to_end_with_stubs():
    model = StubModel()
    session = make_session(model=model)
    frames = fake_frames()
    state80, mask80 = state80_from_obs(fake_state())
    instruction = "Stack the three bowls together."
    result = session.predict(frames, state80, mask80, instruction, torch.Generator().manual_seed(7))

    assert isinstance(result, PredictResult)
    assert result.action80_raw_absolute.shape == (24, 80) and result.action80_raw_absolute.dtype == torch.float32
    assert result.action80_model.shape == (24, 80) and result.action_mask.shape == (24, 80)
    assert result.action_mask[0].nonzero().flatten().tolist() == list(JOINT_ONLY_ACTIVE_SLOTS)
    assert result.video_latent.shape == (1, LATENT_C, 4, 1, 3 * LATENT_H * LATENT_W)
    assert torch.isfinite(result.action80_raw_absolute).all()
    # Masked-out slots are exactly zero in both domains.
    inactive = ~result.action_mask
    assert torch.all(result.action80_model[inactive] == 0) and torch.all(result.action80_raw_absolute[inactive] == 0)
    # Frame 0 of the strip is the packed per-view observation latent; frames 1.. were sampled.
    windows = []
    for frame in frames:
        pixels = frame_to_model_tensor(frame, (256, 320))
        windows.append(encode_video(session.vae, frames_to_vae_input(pixels.unsqueeze(0))))
    strip, _ = pack_multiview_latents(windows)
    assert torch.equal(result.video_latent[:, :, :1].float(), strip.float())
    # Model saw one call per step with the deploy data_info.
    assert len(model.calls) == STEPS
    info = model.calls[0]
    assert info["rwm_task"] == "policy" and info["view_slot_ids"].tolist() == [0, 2, 3]
    assert info["view_latent_shape"].tolist() == [[[LATENT_H, LATENT_W]] * 3]
    assert torch.equal(info["initial_state80"][0], normalize_state(state80, mask80, session.normalization))
    assert info["action_timestep"].shape == (1, 24) and float(info["action_timestep"][0, 0]) == 1000.0
    assert info["action80"].shape == (1, 24, 80)
    # Post-processing: raw absolute joints = anchor + denormalized delta on the two joint slices.
    denorm = denormalize_action(result.action80_model, result.action_mask, session.normalization)
    for joint_slice in ROBOT80_JOINT_SLICES:
        active = result.action_mask[:, joint_slice]
        delta = (result.action80_raw_absolute[:, joint_slice] - denorm[:, joint_slice])[active]
        anchor = torch.as_tensor(state80)[joint_slice][None, :].expand(24, -1)[active]
        assert torch.allclose(delta, anchor, atol=1e-6)
    grippers = result.action80_raw_absolute[:, [16, 45]]
    assert torch.all(grippers >= 0) and torch.all(grippers <= 1)
    assert result.receipt["steps"] == STEPS and result.receipt["cfg_scale"] == 1.0 and result.receipt["seed"] == 7
    assert result.receipt["normalization_sha256"] == TRAINING_NORMALIZATION_SHA256
    assert result.receipt["latency_ms"] >= 0 and result.receipt["k_actions"] == 24


def test_predict_is_seed_deterministic_and_prompt_cached():
    session = make_session()
    frames = fake_frames()
    state80, mask80 = state80_from_obs(fake_state())
    a = session.predict(frames, state80, mask80, "Pick up the cup.", torch.Generator().manual_seed(3))
    b = session.predict(frames, state80, mask80, "Pick up the cup.", torch.Generator().manual_seed(3))
    c = session.predict(frames, state80, mask80, "Pick up the cup.", torch.Generator().manual_seed(4))
    assert torch.equal(a.action80_model, b.action80_model) and torch.equal(a.video_latent, b.video_latent)
    assert not torch.equal(a.action80_model, c.action80_model)
    assert session.tokenizer.calls == [4]


def test_predict_from_latent_generator_matches_explicit_noise_in_video_then_action_order():
    session = make_session()
    strip = torch.randn(1, LATENT_C, 4, 1, 3 * LATENT_H * LATENT_W)
    state80, mask80 = state80_from_obs(fake_state())
    normalized = normalize_state(state80, mask80, session.normalization)[None]
    action_mask = torch.as_tensor(mask80).reshape(1, 1, 80).expand(1, 24, 80).contiguous()
    clean_action = torch.randn(1, 24, 80).masked_fill(~action_mask, 0)
    manifests = glob.glob(MANIFEST_GLOB)
    if manifests:
        with open(manifests[0]) as f:
            prompt = json.load(f)["prompt"]
        rows_cond, rows_uncond = tuple(prompt["conditional"]), tuple(prompt["unconditional"])
    else:
        rows_cond = render_token_group_rows("Make toast.")
        rows_uncond = render_token_group_rows("Make toast.", include_instruction=False)
    shapes = ((LATENT_H, LATENT_W),) * 3

    action_g, video_g = session.predict_from_latent(
        strip, shapes, rows_cond, rows_uncond, normalized, torch.as_tensor(mask80)[None], clean_action, action_mask,
        generator=torch.Generator().manual_seed(11),
    )
    g = torch.Generator().manual_seed(11)
    video_noise = torch.randn(strip.shape, dtype=strip.dtype, generator=g)
    action_noise = torch.randn(clean_action.shape, dtype=torch.float32, generator=g)
    action_n, video_n = session.predict_from_latent(
        strip, shapes, rows_cond, rows_uncond, normalized, torch.as_tensor(mask80)[None], clean_action, action_mask,
        video_noise=video_noise, action_noise=action_noise,
    )
    assert action_g.shape == (1, 24, 80) and video_g.shape == strip.shape
    assert torch.equal(action_g, action_n) and torch.equal(video_g, video_n)
    assert torch.equal(video_g[:, :, :1], strip[:, :, :1])
    assert torch.all(action_g.masked_select(~action_mask) == 0)


def test_cfg_scale_above_one_encodes_both_rows_in_one_batch_and_doubles_forwards():
    model = StubModel()
    session = make_session(cfg_scale=2.0, model=model)
    state80, mask80 = state80_from_obs(fake_state())
    session.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))
    assert session.tokenizer.calls == [8]
    assert len(model.calls) == 2 * STEPS


def test_video_only_cfg_still_encodes_and_forwards_the_unconditional_rows_and_is_receipted():
    model = StubModel()
    session = make_session(cfg_scale=6.0, model=model, action_cfg_scale=1.0)
    assert (session.cfg_scale, session.video_cfg_scale, session.action_cfg_scale) == (6.0, 6.0, 1.0)
    assert session.uses_cfg
    state80, mask80 = state80_from_obs(fake_state())
    result = session.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))
    assert session.tokenizer.calls == [8]          # conditional + unconditional rows in one batch
    assert len(model.calls) == 2 * STEPS           # the shared transformer still runs the unconditional pass
    assert result.receipt["cfg_scale"] == 6.0
    assert result.receipt["video_cfg_scale"] == 6.0 and result.receipt["action_cfg_scale"] == 1.0
    # Guiding only the action stream through its own knob also switches the unconditional rows on.
    model = StubModel()
    session = make_session(cfg_scale=1.0, model=model, action_cfg_scale=4.0)
    assert (session.video_cfg_scale, session.action_cfg_scale) == (1.0, 4.0) and session.uses_cfg
    session.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))
    assert session.tokenizer.calls == [8] and len(model.calls) == 2 * STEPS
    # Neither guided -> single row, single forward (unchanged historical path).
    model = StubModel()
    session = make_session(cfg_scale=1.0, model=model)
    assert not session.uses_cfg
    session.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))
    assert session.tokenizer.calls == [4] and len(model.calls) == STEPS


def test_video_only_cfg_action_matches_the_unguided_action_bitwise():
    """With the stub's constant velocities the action stream must be identical whether or not the video is guided."""
    state80, mask80 = state80_from_obs(fake_state())
    plain = make_session(cfg_scale=1.0).predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(3))
    video_only = make_session(cfg_scale=6.0, action_cfg_scale=1.0).predict(
        fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(3)
    )
    assert torch.equal(video_only.action80_model, plain.action80_model)
    assert torch.equal(video_only.action80_raw_absolute, plain.action80_raw_absolute)


# -- OpenWAM canvas line -------------------------------------------------------------------------------------------

import dataclasses  # noqa: E402

from sana_wam_min.openwam_canvas import OPENWAM_VIEW_SLOT_IDS, assemble_openwam_canvas, canvas_to_model_tensor  # noqa: E402
from sana_wam_min.text import ACTION_MODE_TEXT  # noqa: E402

CANVAS_CONFIG = os.path.join(os.path.dirname(__file__), "fixtures", "config_openwam_canvas.yaml")
CANVAS_LATENT_H, CANVAS_LATENT_W = 12, 10
ABSOLUTE_JOINT_ONLY = "joint_only; joint and gripper targets are absolute future targets and no end-effector action is supervised"


class CanvasStubEncoder(StubEncoder):
    """Pools the 384x320 canvas to the trained 12x10 latent grid and records every input shape."""

    def __init__(self):
        self.inputs: list[tuple] = []

    def encode(self, video: torch.Tensor):
        self.inputs.append(tuple(video.shape))
        pooled = torch.nn.functional.adaptive_avg_pool3d(video, (1, CANVAS_LATENT_H, CANVAS_LATENT_W))
        z = torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))


def make_canvas_session(cfg_scale: float = 1.0, model=None, encoder=None, **knobs) -> PolicyInferenceSession:
    train_cfg = load_train_config(CANVAS_CONFIG)
    train_cfg["text_encoder"]["model_max_length"] = CAP_L
    vae = stub_vae()
    vae.causal_encoder = encoder or CanvasStubEncoder()
    # the packaged artifact is the anchor_delta one; the line's own (802f8fe9...) declares absolute, like its yaml
    norm = dataclasses.replace(load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256), joint_target_mode="absolute")
    return PolicyInferenceSession(
        model=model or StubModel(), vae=vae, tokenizer=StubTokenizer(), text_encoder=StubTextEncoder(), normalization=norm,
        train_config=train_cfg, device="cpu", steps=STEPS, cfg_scale=cfg_scale, flow_shift=3.5, checkpoint_path="stub", **knobs,
    )


def test_canvas_session_resolves_the_layout_and_the_absolute_prompt_sentence():
    session = make_canvas_session()
    assert session.visual_layout == "openwam_canvas" and session.canvas
    assert session.view_keys == ("openwam_canvas",) and session.view_slot_ids == OPENWAM_VIEW_SLOT_IDS == (0,)
    assert session.view_order == ("cam_head", "cam_left_wrist", "cam_right_wrist")
    assert (session.action_mode, session.joint_target_mode, session.eef_target_mode) == ("joint_only", "absolute", "anchor_delta")
    assert session.action_mode_text == ABSOLUTE_JOINT_ONLY
    three = make_session()
    assert three.visual_layout == "three_view_strip" and not three.canvas
    assert three.view_slot_ids == VIEW_SLOT_IDS and three.view_keys == three.view_order
    assert three.action_mode_text == ACTION_MODE_TEXT


def test_canvas_predict_composites_once_and_shares_one_prompt():
    model, encoder = StubModel(), CanvasStubEncoder()
    session = make_canvas_session(model=model, encoder=encoder)
    frames = fake_frames()
    state80, mask80 = state80_from_obs(fake_state())
    result = session.predict(frames, state80, mask80, "Stack the three bowls together.", torch.Generator().manual_seed(7))
    assert encoder.inputs == [(1, 3, 1, 384, 320)]  # ONE encode of the composited canvas, not three views
    assert result.video_latent.shape == (1, LATENT_C, 4, CANVAS_LATENT_H, CANVAS_LATENT_W)
    assert result.receipt["visual_layout"] == "openwam_canvas" and result.receipt["view_latent_shapes"] == ((12, 10),)
    canvas = assemble_openwam_canvas(dict(zip(session.view_order, frames, strict=True)))
    expected = encode_video(session.vae, frames_to_vae_input(canvas_to_model_tensor(canvas).unsqueeze(0)))
    assert torch.equal(result.video_latent[:, :, :1].float(), expected.float())
    assert len(model.calls) == STEPS
    info = model.calls[0]
    assert info["view_count"].tolist() == [1] and info["view_slot_ids"].tolist() == [0]
    assert info["view_latent_shape"].tolist() == [[[12, 10]]] and info["num_views_per_sample"] == 1
    assert torch.equal(info["initial_state80"][0], normalize_state(state80, mask80, session.normalization))
    assert session.tokenizer.calls == [1]  # one shared row, no CFG twin
    (row,) = session.tokenizer.prompts[0]
    assert row.split("\n") == [
        "Embodiment Type: dual-arm RoboDojo ARX-X5 robot with parallel grippers.",
        f"Action Mode: {ABSOLUTE_JOINT_ONLY}.",
        "Observation View: a composite view combining the head camera above the left and right wrist cameras.",
        "Instruction: Stack the three bowls together.",
    ]
    # absolute joint targets: the raw chunk is the denormalized prediction, no anchor added
    denorm = denormalize_action(result.action80_model, result.action_mask, session.normalization)
    for joint_slice in ROBOT80_JOINT_SLICES:
        assert torch.equal(result.action80_raw_absolute[:, joint_slice], denorm[:, joint_slice])
    assert result.action_mask[0].nonzero().flatten().tolist() == list(JOINT_ONLY_ACTIVE_SLOTS)


def test_canvas_cfg_encodes_the_shared_row_and_its_unconditional_twin_in_one_batch():
    model = StubModel()
    session = make_canvas_session(cfg_scale=6.0, model=model, action_cfg_scale=1.0)
    state80, mask80 = state80_from_obs(fake_state())
    session.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))
    assert session.tokenizer.calls == [2] and len(model.calls) == 2 * STEPS
    conditional, unconditional = session.tokenizer.prompts[0]
    assert unconditional == "\n".join(conditional.split("\n")[:-1]) and conditional.endswith("Instruction: Open the drawer.")


def test_canvas_encode_refuses_a_wrong_latent_grid_and_needs_three_cameras():
    session = make_canvas_session(encoder=StubEncoder())  # pools to 2x3, not the trained 12x10
    state80, mask80 = state80_from_obs(fake_state())
    with pytest.raises(ValueError, match="canvas latent grid"):
        session.predict(fake_frames(), state80, mask80, "x", torch.Generator().manual_seed(1))
    with pytest.raises(ValueError, match="3 camera frames"):
        make_canvas_session().encode_observation(fake_frames()[:2], 4)


def test_three_view_prompt_sentence_follows_the_training_yaml():
    """A strip-line checkpoint with absolute joint targets or robot_base_eef state renders ITS Action Mode sentence."""

    def _session(joint_target_mode: str, ratio: list[float]) -> PolicyInferenceSession:
        train_cfg = load_train_config(SNAPSHOT_CONFIG)
        train_cfg["text_encoder"]["model_max_length"] = CAP_L
        train_cfg["data"]["extra"]["joint_target_mode"] = joint_target_mode
        train_cfg["data"]["extra"]["action_mode_sample_ratio"] = ratio
        norm = dataclasses.replace(load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256), joint_target_mode=joint_target_mode)
        return PolicyInferenceSession(
            model=StubModel(), vae=stub_vae(), tokenizer=StubTokenizer(), text_encoder=StubTextEncoder(), normalization=norm,
            train_config=train_cfg, device="cpu", steps=STEPS, cfg_scale=1.0, flow_shift=3.5, checkpoint_path="stub",
        )

    state80, mask80 = state80_from_obs(fake_state())
    absolute = _session("absolute", [0.0, 0.0, 1.0])
    assert absolute.action_mode_text == ABSOLUTE_JOINT_ONLY
    absolute.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))
    rows = absolute.tokenizer.prompts[0]
    assert len(rows) == 4 and all(row.split("\n")[1] == f"Action Mode: {ABSOLUTE_JOINT_ONLY}." for row in rows)
    eef = _session("anchor_delta", [0.0, 1.0, 0.0])
    assert eef.action_mode == "robot_base_eef" and eef.action_mode_text.startswith(
        "robot_base_eef; end-effector motion is relative to the first state and expressed in the robot base frame"
    )
    assert make_session().action_mode_text == ACTION_MODE_TEXT  # the historical joint_only / anchor_delta line is unchanged
