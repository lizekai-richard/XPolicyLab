"""Strided video (rwm/strided_video, ``data.extra.robotwin_sft.video_fps``): the stride derivation, the session's
shorter latent window + ``video_frame_stride`` key, the policy port's action-row check and robot-tail RoPE at a stride,
and bitwise parity with the live three-view policy of a rwm/strided_video checkout (``SANA_STRIDED_REPO``, default
~/zekail/Sana_strided). The live tests skip when that checkout is not importable or when another Sana checkout already
owns the ``dev`` package in this interpreter (the other parity files import ~/zekail/Sana / ~/zekail/Sana_openwam), so run
this file on its own for the live parity: ``pytest tests/test_video_frame_stride.py``."""

from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from _tiny_policy import (  # noqa: E402  (puts policy/SANA_WAM and the XPolicyLab parent on sys.path)
    CAP_CH,
    DEPTH,
    FRAMES,
    HEADS,
    HIDDEN,
    IN_CH,
    LHD,
    MML,
    NOISY_T,
    ROBOT_DIM,
    SHD,
    TILE,
    VIEW_H,
    VIEW_W,
    tiny_policy_config,
)
from test_session_cpu import (  # noqa: E402
    CAP_L,
    SNAPSHOT_CONFIG,
    STEPS,
    StubModel,
    StubTextEncoder,
    StubTokenizer,
    fake_frames,
    fake_state,
    stub_vae,
)

from sana_wam_min import session as session_module  # noqa: E402
from sana_wam_min.config import (  # noqa: E402
    load_train_config,
    sft_options_from_train_config,
    video_fps_from_train_config,
    video_frame_stride_from_train_config,
)
from sana_wam_min.frame_stride import (  # noqa: E402
    parse_video_fps,
    strided_video_frames,
    video_frame_stride_from_video_fps,
)
from sana_wam_min.policy_model.checkpoint import load_policy_state_dict  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402
from sana_wam_min.policy_model.model import PolicyModel, _independent_action_rope  # noqa: E402
from sana_wam_min.policy_model.rope import PhysicalTimeWanRotaryPosEmbed  # noqa: E402
from sana_wam_min.robodojo_io import state80_from_obs  # noqa: E402
from sana_wam_min.robot80 import load_normalization  # noqa: E402
from sana_wam_min.session import (  # noqa: E402
    PACKAGED_NORMALIZATION_PATH,
    TRAINING_NORMALIZATION_SHA256,
    PolicyInferenceSession,
    training_normalization_pin,
)

SANA_STRIDED_REPO = os.environ.get("SANA_STRIDED_REPO", os.path.expanduser("~/zekail/Sana_strided"))
STEPS_PER_FRAME = 8   # the causal VAE's temporal stride = action rows per latent frame at stride 1


# -- the stride arithmetic (dev/rwm/tests/test_video_frame_stride.py) ---------------------------------------------


@pytest.mark.parametrize(
    "rows, stride, frames",
    [(33, 1, 33), (33, 2, 17), (33, 4, 9), (25, 3, 9), (49, 2, 25), (97, 4, 25), (17, 2, 9), (25, 1, 25)],
)
def test_strided_windows_keep_the_vae_clock(rows, stride, frames):
    assert strided_video_frames(rows, stride, STEPS_PER_FRAME) == frames


@pytest.mark.parametrize("rows, stride", [(25, 2), (33, 3), (25, 12), (9, 8), (25, 24), (1, 1)])
def test_windows_that_miss_the_clock_are_refused(rows, stride):
    with pytest.raises(ValueError):
        strided_video_frames(rows, stride, STEPS_PER_FRAME)


@pytest.mark.parametrize(
    "rows, video_fps, stride, frames",
    [(25, 8, 3, 9), (25, 24, 1, 25), (33, 16, 2, 17), (33, 8, 4, 9), (33, 32, 1, 33), (49, 16, 3, 17), (97, 24, 4, 25)],
)
def test_the_frame_stride_leaves_video_fps_frames_after_the_observation(rows, video_fps, stride, frames):
    assert video_frame_stride_from_video_fps(rows, video_fps) == stride
    assert strided_video_frames(rows, stride, STEPS_PER_FRAME) == frames == video_fps + 1


def test_video_fps_parsing_and_refusals():
    assert parse_video_fps(None) is None
    assert parse_video_fps(8) == 8
    for bad in (0, -1, True, 2.5, "8"):
        with pytest.raises(ValueError, match="video_fps"):
            parse_video_fps(bad)
    with pytest.raises(ValueError, match="multiple"):
        video_frame_stride_from_video_fps(25, 7)
    # video_fps 12 on 25 rows = stride 2 = 13 video frames: not 1 + 8k
    with pytest.raises(ValueError, match="causal VAE"):
        strided_video_frames(25, video_frame_stride_from_video_fps(25, 12), STEPS_PER_FRAME)


# -- the training yaml -----------------------------------------------------------------------------------------------


def _cfg(video_fps=None, rows=None) -> dict:
    cfg = load_train_config(SNAPSHOT_CONFIG)          # the 25-row RoboDojo window
    options = cfg["data"]["extra"]["robotwin_sft"]
    if video_fps is not None:
        options["video_fps"] = video_fps
    if rows is not None:
        options["tier_num_frames"] = rows
        cfg["data"]["num_frames"] = rows
        cfg["data"]["multi_fps"] = {"25": [rows]}
    return cfg


def test_the_yaml_declares_video_fps_and_the_stride_follows_the_window_rows():
    assert video_fps_from_train_config(_cfg()) is None and video_frame_stride_from_train_config(_cfg()) == 1
    assert video_frame_stride_from_train_config(_cfg(8)) == 3
    assert video_frame_stride_from_train_config(_cfg(24)) == 1          # every row after the observation: dense
    assert video_frame_stride_from_train_config(_cfg(16, rows=33)) == 2
    assert video_frame_stride_from_train_config(_cfg(8, rows=33)) == 4
    with pytest.raises(ValueError, match="multiple"):
        video_frame_stride_from_train_config(_cfg(7))
    cfg = _cfg(8)
    del cfg["data"]["extra"]["robotwin_sft"]["tier_num_frames"]
    with pytest.raises(ValueError, match="tier_num_frames"):
        video_frame_stride_from_train_config(cfg)
    cfg = _cfg(8)
    cfg["data"]["num_frames"] = 33
    with pytest.raises(ValueError, match="num_frames"):
        video_frame_stride_from_train_config(cfg)


def _renamed(cfg: dict) -> dict:
    """The same yaml with the SFT block under its 2026-09-24 name (Sana 875b3f659: robotwin_sft -> robot_sft)."""

    extra = cfg["data"]["extra"]
    extra["robot_sft"] = extra.pop("robotwin_sft")
    return cfg


def test_the_sft_block_is_read_under_its_renamed_key():
    """Configs frozen after 875b3f659 (e.g. sft_robodojo_sanavideo_eefabs_f33fps8_sana_pixel_aligned) name the block
    robot_sft; reading only the retired name would serve a strided checkpoint as dense video and drop the norm pin."""

    old, new = _cfg(8, rows=33), _renamed(_cfg(8, rows=33))
    assert "robotwin_sft" not in new["data"]["extra"] and sft_options_from_train_config(new)["video_fps"] == 8
    assert (video_fps_from_train_config(new), video_frame_stride_from_train_config(new)) == (8, 4)
    assert training_normalization_pin(new) == training_normalization_pin(old) == TRAINING_NORMALIZATION_SHA256
    both = _cfg(8, rows=33)
    both["data"]["extra"]["robot_sft"] = dict(both["data"]["extra"]["robotwin_sft"])
    with pytest.raises(ValueError, match="both robot_sft and the retired robotwin_sft"):
        video_frame_stride_from_train_config(both)
    neither = _cfg()
    del neither["data"]["extra"]["robotwin_sft"]
    assert sft_options_from_train_config(neither) == {} and training_normalization_pin(neither) is None
    assert (video_fps_from_train_config(neither), video_frame_stride_from_train_config(neither)) == (None, 1)


# -- the session: shorter latent window, dense action rows, the data_info key ------------------------------------------


def _session(cfg: dict, model=None) -> PolicyInferenceSession:
    cfg["text_encoder"]["model_max_length"] = CAP_L
    return PolicyInferenceSession(
        model=model or StubModel(),
        vae=stub_vae(),
        tokenizer=StubTokenizer(),
        text_encoder=StubTextEncoder(),
        normalization=load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256),
        train_config=cfg,
        device="cpu",
        steps=STEPS,
        cfg_scale=1.0,
        flow_shift=3.5,
        checkpoint_path="stub",
    )


def _predict(session: PolicyInferenceSession):
    state80, mask80 = state80_from_obs(fake_state())
    return session.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))


def test_a_strided_session_fills_a_shorter_window_and_tags_the_batch():
    dense, strided = _session(_cfg()), _session(_cfg(8))
    assert (dense.video_fps, dense.video_frame_stride) == (None, 1)
    assert (strided.video_fps, strided.video_frame_stride) == (8, 3)
    r_dense, r_strided = _predict(dense), _predict(strided)
    seen_dense, seen = dense.model.calls[0], strided.model.calls[0]
    # a dense checkpoint's batch carries no stride key at all (the historical batch, byte for byte)
    assert "video_frame_stride" not in seen_dense
    assert seen["video_frame_stride"].tolist() == [3] and seen["video_frame_stride"].dtype == torch.int64
    # 25 rows -> 9 video frames (rows 0, 3, .., 24) -> 2 latent frames; dense: 4. The 24 action rows stay.
    assert r_dense.video_latent.shape[2] == 4 and r_strided.video_latent.shape[2] == 2
    assert seen["action80"].shape == seen_dense["action80"].shape == (1, 24, ROBOT_DIM)
    assert (r_dense.receipt["frames"], r_dense.receipt["video_frames"], r_dense.receipt["video_frame_stride"]) == (25, 25, 1)
    assert (r_strided.receipt["frames"], r_strided.receipt["video_frames"], r_strided.receipt["video_frame_stride"]) == (25, 9, 3)
    assert r_strided.receipt["k_actions"] == r_dense.receipt["k_actions"] == 24
    # frame 0 of both windows is the same observation encode (the stub encoder is deterministic)
    torch.testing.assert_close(r_strided.video_latent[:, :, :1], r_dense.video_latent[:, :, :1], rtol=0, atol=0)
    assert r_strided.action80_raw_absolute.shape == (24, ROBOT_DIM)


def test_a_33_row_strided_session_serves_32_targets_from_3_latent_frames():
    session = _session(_cfg(16, rows=33))
    assert session.video_frame_stride == 2
    result = _predict(session)
    seen = session.model.calls[0]
    assert result.video_latent.shape[2] == 3 and seen["video_frame_stride"].tolist() == [2]
    assert seen["action80"].shape == (1, 32, ROBOT_DIM) and result.receipt["k_actions"] == 32
    assert (result.receipt["frames"], result.receipt["video_frames"]) == (33, 17)
    assert result.action80_raw_absolute.shape == (32, ROBOT_DIM)


def test_a_renamed_sft_block_serves_the_same_strided_window():
    old, new = _session(_cfg(8, rows=33)), _session(_renamed(_cfg(8, rows=33)))
    assert (new.video_fps, new.video_frame_stride) == (old.video_fps, old.video_frame_stride) == (8, 4)
    r_old, r_new = _predict(old), _predict(new)
    assert new.model.calls[0]["video_frame_stride"].tolist() == [4] and r_new.video_latent.shape[2] == 2
    assert (r_new.receipt["frames"], r_new.receipt["video_frames"]) == (r_old.receipt["frames"], r_old.receipt["video_frames"]) == (33, 9)
    torch.testing.assert_close(r_new.video_latent, r_old.video_latent, rtol=0, atol=0)


def test_the_canvas_policy_class_refuses_a_strided_yaml(monkeypatch):
    """The rwm/openwam canvas class predates frame striding; the canvas MODES of the one policy class take it (below)."""

    monkeypatch.setattr(session_module, "resolve_visual_layout", lambda cfg: session_module.VISUAL_LAYOUT_OPENWAM_CANVAS)
    monkeypatch.setattr(session_module, "is_openwam_canvas_policy_class", lambda cfg: True)
    with pytest.raises(NotImplementedError, match="predates frame striding"):
        _session(_cfg(8))


# -- the policy port --------------------------------------------------------------------------------------------------


def _inputs(stride: int = 1, batch: int = 2, views: int = 3, fps: float = 25.0, seed: int = 7) -> dict:
    """Three-view strip inputs of the tiny policy; ``stride`` scales the action rows, not the latent frames
    (dev/rwm/tests/test_video_frame_stride.py _policy_inputs)."""

    torch.manual_seed(seed)
    steps = (FRAMES - 1) * STEPS_PER_FRAME * stride
    x = torch.randn(batch, IN_CH, FRAMES, 1, views * VIEW_H * VIEW_W)
    timestep = torch.zeros(batch, 1, FRAMES)
    timestep[:, :, 1:] = NOISY_T
    y = torch.randn(batch, views + 1, 1, MML, CAP_CH)
    mask = torch.ones(batch, views + 1, MML, dtype=torch.int16)
    mask[..., 5:] = 0
    action_mask = torch.ones(batch, steps, ROBOT_DIM, dtype=torch.bool)
    action_mask[:, :, 66:] = False
    state_mask = torch.ones(batch, ROBOT_DIM, dtype=torch.bool)
    state_mask[:, 58:66] = False
    data_info = {
        "rwm_task": "policy",
        "view_count": torch.full((batch,), views, dtype=torch.long),
        "view_latent_shape": torch.tensor([[[VIEW_H, VIEW_W]] * views] * batch),
        "view_slot_ids": torch.tensor((0, 2, 3)[:views]),
        "model_fps": torch.full((batch,), fps),
        "initial_state80": torch.randn(batch, ROBOT_DIM),
        "initial_state_condition_mask80": state_mask,
        "action80": torch.randn(batch, steps, ROBOT_DIM),
        "action_mask80": action_mask,
        "action_timestep": torch.full((batch, steps), NOISY_T),
        "camera_conditioning_enabled": False,
    }
    if stride != 1:
        data_info["video_frame_stride"] = torch.full((batch,), stride, dtype=torch.long)
    return {"x": x, "timestep": timestep, "y": y, "mask": mask, "data_info": data_info}


def _run(model, inputs: dict) -> dict:
    with torch.no_grad():
        return model(
            inputs["x"].clone(), inputs["timestep"].clone(), inputs["y"].clone(), mask=inputs["mask"].clone(),
            data_info=copy.deepcopy(inputs["data_info"]),
        )


@pytest.fixture(scope="module")
def mirror() -> PolicyModel:
    torch.manual_seed(20260919)
    model = PolicyModel(tiny_policy_config())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    return model.eval()


def _reference_physical_robot_rope(rope, fps, batch, steps, device):
    """The dense robot tail: state at time 0, action row i at the time of source frame i (base-fps units)."""

    action_ids = torch.arange(1, steps + 1, device=device, dtype=torch.float64)
    action_time = rope.base_fps * action_ids[None] / (STEPS_PER_FRAME * fps[:, None])
    condition_time = torch.cat((torch.zeros(batch, 1, device=device), action_time), dim=1)
    ids = torch.stack((condition_time, torch.zeros_like(condition_time), torch.zeros_like(condition_time)), dim=-1)
    return rope.from_position_ids(ids)


def _robot_tail(model: PolicyModel, rope, inputs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(video rows, robot rows, fps) of the model's RoPE table for ``inputs``."""

    data_info = inputs["data_info"]
    batch, steps = inputs["x"].shape[0], data_info["action80"].shape[1]
    device = inputs["x"].device
    model.f, model.h, model.w = FRAMES, VIEW_H, 3 * VIEW_W
    shapes = ((VIEW_H, VIEW_W),) * 3
    fps = PhysicalTimeWanRotaryPosEmbed._normalize_fps(data_info["model_fps"], batch, device)
    with rope.use_model_fps(fps, batch_size=batch, device=device):
        table = model._robot_rope(rope, data_info, fps, shapes, steps, device)
    video_tokens = FRAMES * 3 * VIEW_H * VIEW_W
    assert table.shape[2] == video_tokens + 1 + steps
    return table[:, :, :video_tokens], table[:, :, video_tokens:], fps


def test_stride_one_robot_rope_is_the_physical_time_table(mirror):
    inputs = _inputs()
    for rope in (mirror.rope_linear, mirror.rope_softmax):
        _, tail, fps = _robot_tail(mirror, rope, inputs)
        batch, steps = inputs["x"].shape[0], inputs["data_info"]["action80"].shape[1]
        torch.testing.assert_close(
            tail, _reference_physical_robot_rope(rope, fps, batch, steps, inputs["x"].device), rtol=0, atol=0
        )


def test_strided_robot_rope_is_a_zero_phase_state_plus_independent_1d_actions(mirror):
    inputs = _inputs(stride=2)
    batch, steps = inputs["x"].shape[0], inputs["data_info"]["action80"].shape[1]
    device = inputs["x"].device
    for rope in (mirror.rope_linear, mirror.rope_softmax):
        head_dim = sum(rope.axis_dims)
        _, tail, fps = _robot_tail(mirror, rope, inputs)
        physical = _reference_physical_robot_rope(rope, fps, batch, steps, device)
        torch.testing.assert_close(tail[:, :, 0], torch.ones_like(tail[:, :, 0]), rtol=0, atol=0)
        actions = tail[:, :, 1:]
        torch.testing.assert_close(actions, _independent_action_rope(rope, steps, batch, device), rtol=0, atol=0)
        inverse_frequency = rope.theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        for position in (0, 1, 5, steps - 1):
            expected = torch.polar(torch.ones(head_dim // 2, dtype=torch.float64), position * inverse_frequency)
            torch.testing.assert_close(actions[0, 0, position], expected, rtol=0, atol=0)
        assert not torch.allclose(actions, physical[:, :, 1:])
        assert actions.dtype == physical.dtype


def test_the_video_rows_do_not_move_with_the_stride(mirror):
    dense_video, dense_tail, _ = _robot_tail(mirror, mirror.rope_linear, _inputs())
    strided_video, strided_tail, _ = _robot_tail(mirror, mirror.rope_linear, _inputs(stride=2))
    assert dense_tail.shape[2] == 1 + 16 and strided_tail.shape[2] == 1 + 32
    torch.testing.assert_close(dense_video, strided_video, rtol=0, atol=0)


@pytest.mark.parametrize("stride", [2, 3, 4])
def test_a_strided_batch_carries_stride_times_the_action_rows(mirror, stride):
    inputs = _inputs(stride=stride)
    output = _run(mirror, inputs)
    assert output["action_pred"].shape == (2, 16 * stride, ROBOT_DIM)
    assert output["x"].shape == inputs["x"].shape
    assert torch.isfinite(output["action_pred"]).all() and torch.isfinite(output["x"]).all()
    dense = copy.deepcopy(inputs)
    dense["data_info"].pop("video_frame_stride")
    with pytest.raises(ValueError, match="video frame stride 1"):
        _run(mirror, dense)
    mixed = copy.deepcopy(inputs)
    mixed["data_info"]["video_frame_stride"] = torch.tensor([stride, 1])
    with pytest.raises(ValueError, match="uniform"):
        _run(mirror, mixed)


def test_a_dense_batch_is_unchanged_by_the_new_code_path(mirror):
    inputs = _inputs()
    plain = _run(mirror, inputs)
    tagged_inputs = copy.deepcopy(inputs)
    tagged_inputs["data_info"]["video_frame_stride"] = torch.ones(2, dtype=torch.long)
    tagged = _run(mirror, tagged_inputs)
    torch.testing.assert_close(plain["action_pred"], tagged["action_pred"], rtol=0, atol=0)
    torch.testing.assert_close(plain["x"], tagged["x"], rtol=0, atol=0)


# -- bitwise parity with the live rwm/strided_video three-view policy ---------------------------------------------------


class _Cfg(SimpleNamespace):
    """Unset fields read as None (dev/rwm/tests/test_policy_model_ownership.py)."""

    def __getattr__(self, name):
        return None


def _live_class():
    if not os.path.isdir(SANA_STRIDED_REPO):
        pytest.skip(f"no rwm/strided_video checkout at {SANA_STRIDED_REPO} (set SANA_STRIDED_REPO)")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    if SANA_STRIDED_REPO not in sys.path:
        sys.path.insert(0, SANA_STRIDED_REPO)
    module = pytest.importorskip(
        "dev.rwm.diffusion.model.nets.sana_qwennext_action_policy",
        reason="Sana rwm/strided_video checkout not importable (set SANA_STRIDED_REPO)",
    )
    if not os.path.abspath(module.__file__).startswith(os.path.abspath(SANA_STRIDED_REPO)):
        pytest.skip(
            f"the dev package is already imported from another Sana checkout ({module.__file__}); "
            "run this file on its own for the live parity"
        )
    live = module.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy
    if not hasattr(live, "_video_frame_stride"):
        pytest.skip("this Sana checkout has no strided-video support")
    return live


def _twins():
    Live = _live_class()
    torch.manual_seed(20260919)
    kwargs = dict(
        depth=DEPTH, hidden_size=HIDDEN, patch_size=(1, 1, 1), num_heads=HEADS, in_channels=IN_CH,
        caption_channels=CAP_CH, model_max_length=MML, linear_head_dim=LHD, softmax_head_dim=SHD, softmax_ratio=0.5,
        attn_res_block_size=2, use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True,
    )
    config = _Cfg(
        model=_Cfg(
            extra={"action_dim": ROBOT_DIM, "state_dim": ROBOT_DIM},
            multiview_spatial_rope_layout="semantic_2x2",
            multiview_spatial_rope_tile_shape=TILE,
        ),
        vae=_Cfg(vae_stride=[8, 32, 32]),
    )
    live = Live(config=config, **kwargs)
    if getattr(live, "use_xformers_cross_attention", False):
        pytest.skip("the live model kept xformers cross-attention (imported before DISABLE_XFORMERS=1 took effect)")
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.uniform_(-0.05, 0.05)
    live.eval()
    mirror = PolicyModel(PolicyConfig.from_sana_kwargs(config=config, **kwargs))
    load_policy_state_dict(mirror, live.state_dict())
    return live, mirror.eval()


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_forward_bitwise_against_the_live_strided_policy(stride):
    live, mirror = _twins()
    inputs = _inputs(stride=stride, seed=11 + stride)
    out_live, out_mirror = _run(live, inputs), _run(mirror, inputs)
    assert out_mirror["action_pred"].shape == out_live["action_pred"].shape == (2, 16 * stride, ROBOT_DIM)
    assert out_mirror["x"].shape == out_live["x"].shape == inputs["x"].shape
    torch.testing.assert_close(out_mirror["x"], out_live["x"], rtol=0, atol=0)
    torch.testing.assert_close(out_mirror["action_pred"], out_live["action_pred"], rtol=0, atol=0)
    assert out_live["action_pred"].abs().sum() > 0


def test_the_live_and_the_mirror_robot_tables_agree_at_a_stride():
    live, mirror = _twins()
    for stride in (1, 2):
        inputs = _inputs(stride=stride)
        data_info = inputs["data_info"]
        batch, steps = 2, data_info["action80"].shape[1]
        device = inputs["x"].device
        _, mirror_tail, fps = _robot_tail(mirror, mirror.rope_linear, inputs)
        with live.rope_linear.use_model_fps(data_info["model_fps"], batch_size=batch, device=device):
            live_tail = live._robot_rope(live.rope_linear, data_info, batch, steps, device)
        torch.testing.assert_close(mirror_tail, live_tail, rtol=0, atol=0)


def test_live_and_mirror_refuse_dense_rows_at_a_stride():
    live, mirror = _twins()
    inputs = _inputs(stride=2)
    inputs["data_info"]["action80"] = inputs["data_info"]["action80"][:, :16]
    inputs["data_info"]["action_mask80"] = inputs["data_info"]["action_mask80"][:, :16]
    inputs["data_info"]["action_timestep"] = inputs["data_info"]["action_timestep"][:, :16]
    for model in (live, mirror):
        with pytest.raises(ValueError, match="expected 32 .*video frame stride 2"):
            _run(model, inputs)
