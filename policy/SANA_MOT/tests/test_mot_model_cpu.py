"""CPU tests of the MoT mirror on its own: forward contracts, config resolution, checkpoint adapter, real-scale layout."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _tiny_mot import (  # noqa: E402
    B,
    CONFIG_CANVAS,
    CONFIG_F33FPS8,
    CONFIG_MULTIVIEW,
    CONFIG_NSC_F25,
    CONFIG_NSC_F25_SAC,
    F,
    Hh,
    IN_CH,
    Ww,
    tiny_inputs,
    tiny_mot_config,
)

from sana_mot_min.checkpoint import build_mot_model, load_mot_weights, read_mot_state_dict  # noqa: E402
from sana_mot_min.config import (  # noqa: E402
    load_train_config,
    mot_config_from_train_config,
    require_mot_factory,
    resolve_video_layout,
    strided_video_fps,
)
from sana_mot_min.mot_model import MoTLayer, MoTPolicyModel, detect_context_layout, load_mot_state_dict  # noqa: E402
from sana_mot_min.mot_model.checkpoint import UNMODELED_KEYS, checkpoint_layer_layout, strip_unmodeled_mot_state  # noqa: E402


def _with_buffers(model: MoTPolicyModel) -> dict:
    """A trained checkpoint carries the buffers the mirror strips; add them to the mirror's own state dict."""

    cfg = model.mot_config.video
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["video_dit.pos_embed"] = torch.zeros(1, cfg.input_size * cfg.input_size, cfg.hidden_size)
    for key in UNMODELED_KEYS[model.mot_config.context_layout][1:]:
        state[key] = torch.zeros(cfg.model_max_length, cfg.caption_channels)
    return state


def _randomized(config) -> MoTPolicyModel:
    torch.manual_seed(0)
    model = MoTPolicyModel(config).double()
    with torch.no_grad():
        torch.nn.init.normal_(model.action_dit.action_head.linear.weight, std=0.02)
        torch.nn.init.normal_(model.action_dit.action_embed.proj.weight, std=0.05)
    return model


@pytest.mark.parametrize(
    "layout, views, groups",
    [("multiview", 3, None), ("multiview", 1, None), ("openwam_canvas", 1, 1)],
)
@pytest.mark.parametrize("state_as_context", [False, True])
def test_forward_contract(layout, views, groups, state_as_context):
    model = _randomized(tiny_mot_config(state_as_context, video_layout=layout))
    x, timestep, y, mask, data_info = tiny_inputs(view_count=views, groups=groups)
    out = model(x, timestep, y, mask=mask, data_info=data_info)
    assert out["x"].shape == x.shape and out["action_pred"].shape == (B, (F - 1) * 8, 80)
    assert torch.isfinite(out["x"]).all() and torch.isfinite(out["action_pred"]).all()
    assert not out["x"].requires_grad
    again = model(x, timestep, y, mask=mask, data_info=data_info)
    assert torch.equal(out["x"], again["x"]) and torch.equal(out["action_pred"], again["action_pred"])
    assert torch.all(out["action_pred"][:, :, 60:] == 0)                          # masked action slots stay zero
    assert model.video_dit.blocks is None and model.action_dit.blocks is None
    assert all(isinstance(layer, MoTLayer) for layer in model.blocks)
    assert [layer.is_softmax for layer in model.blocks] == [False, True, False, True]
    assert sum(1 for _ in model.named_parameters(remove_duplicate=False)) == sum(1 for _ in model.named_parameters())
    assert hasattr(model, "context_embedder") and not hasattr(model.action_dit, "context_mlp")
    assert not hasattr(model.video_dit, "y_embedder")


def test_strip_layout_contract():
    model = _randomized(tiny_mot_config(False))
    x, timestep, y, mask, data_info = tiny_inputs(view_count=3)
    assert x.shape == (B, IN_CH, F, 1, 3 * Hh * Ww)
    grid = x.reshape(B, IN_CH, F, 3, Hh * Ww)                                     # a [.., 3, 6] grid is not a strip
    with pytest.raises(ValueError, match="packed latent strip"):
        model(grid, timestep, y, mask=mask, data_info=data_info)
    for stride in (2, 3):
        xs, ts, ys, ms, info = tiny_inputs(view_count=3, stride=stride)
        assert model(xs, ts, ys, mask=ms, data_info=info)["action_pred"].shape[1] == 8 * stride
    with pytest.raises(ValueError, match="uniform"):
        model(xs, ts, ys, mask=ms, data_info=dict(info, video_frame_stride=torch.tensor([2, 3])))


def test_forward_refusals():
    model = MoTPolicyModel(tiny_mot_config(False)).double()
    x, timestep, y, mask, data_info = tiny_inputs(view_count=3)
    with pytest.raises(TypeError):
        model(x, timestep, y, mask=mask, data_info=data_info, action=torch.zeros(1))
    with pytest.raises(ValueError, match="rwm_task"):
        model(x, timestep, y, mask=mask, data_info=dict(data_info, rwm_task="forward"))
    with pytest.raises(NotImplementedError):
        model(x, timestep, y, mask=mask, data_info=dict(data_info, image_embeds=torch.zeros(1)))
    bad_t = timestep.clone()
    bad_t[:, :, 0] = 5.0
    with pytest.raises(ValueError, match="clean first"):
        model(x, bad_t, y, mask=mask, data_info=data_info)
    with pytest.raises(ValueError, match="lockstep"):
        model(x, timestep, y, mask=mask, data_info=dict(data_info, action_timestep=data_info["action_timestep"] + 1))
    model.train()
    with pytest.raises(RuntimeError):
        model(x, timestep, y, mask=mask, data_info=data_info)


def test_config_validation():
    good = tiny_mot_config(False)
    for bad in (
        dict(video_layout="grid"),
        dict(multiview_spatial_rope_layout="semantic_3x3"),
        dict(multiview_spatial_rope_tile_shape=(0, 30)),
        dict(context_layout="text_proj"),
        dict(context_layout="legacy_action_mlp", video_layout="multiview"),
        dict(action_attn_res_block_size=4),
    ):
        fields = {**good.__dict__, **bad}
        with pytest.raises(ValueError):
            type(good)(**fields).validate()


def test_training_yaml_resolution():
    multiview = load_train_config(CONFIG_MULTIVIEW)
    assert require_mot_factory(multiview) == "SanaRWMMoTAttnResPolicy_5B_P1_D36"
    cfg = mot_config_from_train_config(multiview)
    assert (cfg.video_layout, cfg.multiview_spatial_rope_layout, cfg.multiview_spatial_rope_tile_shape) == ("multiview", "semantic_2x2", (15, 30))
    assert cfg.context_layout == "context_embedder" and cfg.action_state_as_context is False
    video = cfg.video
    assert (video.depth, video.hidden_size, video.num_heads, video.in_channels, video.input_size) == (32, 2560, 20, 128, 10)
    assert (video.caption_channels, video.model_max_length, video.linear_head_dim, video.softmax_head_dim) == (2304, 300, 128, 256)
    assert video.softmax_layer_indices == (3, 7, 11, 15, 19, 23, 27, 31) and video.attn_res_block_size == 8
    assert (cfg.action_hidden_size, cfg.action_mlp_ratio, cfg.action_cross_attn_heads) == (1024, 4.0, 8)
    assert strided_video_fps(multiview) is None

    assert mot_config_from_train_config(load_train_config(CONFIG_CANVAS)).video_layout == "openwam_canvas"
    strided = load_train_config(CONFIG_F33FPS8)
    assert strided_video_fps(strided) == 8 and mot_config_from_train_config(strided).video_layout == "multiview"

    mislabeled = {**multiview, "model": {**multiview["model"], "extra": {**multiview["model"]["extra"], "video_layout": "openwam_canvas"}}}
    with pytest.raises(ValueError, match="data.type"):
        resolve_video_layout(mislabeled)
    no_type = {**multiview, "data": {k: v for k, v in multiview["data"].items() if k != "type"}}
    assert resolve_video_layout(no_type) == "multiview"
    assert resolve_video_layout(no_type, "legacy_action_mlp") == "multiview"  # the explicit key wins
    bare = {**no_type, "model": {**no_type["model"], "extra": {k: v for k, v in no_type["model"]["extra"].items() if k != "video_layout"}}}
    assert resolve_video_layout(bare) == "multiview" and resolve_video_layout(bare, "legacy_action_mlp") == "openwam_canvas"
    with pytest.raises(ValueError, match="existed only for the OpenWAM canvas"):
        mot_config_from_train_config(multiview, context_layout="legacy_action_mlp")
    other = {**multiview, "model": {**multiview["model"], "model": "SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy_5B_P1_D36"}}
    with pytest.raises(ValueError, match="SANA_MOT adapter serves"):
        mot_config_from_train_config(other)


def test_pre_switch_canvas_yaml_resolves_to_the_canvas():
    """The NSC f25 canvas yamls predate model.extra.video_layout (canvas was the only MoT video layout before 33b220373).

    The same yaml bytes shipped with the epoch-5 snapshot (legacy_action_mlp) and the step-36,250 upload
    (context_embedder, trained at 295f48f15 in the canvas-only window): both must serve the canvas. The
    state_as_context_true twin resolves the same way with the state projector on.
    """

    nsc = load_train_config(CONFIG_NSC_F25)
    assert "video_layout" not in (nsc["model"].get("extra") or {}) and "Canvas" in nsc["data"]["type"]
    resolved = []
    for layout in ("legacy_action_mlp", "context_embedder"):
        cfg = mot_config_from_train_config(nsc, context_layout=layout)
        assert (cfg.video_layout, cfg.context_layout, cfg.action_state_as_context) == ("openwam_canvas", layout, False)
        resolved.append(cfg)
    sac = mot_config_from_train_config(load_train_config(CONFIG_NSC_F25_SAC))
    assert (sac.video_layout, sac.context_layout, sac.action_state_as_context) == ("openwam_canvas", "context_embedder", True)
    for cfg in resolved + [sac]:
        assert (cfg.video.input_size, cfg.video.softmax_layer_indices) == (10, (3, 7, 11, 15, 19, 23, 27, 31))
        assert (cfg.action_hidden_size, cfg.action_cross_attn_heads, cfg.action_attn_res_block_size) == (1024, 8, 8)
    # an explicit key that contradicts the canvas dataset is still refused
    contradicted = {**nsc, "model": {**nsc["model"], "extra": {**nsc["model"]["extra"], "video_layout": "multiview"}}}
    with pytest.raises(ValueError, match="data.type"):
        mot_config_from_train_config(contradicted)


@pytest.mark.parametrize("state_as_context", [False, True])
def test_checkpoint_round_trip_through_disk(tmp_path, state_as_context):
    source = _randomized(tiny_mot_config(state_as_context))
    ckpt = tmp_path / "ckpt"
    (ckpt / "model").mkdir(parents=True)
    torch.save(_with_buffers(source), ckpt / "model" / "pytorch_model_fsdp.bin")

    state, path = read_mot_state_dict(str(ckpt))
    assert detect_context_layout(state) == "context_embedder" and path.endswith("pytorch_model_fsdp.bin")
    target = build_mot_model(tiny_mot_config(state_as_context), dtype=torch.float64, device="cpu")
    report = load_mot_weights(target, str(ckpt), device="cpu")
    assert report["stripped"] == sorted(UNMODELED_KEYS["context_embedder"])
    assert report["context_layout"] == "context_embedder" and report["video_layout"] == "multiview"
    assert report["tensors_loaded"] == report["tensors_total"] - 3 and report["state_as_context"] is state_as_context
    x, timestep, y, mask, data_info = tiny_inputs(view_count=3)
    for key in ("x", "action_pred"):
        assert torch.equal(source(x, timestep, y, mask=mask, data_info=data_info)[key], target(x, timestep, y, mask=mask, data_info=data_info)[key])
    with pytest.raises(ValueError, match="state-conditioning"):
        load_mot_weights(MoTPolicyModel(tiny_mot_config(not state_as_context)).double(), state)
    with pytest.raises(ValueError, match="disagrees with the model built"):
        load_mot_weights(MoTPolicyModel(tiny_mot_config(state_as_context, video_layout="openwam_canvas", context_layout="legacy_action_mlp")).double(), state)
    drifted = dict(state)
    drifted["video_dit.pos_embed"] = torch.zeros(1, 9, 64)
    with pytest.raises(ValueError, match="pos_embed"):
        strip_unmodeled_mot_state(drifted, context_layout="context_embedder", input_size=2, hidden_size=64, model_max_length=8, caption_channels=32)


def test_legacy_checkpoint_round_trip(tmp_path):
    source = _randomized(tiny_mot_config(False, video_layout="openwam_canvas", context_layout="legacy_action_mlp"))
    state = _with_buffers(source)
    assert detect_context_layout(state) == "legacy_action_mlp"
    assert set(UNMODELED_KEYS["legacy_action_mlp"]) <= set(state)
    target = build_mot_model(tiny_mot_config(False, video_layout="openwam_canvas", context_layout="legacy_action_mlp"), dtype=torch.float64)
    report = load_mot_weights(target, state, source="in-memory")
    assert report["context_layout"] == "legacy_action_mlp" and report["tensors_loaded"] == report["tensors_total"] - 2
    x, timestep, y, mask, data_info = tiny_inputs(view_count=1, groups=1)
    assert torch.equal(source(x, timestep, y, mask=mask, data_info=data_info)["action_pred"], target(x, timestep, y, mask=mask, data_info=data_info)["action_pred"])


def test_transient_and_foreign_layouts_are_refused():
    state = _with_buffers(_randomized(tiny_mot_config(False)))
    transient = {k: v for k, v in state.items() if not k.startswith("context_embedder.action_y_embedder")}
    transient["context_embedder.text_proj.0.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="transient context layout"):
        detect_context_layout(transient)
    with pytest.raises(ValueError, match="not a MoT policy state dict"):
        detect_context_layout({"blocks.0.attn.qkv.weight": torch.zeros(1)})
    indices, video_softmax, action_softmax = checkpoint_layer_layout(state)
    assert indices == [0, 1, 2, 3] and video_softmax == action_softmax == (1, 3)


def _meta_counts(cfg):
    with torch.device("meta"):
        model = MoTPolicyModel(cfg)
    total = sum(p.numel() for p in model.parameters())
    return model, total, len(model.state_dict())


def test_real_scale_parameter_counts_on_the_meta_device():
    """Design doc: 5,436.18M parameters and 1,595 real keys (current layout) / 1,593 (606e48dd9); the mirror drops the buffers."""

    try:
        model, total, keys = _meta_counts(mot_config_from_train_config(load_train_config(CONFIG_MULTIVIEW)))
        legacy_model, legacy_total, legacy_keys = _meta_counts(
            mot_config_from_train_config(load_train_config(CONFIG_NSC_F25), context_layout="legacy_action_mlp")
        )
    except (RuntimeError, NotImplementedError) as exc:  # pragma: no cover - depends on the torch meta-kernel coverage
        pytest.skip(f"meta-device construction unsupported here: {exc}")
    assert keys + len(UNMODELED_KEYS["context_embedder"]) == 1595
    assert legacy_keys + len(UNMODELED_KEYS["legacy_action_mlp"]) == 1593
    for value in (total, legacy_total):
        assert abs(value - 5_436_180_000) < 3_000_000, value
    context = sum(p.numel() for n, p in model.named_parameters() if n.startswith("context_embedder."))
    assert abs(context - 15_870_000) < 50_000, context                             # "context embedder 15.87M"
    assert model.softmax_layer_indices == legacy_model.softmax_layer_indices == (3, 7, 11, 15, 19, 23, 27, 31)
