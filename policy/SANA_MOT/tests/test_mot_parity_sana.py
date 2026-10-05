"""Parity of the MoT mirror (current text layout) against Sana's live ``SanaRWMMoTAttnResPolicy`` at the rwm/mot tip.

Skipped when a Sana checkout with the current MoT policy is not importable: point ``SANA_MOT_REPO`` at one (default
``~/zekail/Sana_mot``; validated against b3b9e0e9e -- run it with ``SANA_MOT_REPO`` at a worktree of that revision:
the branch tip replaced ``video_layout`` with ``data.extra.multiview`` and is covered by test_mot_head_parity.py). The tiny live model (the dims of Sana's
manual_cpu_selfcheck_mot_policy.py) is built in fp64 with its zero-initialized IO projections randomized, its state dict
is loaded STRICTLY into the mirror after the three unmodeled buffers are stripped, and ``x`` / ``action_pred`` are
compared on identical inputs: multiview V=3 (semantic_2x2 and local_reset), multiview V=1, the canvas switch, strided
video (strides 2 and 4), the fp32-attention path, in both state-conditioning modes; the two models must refuse the same
malformed inputs. Past revisions (the 606e48dd9 legacy text layout, the 20c4d63d9 pre-switch canvas) are covered by
test_mot_reference_parity.py.
"""

from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("DISABLE_XFORMERS", "1")

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
SANA_MOT_REPO = os.environ.get("SANA_MOT_REPO", os.path.expanduser("~/zekail/Sana_mot"))
if SANA_MOT_REPO not in sys.path and os.path.isdir(SANA_MOT_REPO):
    sys.path.insert(0, SANA_MOT_REPO)

from _tiny_mot import (  # noqa: E402
    ACTION_CROSS_HEADS,
    ACTION_HIDDEN,
    CAP_CH,
    DEPTH,
    HEADS,
    HIDDEN,
    IN_CH,
    INPUT_SIZE,
    LHD,
    MML,
    SHD,
    TILE,
    tiny_inputs,
    tiny_mot_config,
)

live = pytest.importorskip(
    "dev.rwm.diffusion.model.nets.sana_qwennext_mot_policy", reason="Sana rwm/mot checkout not importable (set SANA_MOT_REPO)"
)
if not hasattr(live, "VIDEO_LAYOUTS"):
    pytest.skip("SANA_MOT_REPO holds a pre-multiview rwm/mot revision; point it at the branch tip", allow_module_level=True)

from sana_mot_min.mot_model import MoTPolicyModel, load_mot_state_dict  # noqa: E402
from sana_mot_min.mot_model.checkpoint import (  # noqa: E402
    UNMODELED_KEYS,
    checkpoint_layer_layout,
    detect_context_layout,
    normalize_action_attention_keys,
)

ATOL = 1e-10
MODES = [False, True]


def build_live(video_layout: str, state_as_context: bool, rope_layout: str = "semantic_2x2"):
    torch.manual_seed(20260914)
    model = live.SanaRWMMoTAttnResPolicy(
        action_hidden_size=ACTION_HIDDEN,
        action_cross_attn_heads=ACTION_CROSS_HEADS,
        action_attn_res_block_size=2,
        action_state_as_context=state_as_context,
        video_donor=None,
        video_layout=video_layout,
        multiview_spatial_rope_layout=rope_layout,
        multiview_spatial_rope_tile_shape=TILE,
        config=SimpleNamespace(work_dir=None, model=SimpleNamespace(extra={"action_dim": 80}), vae=SimpleNamespace(vae_stride=(8, 32, 32))),
        input_size=INPUT_SIZE,
        patch_size=(1, 1, 1),
        in_channels=IN_CH,
        hidden_size=HIDDEN,
        depth=DEPTH,
        num_heads=HEADS,
        mlp_ratio=2.0,
        caption_channels=CAP_CH,
        model_max_length=MML,
        qk_norm=True,
        cross_norm=True,
        y_norm=True,
        class_dropout_prob=0.1,
        linear_attn_type="GatedDeltaNet",
        softmax_attn_type="GatedSoftmaxAttention",
        softmax_layer_indices=[1, 3],
        ffn_type="SwiGLU",
        use_pe=True,
        pos_embed_type="wan_rope",
        linear_head_dim=LHD,
        softmax_head_dim=SHD,
        use_attn_res=True,
        attn_res_block_size=2,
        use_time_conditioning=False,
        pred_sigma=False,
    ).double()
    with torch.no_grad():
        # zero-initialized IO modules would hide the state / action / state-context paths from the comparison
        torch.nn.init.normal_(model.action_dit.action_head.linear.weight, std=0.02)
        torch.nn.init.normal_(model.action_dit.action_head.linear.bias, std=0.02)
        projectors = [model.action_dit.action_embed]
        projectors.append(model.context_embedder.state_proj if state_as_context else model.action_dit.state_embed)
        for projector in projectors:
            torch.nn.init.normal_(projector.proj.weight, std=0.05)
        for expert in (model.video_dit, model.action_dit):
            for proj in (expert.attn_res.attn_proj, expert.attn_res.mlp_proj, expert.attn_res.final_proj):
                torch.nn.init.normal_(proj.weight, std=0.5)
        for norm in (model.context_embedder.y_norm, model.context_embedder.action_y_norm):
            norm.weight.uniform_(0.5, 1.5)
    return model.eval()


def build_mirror_from(live_model, video_layout: str, state_as_context: bool, rope_layout: str = "semantic_2x2") -> MoTPolicyModel:
    mirror = MoTPolicyModel(tiny_mot_config(state_as_context, video_layout=video_layout, rope_layout=rope_layout)).double()
    load_mot_state_dict(mirror, live_model.state_dict())
    return mirror.eval()


def _compare(live_model, mirror, x, timestep, y, mask, data_info, atol=ATOL):
    with torch.no_grad():
        ref = live_model(x, timestep, y, mask=mask, data_info=data_info)
        out = mirror(x, timestep, y, mask=mask, data_info=data_info)
    for key in ("x", "action_pred"):
        assert ref[key].shape == out[key].shape, key
        diff = (ref[key] - out[key]).abs().max().item()
        assert torch.allclose(ref[key], out[key], rtol=0, atol=atol), f"{key} max|d|={diff:.3e}"
    assert out["action_pred"].abs().sum() > 0 and torch.isfinite(out["x"]).all()
    return ref, out


@pytest.mark.parametrize("state_as_context", MODES)
def test_state_dict_layout_and_strict_load(state_as_context):
    live_model = build_live("multiview", state_as_context)
    state = live_model.state_dict()
    assert detect_context_layout(state) == "context_embedder"
    assert set(UNMODELED_KEYS["context_embedder"]) <= set(state)
    indices, video_softmax, action_softmax = checkpoint_layer_layout(state)
    assert indices == list(range(DEPTH)) and video_softmax == (1, 3) and action_softmax == (1, 3)
    mirror = build_mirror_from(live_model, "multiview", state_as_context)
    # rwm/mot 3058e5785 renamed action_block.attn_head -> attn (key names only); the mirror carries the current name and
    # the loader renames the keys of checkpoints written before it
    renamed, count = normalize_action_attention_keys(state)
    assert count == (0 if any(".action_block.attn." in key for key in state) else sum(".action_block.attn_head." in k for k in state))
    assert set(mirror.state_dict()) == {k for k in renamed if k not in UNMODELED_KEYS["context_embedder"]}
    for key, tensor in mirror.state_dict().items():
        assert torch.equal(tensor, renamed[key]), key
    assert ("context_embedder.state_proj.proj.weight" in state) == state_as_context
    with pytest.raises(RuntimeError):
        load_mot_state_dict(MoTPolicyModel(tiny_mot_config(not state_as_context)).double(), state)
    # the canvas switch has the same parameters: one checkpoint layout, two forward contracts
    assert set(build_live("openwam_canvas", state_as_context).state_dict()) == set(state)


@pytest.mark.parametrize("state_as_context", MODES)
@pytest.mark.parametrize("rope_layout", ["semantic_2x2", "local_reset"])
def test_multiview_three_views(state_as_context, rope_layout):
    live_model = build_live("multiview", state_as_context, rope_layout)
    mirror = build_mirror_from(live_model, "multiview", state_as_context, rope_layout)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=3)
    ref, out = _compare(live_model, mirror, x, timestep, y, mask, data_info)
    assert out["x"].shape == x.shape                               # the strip comes back as a strip
    _compare(live_model, mirror, x, timestep.squeeze(1), y, mask, data_info)
    _compare(live_model, mirror, x, timestep, y, None, data_info)
    _compare(live_model, mirror, x, timestep, y, mask[:1], data_info)       # batch-broadcast mask
    other_slots = dict(data_info, view_slot_ids=torch.tensor([[3, 1, 0]] * x.shape[0]))
    _compare(live_model, mirror, x, timestep, y, mask, other_slots)
    # each view reads its own text group: changing group 1 moves the video, the robot group moves the action
    y_view = y.clone()
    y_view[:, 1] = torch.randn_like(y[:, 1])
    moved, _ = _compare(live_model, mirror, x, timestep, y_view, mask, data_info)
    assert not torch.allclose(moved["x"], ref["x"])


@pytest.mark.parametrize("state_as_context", MODES)
def test_multiview_single_view_native_grid(state_as_context):
    live_model = build_live("multiview", state_as_context)
    mirror = build_mirror_from(live_model, "multiview", state_as_context)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=1, seed=2)
    _compare(live_model, mirror, x, timestep, y, mask, data_info)
    _compare(live_model, mirror, x, timestep, y, mask, dict(data_info, view_count=1))


@pytest.mark.parametrize("state_as_context", MODES)
def test_canvas_switch(state_as_context):
    live_model = build_live("openwam_canvas", state_as_context)
    mirror = build_mirror_from(live_model, "openwam_canvas", state_as_context)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=1, groups=1, seed=3)
    _compare(live_model, mirror, x, timestep, y, mask, data_info)
    _compare(live_model, mirror, x, timestep, y[:, 0], mask[:, 0], data_info)      # the base text layout
    _compare(live_model, mirror, x, timestep, y, None, dict(data_info, view_count=1))


@pytest.mark.parametrize("state_as_context", MODES)
@pytest.mark.parametrize("stride", [2, 4])
def test_strided_video(state_as_context, stride):
    live_model = build_live("multiview", state_as_context)
    mirror = build_mirror_from(live_model, "multiview", state_as_context)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=3, seed=4, stride=stride)
    _, out = _compare(live_model, mirror, x, timestep, y, mask, data_info)
    assert out["action_pred"].shape[1] == 8 * stride
    canvas_live = build_live("openwam_canvas", state_as_context)
    canvas_mirror = build_mirror_from(canvas_live, "openwam_canvas", state_as_context)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=1, groups=1, seed=5, stride=stride)
    _compare(canvas_live, canvas_mirror, x, timestep, y, mask, data_info)


def test_explicit_stride_one_equals_dense():
    live_model = build_live("multiview", False)
    mirror = build_mirror_from(live_model, "multiview", False)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=3, seed=6)
    _, dense = _compare(live_model, mirror, x, timestep, y, mask, data_info)
    _, explicit = _compare(live_model, mirror, x, timestep, y, mask, dict(data_info, video_frame_stride=1))
    assert torch.equal(dense["action_pred"], explicit["action_pred"])


@pytest.mark.parametrize("video_layout", ["multiview", "openwam_canvas"])
def test_fp32_attention_path(video_layout):
    from diffusion.model.utils import set_fp32_attention

    live_model = build_live(video_layout, False)
    set_fp32_attention(live_model)
    mirror = build_mirror_from(live_model, video_layout, False)
    mirror.set_fp32_attention(True)
    views = 3 if video_layout == "multiview" else 1
    x, timestep, y, mask, data_info = tiny_inputs(view_count=views, groups=None if views == 3 else 1, seed=7)
    _compare(live_model, mirror, x, timestep, y, mask, data_info, atol=1e-6)


def test_mirror_refuses_what_the_live_model_refuses():
    multiview_live = build_live("multiview", False)
    multiview = build_mirror_from(multiview_live, "multiview", False)
    canvas_live = build_live("openwam_canvas", False)
    canvas = build_mirror_from(canvas_live, "openwam_canvas", False)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=3)
    cases = [
        ("G = V + 1", (multiview_live, multiview), (x, timestep, y[:, :3], mask[:, :3], data_info)),
        ("equal view tiles", (multiview_live, multiview), (x, timestep, y, mask, dict(data_info, view_latent_shape=torch.tensor([[[2, 3], [3, 2], [2, 3]]] * 2)))),
        ("unique IDs", (multiview_live, multiview), (x, timestep, y, mask, dict(data_info, view_slot_ids=torch.tensor([[0, 0, 3]] * 2)))),
        ("camera", (multiview_live, multiview), (x, timestep, y, mask, dict(data_info, camera_conditioning_enabled=torch.ones(2, dtype=torch.bool)))),
        ("motion80 row", (multiview_live, multiview), (x, timestep, y, mask, dict(data_info, video_frame_stride=torch.full((2,), 2)))),
        ("view_count", (canvas_live, canvas), (x, timestep, y[:, :1], mask[:, :1], data_info)),
    ]
    for needle, models, args in cases:
        for model in models:
            with pytest.raises(ValueError, match=re.escape(needle)):
                model(args[0], args[1], args[2], mask=args[3], data_info=args[4])
    x1, timestep1, y1, mask1, info1 = tiny_inputs(view_count=1, groups=2)
    for model in (canvas_live, canvas):
        with pytest.raises(ValueError, match="G=1"):
            model(x1, timestep1, y1, mask=mask1, data_info=info1)
