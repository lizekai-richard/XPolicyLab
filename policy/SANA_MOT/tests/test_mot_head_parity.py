"""Parity of the MoT mirror with Sana's live ``SanaRWMMoTAttnResPolicy`` at the rwm/mot tip (>= 71ac93f43).

The 2026-09-21..23 contracts: ``multiview`` decides the caption path at build time (``sana_latent``: one caption embedder
per expert, G = V + 1; ``openwam`` / ``sana_pixel``: ONE shared caption embedder, G = 1, the action blocks'
``kv_linear`` reading the 2560-wide -- here HIDDEN-wide -- embedding) and ``rope`` places the streams on the RoPE clock
(``aligned``: the video frame stride folded into the video time axis, the robot rows on it; ``independent``: the action
expert's own 1D clock, state 0 / actions 1 .. S, 0 .. S-1 under state_as_context). The tiny live model is built in fp64
with its zero-initialized IO projections randomized, its state dict strict-loads into the mirror (the tip names the
action attention ``attn``), and ``x`` / ``action_pred`` are compared at 1e-10. Point ``SANA_MOT_REPO`` at a tip checkout
(default the 71ac93f43 worktree ~/zekail/Sana_mot_71ac93f43); run this file on its own (the first parity file of a session owns the ``dev`` package).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("DISABLE_XFORMERS", "1")

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
# pinned to the tree whose contract this file covers (rwm/mot 71ac93f43: canvas modes with the SHARED caption embedder);
# ~/zekail/Sana_mot moves with every pull and the later contract (b010eb9a5 on) is test_mot_tip_parity.py's
SANA_MOT_REPO = os.environ.get("SANA_MOT_REPO", os.path.expanduser("~/zekail/Sana_mot_71ac93f43"))
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
    tiny_inputs,
    tiny_video_config,
)

live = pytest.importorskip(
    "dev.rwm.diffusion.model.nets.sana_qwennext_mot_policy", reason="Sana rwm/mot checkout not importable (set SANA_MOT_REPO)"
)
if not os.path.abspath(live.__file__).startswith(os.path.abspath(SANA_MOT_REPO)):
    pytest.skip("the dev package belongs to another Sana checkout; run this file on its own", allow_module_level=True)
if "rope" not in live.SanaRWMMoTAttnResPolicy.__init__.__code__.co_varnames:
    pytest.skip("SANA_MOT_REPO predates the MoT RoPE modes (71ac93f43); point it at the branch tip", allow_module_level=True)

from sana_mot_min.mot_model import MoTPolicyModel, load_mot_state_dict  # noqa: E402
from sana_mot_min.mot_model.checkpoint import UNMODELED_KEYS, detect_context_layout  # noqa: E402
from sana_mot_min.mot_model.model import MoTConfig  # noqa: E402

ATOL = 1e-10
LAYOUT = {"sana_latent": "multiview", "openwam": "openwam_canvas", "sana_pixel": "sana_pixel_canvas"}


def build_live(multiview: str, rope: str, state_as_context: bool):
    torch.manual_seed(20260923)
    model = live.SanaRWMMoTAttnResPolicy(
        action_hidden_size=ACTION_HIDDEN,
        action_cross_attn_heads=ACTION_CROSS_HEADS,
        action_attn_res_block_size=2,
        action_state_as_context=state_as_context,
        multiview=multiview,
        rope=rope,
        video_donor=None,
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
            if norm is not None:
                norm.weight.uniform_(0.5, 1.5)
    return model.eval()


# the tip's fixed strip tile: (15, 30) up to rwm/mot 66ded97a6, (8, 16) since (no yaml key; the mirror must match it)
try:
    from dev.rwm.diffusion.multiview_utils import MULTIVIEW_SPATIAL_ROPE_TILE_SHAPE as LIVE_TILE
except ImportError:
    LIVE_TILE = (15, 30)
LIVE_TILE = tuple(int(v) for v in LIVE_TILE)


def mirror_config(multiview: str, rope: str, state_as_context: bool) -> MoTConfig:
    return MoTConfig(
        video=tiny_video_config(),
        action_hidden_size=ACTION_HIDDEN,
        action_mlp_ratio=4.0,
        action_cross_attn_heads=ACTION_CROSS_HEADS,
        action_rope_theta=10000.0,
        action_attn_res_block_size=2,
        action_state_as_context=state_as_context,
        video_layout=LAYOUT[multiview],
        multiview_spatial_rope_layout="semantic_2x2",
        multiview_spatial_rope_tile_shape=LIVE_TILE,
        context_layout="context_embedder" if multiview == "sana_latent" else "shared_caption_embedder",
        canvas_text_groups=1,
        rope=rope,
    )


def twins(multiview: str, rope: str, state_as_context: bool):
    live_model = build_live(multiview, rope, state_as_context)
    mirror = MoTPolicyModel(mirror_config(multiview, rope, state_as_context)).double()
    load_mot_state_dict(mirror, live_model.state_dict())
    return live_model, mirror.eval()


def compare(live_model, mirror, inputs, fps=None):
    x, timestep, y, mask, data_info = inputs
    if fps is not None:
        data_info = dict(data_info, model_fps=fps)
    with torch.no_grad():
        ref = live_model(x, timestep, y, mask=mask, data_info=data_info)
        out = mirror(x, timestep, y, mask=mask, data_info=data_info)
    for key in ("x", "action_pred"):
        assert ref[key].shape == out[key].shape, key
        diff = (ref[key] - out[key]).abs().max().item()
        assert torch.allclose(ref[key], out[key], rtol=0, atol=ATOL), f"{key} max|d|={diff:.3e}"
    assert out["action_pred"].abs().sum() > 0
    return ref, out


@pytest.mark.parametrize("state_as_context", [False, True])
@pytest.mark.parametrize("rope", ["aligned", "independent"])
@pytest.mark.parametrize("stride", [1, 2, 4])
def test_sana_latent_strip(rope, stride, state_as_context):
    live_model, mirror = twins("sana_latent", rope, state_as_context)
    inputs = tiny_inputs(view_count=3, seed=10 + stride, stride=stride)
    for fps in (16.0, 25.0):
        compare(live_model, mirror, inputs, fps)


@pytest.mark.parametrize("multiview", ["openwam", "sana_pixel"])
@pytest.mark.parametrize("state_as_context", [False, True])
@pytest.mark.parametrize("rope", ["aligned", "independent"])
@pytest.mark.parametrize("stride", [1, 4])
def test_canvas_with_the_shared_caption_embedder(multiview, rope, stride, state_as_context):
    live_model, mirror = twins(multiview, rope, state_as_context)
    inputs = tiny_inputs(view_count=1, groups=1, seed=20 + stride, stride=stride)
    for fps in (16.0, 25.0):
        compare(live_model, mirror, inputs, fps)


@pytest.mark.parametrize("state_as_context", [False, True])
def test_key_layouts_of_the_two_builds(state_as_context):
    strip = build_live("sana_latent", "independent", state_as_context).state_dict()
    canvas = build_live("openwam", "independent", state_as_context).state_dict()
    assert detect_context_layout(strip) == "context_embedder" and detect_context_layout(canvas) == "shared_caption_embedder"
    assert not any(key.startswith("context_embedder.action_") for key in canvas)
    assert any(".action_block.attn." in key for key in canvas) and not any("attn_head" in key for key in canvas)
    kv = canvas["blocks.0.action_block.cross_attn.kv_linear.weight"]
    assert tuple(kv.shape) == (2 * ACTION_HIDDEN, HIDDEN)                      # the video-width shared embedding
    assert tuple(strip["blocks.0.action_block.cross_attn.kv_linear.weight"].shape) == (2 * ACTION_HIDDEN, ACTION_HIDDEN)
    mirror = MoTPolicyModel(mirror_config("openwam", "independent", state_as_context)).double()
    load_mot_state_dict(mirror, canvas)
    assert set(mirror.state_dict()) == {key for key in canvas if key not in UNMODELED_KEYS["shared_caption_embedder"]}
    with pytest.raises(RuntimeError):
        load_mot_state_dict(MoTPolicyModel(mirror_config("sana_latent", "independent", state_as_context)).double(), canvas)


def test_the_two_rope_modes_differ_and_both_models_refuse_the_same_text():
    aligned_live, aligned = twins("sana_latent", "aligned", False)
    _, independent = twins("sana_latent", "independent", False)
    inputs = tiny_inputs(view_count=3, seed=31, stride=2)
    with torch.no_grad():
        a = aligned(*inputs[:3], mask=inputs[3], data_info=dict(inputs[4], model_fps=25.0))["action_pred"]
        b = independent(*inputs[:3], mask=inputs[3], data_info=dict(inputs[4], model_fps=25.0))["action_pred"]
    assert not torch.allclose(a, b)
    canvas_live, canvas = twins("sana_pixel", "independent", False)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=1, groups=2, seed=32)
    for model in (canvas_live, canvas):
        with pytest.raises(ValueError, match="G=2"):
            model(x, timestep, y, mask=mask, data_info=data_info)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=3, groups=2, seed=33)
    for model in (aligned_live, aligned):
        with pytest.raises(ValueError):
            model(x, timestep, y, mask=mask, data_info=data_info)
