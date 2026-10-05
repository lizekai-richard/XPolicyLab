"""Parity of the MoT mirror with Sana's live ``SanaRWMMoTAttnResPolicy`` from rwm/mot b010eb9a5 on (2026-09-23).

b010eb9a5 gave every multiview mode one caption embedder per expert again: a canvas mode (``openwam`` / ``sana_pixel``)
still ships ONE prompt (G = 1) and each expert projects it through its OWN embedder (the donor's path for the video
expert, the fresh action-width path for the action expert; under state_as_context both contexts carry the state token),
so a canvas and a ``sana_latent`` build have the same keys and shapes. This is the tree of the first MoT checkpoint of the
new contracts, logits/sft_robodojo_mot_eefabs_sana_pixel_320x480_sanavideo_independent_f33fps8 (pin e54ec7d97). The tiny
live model is built in fp64 with its zero-initialized IO projections randomized, its state dict strict-loads into the
mirror (``context_embedder`` layout, canvas_text_groups 1) and ``x`` / ``action_pred`` are compared at 1e-10. Point
``SANA_MOT_TIP_REPO`` at a checkout >= b010eb9a5 (default ~/zekail/Sana_mot_e54ec7d97); run this file on its own (the
first parity file of a session owns the ``dev`` package).
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
SANA_MOT_REPO = os.environ.get("SANA_MOT_TIP_REPO", os.path.expanduser("~/zekail/Sana_mot_e54ec7d97"))
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
    "dev.rwm.diffusion.model.nets.sana_qwennext_mot_policy", reason="Sana rwm/mot checkout >= b010eb9a5 not importable (set SANA_MOT_TIP_REPO)"
)
if not os.path.abspath(live.__file__).startswith(os.path.abspath(SANA_MOT_REPO)):
    pytest.skip("the dev package belongs to another Sana checkout; run this file on its own", allow_module_level=True)
context_ref = pytest.importorskip("dev.rwm.diffusion.model.layers.mot_context_embedder")
if "single_prompt" not in context_ref.ContextEmbedder.__init__.__code__.co_varnames:
    pytest.skip("SANA_MOT_TIP_REPO predates b010eb9a5 (one caption embedder per expert in every mode)", allow_module_level=True)

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


# the live tree's fixed strip tile: (15, 30) up to rwm/mot 66ded97a6, (8, 16) since (no yaml key; the mirror must match it)
try:
    from dev.rwm.diffusion.multiview_utils import MULTIVIEW_SPATIAL_ROPE_TILE_SHAPE as LIVE_TILE
except ImportError:  # trees before the constant existed
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
        context_layout="context_embedder",
        canvas_text_groups=1,
        rope=rope,
    )


def twins(multiview: str, rope: str, state_as_context: bool):
    live_model = build_live(multiview, rope, state_as_context)
    state = live_model.state_dict()
    assert detect_context_layout(state) == "context_embedder"
    mirror = MoTPolicyModel(mirror_config(multiview, rope, state_as_context)).double()
    load_mot_state_dict(mirror, state)
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


@pytest.mark.parametrize("multiview", ["openwam", "sana_pixel"])
@pytest.mark.parametrize("state_as_context", [False, True])
@pytest.mark.parametrize("rope", ["aligned", "independent"])
@pytest.mark.parametrize("stride", [1, 4])
def test_canvas_with_one_caption_embedder_per_expert(multiview, rope, stride, state_as_context):
    live_model, mirror = twins(multiview, rope, state_as_context)
    inputs = tiny_inputs(view_count=1, groups=1, seed=40 + stride, stride=stride)
    for fps in (16.0, 25.0):
        compare(live_model, mirror, inputs, fps)


@pytest.mark.parametrize("state_as_context", [False, True])
@pytest.mark.parametrize("rope", ["aligned", "independent"])
@pytest.mark.parametrize("stride", [1, 4])
def test_sana_latent_strip_at_the_tip(rope, stride, state_as_context):
    live_model, mirror = twins("sana_latent", rope, state_as_context)
    inputs = tiny_inputs(view_count=3, seed=50 + stride, stride=stride)
    for fps in (16.0, 25.0):
        compare(live_model, mirror, inputs, fps)


@pytest.mark.parametrize("state_as_context", [False, True])
def test_every_mode_has_one_schema(state_as_context):
    strip = build_live("sana_latent", "independent", state_as_context).state_dict()
    for multiview in ("openwam", "sana_pixel"):
        canvas = build_live(multiview, "independent", state_as_context).state_dict()
        assert {k: tuple(v.shape) for k, v in canvas.items()} == {k: tuple(v.shape) for k, v in strip.items()}
        kv = canvas["blocks.0.action_block.cross_attn.kv_linear.weight"]
        assert tuple(kv.shape) == (2 * ACTION_HIDDEN, ACTION_HIDDEN)            # the action expert's own width again
        mirror = MoTPolicyModel(mirror_config(multiview, "independent", state_as_context)).double()
        load_mot_state_dict(mirror, canvas)
        assert set(mirror.state_dict()) == {key for key in canvas if key not in UNMODELED_KEYS["context_embedder"]}


def test_a_canvas_refuses_the_grouped_prompt():
    live_model, mirror = twins("sana_pixel", "independent", False)
    x, timestep, y, mask, data_info = tiny_inputs(view_count=1, groups=2, seed=61, stride=4)
    with torch.no_grad(), pytest.raises(ValueError):
        live_model(x, timestep, y, mask=mask, data_info=data_info)
    with torch.no_grad(), pytest.raises(ValueError):
        mirror(x, timestep, y, mask=mask, data_info=data_info)
