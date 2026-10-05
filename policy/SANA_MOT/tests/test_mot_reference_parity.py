"""Parity of the MoT mirror against outputs recorded from PAST Sana rwm/mot revisions (made by
``tools/make_reference_parity_fixture.py``); the branch tip no longer carries their code verbatim.

* ``fixtures/legacy_606e48d_parity.pt`` -- the legacy text layout (``action_dit.context_mlp``), canvas-only. Carried by
  checkpoints of runs launched before the ContextEmbedder refactor (the NSC canvas run's epoch-5 snapshot
  ``mot_canvas_nsc_s15k``).
* ``fixtures/canvas_20c4d63d9_parity.pt`` -- the ``context_embedder`` layout while the OpenWAM canvas was the only video
  layout (8fd95e219 .. 33b220373; state token in BOTH experts' text under state-as-context). Carried by the NSC f25
  canvas runs at step 36,250, whose yamls have no ``model.extra.video_layout`` -- see
  ``test_pre_switch_canvas_yaml_resolves_to_the_canvas`` in ``test_mot_model_cpu.py``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from _tiny_mot import FIXTURES  # noqa: E402

from sana_mot_min.mot_model import MoTConfig, MoTPolicyModel, detect_context_layout, load_mot_state_dict  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402

REFERENCES = {
    "606e48dd9": os.path.join(FIXTURES, "legacy_606e48d_parity.pt"),
    "20c4d63d9": os.path.join(FIXTURES, "canvas_20c4d63d9_parity.pt"),
}
CONTEXT_LAYOUTS = {"606e48dd9": "legacy_action_mlp", "20c4d63d9": "context_embedder"}


@pytest.fixture(scope="module", params=sorted(REFERENCES))
def reference(request):
    return request.param, torch.load(REFERENCES[request.param], map_location="cpu", weights_only=True)


def build_canvas_mirror(dims: dict, state_as_context: bool, context_layout: str) -> MoTPolicyModel:
    video = PolicyConfig.from_sana_kwargs(
        depth=dims["depth"],
        hidden_size=dims["hidden_size"],
        patch_size=(1, 1, 1),
        num_heads=dims["num_heads"],
        input_size=dims["input_size"],
        in_channels=dims["in_channels"],
        caption_channels=dims["caption_channels"],
        model_max_length=dims["model_max_length"],
        mlp_ratio=dims["mlp_ratio"],
        linear_head_dim=dims["linear_head_dim"],
        softmax_head_dim=dims["softmax_head_dim"],
        softmax_layer_indices=dims["softmax_layer_indices"],
        attn_res_block_size=dims["attn_res_block_size"],
        qk_norm=True,
        cross_norm=True,
        y_norm=True,
        pred_sigma=False,
        use_fp32_attention=False,
        config=SimpleNamespace(model=SimpleNamespace(), vae=SimpleNamespace(vae_stride=[8, 32, 32])),
    )
    config = MoTConfig(
        video=video,
        action_hidden_size=dims["action_hidden_size"],
        action_cross_attn_heads=dims["action_cross_attn_heads"],
        action_attn_res_block_size=dims["attn_res_block_size"],
        action_state_as_context=state_as_context,
        video_layout="openwam_canvas",
        context_layout=context_layout,
    )
    return MoTPolicyModel(config).double()


@pytest.mark.parametrize("mode", ["state_token", "state_context"])
def test_canvas_forward_matches_the_recorded_revision(reference, mode):
    revision, record = reference
    assert record["meta"]["source"] == f"Sana rwm/mot @ {revision}"
    entry = record["modes"][mode]
    state = {k: v.double() for k, v in entry["state_dict"].items()}
    layout = detect_context_layout(state)
    assert layout == CONTEXT_LAYOUTS[revision]
    mirror = build_canvas_mirror(record["meta"]["dims"], entry["state_as_context"], layout)
    load_mot_state_dict(mirror, state)
    mirror.eval()
    for case in entry["cases"]:
        with torch.no_grad():
            out = mirror(entry["x"], entry["timestep"], case["y"], mask=case["mask"], data_info=entry["data_info"])
        for key, want in (("x", case["x_out"]), ("action_pred", case["action_pred"])):
            assert want.abs().mean() > 1e-3, f"{revision}/{mode}/{case['name']} {key}: degenerate reference output"
            diff = (out[key] - want).abs().max().item()
            assert torch.allclose(out[key], want, rtol=0, atol=1e-10), f"{revision}/{mode}/{case['name']} {key} max|d|={diff:.3e}"


def test_state_token_joins_both_experts_text_under_the_canvas():
    """20c4d63d9 (and the tip's canvas switch) append the state token to the ONE shared prompt: both experts' text
    contexts get L + 1 tokens, the extra one always valid and driven by the state. (The forward outputs cannot show
    this -- the joint self-attention carries the state into the video stream either way.)"""

    record = torch.load(REFERENCES["20c4d63d9"], map_location="cpu", weights_only=True)
    entry = record["modes"]["state_context"]
    state = {k: v.double() for k, v in entry["state_dict"].items()}
    mirror = build_canvas_mirror(record["meta"]["dims"], True, "context_embedder")
    load_mot_state_dict(mirror, state)
    case = entry["cases"][0]
    batch, length = case["y"].shape[0], case["y"].shape[-2]
    state80 = entry["data_info"]["initial_state80"].reshape(batch, 1, -1)
    state_mask80 = entry["data_info"]["initial_state_condition_mask80"].reshape(batch, 1, -1)
    with torch.no_grad():
        video, action, video_mask, action_mask = mirror.context_embedder(case["y"], case["mask"], state80, state_mask80, shared_prompt=True)
        video_moved, action_moved, _, _ = mirror.context_embedder(case["y"], case["mask"], state80 + 1.0, state_mask80, shared_prompt=True)
    assert video.shape[:3] == (batch, 1, length + 1) and action.shape[:2] == (batch, length + 1)
    assert video_mask[:, 0, -1].eq(1).all() and action_mask[:, -1].eq(1).all()
    assert torch.equal(video_mask[:, 0, :-1], case["mask"].reshape(batch, length).to(video_mask.dtype))
    for before, after in ((video[:, 0], video_moved[:, 0]), (action, action_moved)):
        assert torch.equal(before[:, :-1], after[:, :-1])
        assert not torch.allclose(before[:, -1], after[:, -1], rtol=0, atol=1e-8)


def test_legacy_layout_is_canvas_only():
    dims = {"depth": 2, "hidden_size": 32, "num_heads": 2, "input_size": 2, "in_channels": 4, "caption_channels": 16,
            "model_max_length": 6, "mlp_ratio": 2.0, "linear_head_dim": 16, "softmax_head_dim": 16,
            "softmax_layer_indices": [1], "attn_res_block_size": 1, "action_hidden_size": 16, "action_cross_attn_heads": 2}
    mirror = build_canvas_mirror(dims, False, "legacy_action_mlp")
    with pytest.raises(ValueError, match="existed only for the OpenWAM canvas"):
        MoTConfig(video=mirror.mot_config.video, action_hidden_size=16, action_cross_attn_heads=2,
                  action_attn_res_block_size=1, video_layout="multiview", context_layout="legacy_action_mlp").validate()
    assert hasattr(mirror.action_dit, "context_mlp") and not hasattr(mirror, "context_embedder")
    assert hasattr(mirror.video_dit, "y_embedder") and hasattr(mirror.video_dit, "attention_y_norm")
