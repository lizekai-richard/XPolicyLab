"""CPU tests for sana_wam_min.config and the vendored policy_model forward."""

from __future__ import annotations

import os
import re
import sys

import pytest
import torch

# policy/SANA_WAM (plain ``sana_wam_min`` imports) and the XPolicyLab parent (``XPolicyLab.policy.SANA_WAM`` imports).
ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(ADAPTER_DIR, "..", "..", ".."))
for _p in (ADAPTER_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _tiny_policy import (  # noqa: E402
    FRAMES,
    IN_CH,
    ROBOT_DIM,
    VIEW_H,
    VIEW_W,
    tiny_inputs,
    tiny_policy_config,
)
from sana_wam_min import config as wam_config  # noqa: E402
from sana_wam_min.checkpoint import build_policy_model  # noqa: E402

# The resolved training yaml of the RoboDojo 320px joint-only line (see the header of the fixture).
TRAIN_YAML = os.path.join(os.path.dirname(__file__), "fixtures", "config.yaml")
PACKAGE_DIR = os.path.join(ADAPTER_DIR, "sana_wam_min")


def test_both_import_paths_resolve_to_the_same_package() -> None:
    import XPolicyLab.policy.SANA_WAM.sana_wam_min as qualified
    import sana_wam_min as plain

    assert qualified.__version__ == plain.__version__
    assert os.path.samefile(os.path.dirname(qualified.__file__), os.path.dirname(plain.__file__))


def test_no_forbidden_imports_in_package() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+(dev|diffusion|sana|xformers|XPolicyLab\.policy\.(?!SANA_WAM))\b")
    offenders = []
    for root, _, files in os.walk(PACKAGE_DIR):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path) as handle:
                for number, line in enumerate(handle, 1):
                    if pattern.match(line):
                        offenders.append(f"{path}:{number}: {line.strip()}")
    assert offenders == []


def test_vendored_attention_never_uses_xformers() -> None:
    from sana_wam_min.policy_model import attention

    assert attention._xformers_ops is None
    assert not hasattr(attention, "_DISABLE_XFORMERS")


@pytest.mark.skipif(not os.path.exists(TRAIN_YAML), reason="train yaml not on this host")
def test_policy_config_from_production_yaml() -> None:
    cfg = wam_config.load_train_config(TRAIN_YAML)
    policy_cfg = wam_config.policy_config_from_train_config(cfg)
    assert policy_cfg.input_size == 10
    assert policy_cfg.depth == 32
    assert policy_cfg.hidden_size == 2560
    assert policy_cfg.num_heads == 20
    assert policy_cfg.patch_size == (1, 1, 1)
    assert policy_cfg.in_channels == 128
    assert policy_cfg.out_channels == 128
    assert policy_cfg.caption_channels == 2304
    assert policy_cfg.model_max_length == 300
    assert policy_cfg.mlp_ratio == 4.0
    assert policy_cfg.linear_head_dim == 128
    assert policy_cfg.softmax_head_dim == 256
    assert policy_cfg.softmax_layer_indices == (3, 7, 11, 15, 19, 23, 27, 31)
    assert policy_cfg.attn_res_block_size == 8
    assert policy_cfg.multiview_spatial_rope_layout == "semantic_2x2"
    assert policy_cfg.multiview_spatial_rope_tile_shape == (15, 30)
    assert policy_cfg.fp32_attention is True
    assert policy_cfg.action_dim == 80 and policy_cfg.state_dim == 80
    assert policy_cfg.action_temporal_compression == 8
    assert policy_cfg.y_norm_scale_factor == 1.0
    assert policy_cfg.timestep_norm_scale_factor == 1.0
    assert policy_cfg.block_attn_types.count("GatedSoftmaxAttention") == 8


@pytest.mark.skipif(not os.path.exists(TRAIN_YAML), reason="train yaml not on this host")
def test_sampling_defaults_from_production_yaml() -> None:
    cfg = wam_config.load_train_config(TRAIN_YAML)
    defaults = wam_config.sampling_defaults_from_train_config(cfg)
    assert defaults == {"steps": 50, "flow_shift": 3.5, "cfg_scale": 1.0}
    # The yaml's inference_cfg_scale is deliberately not consulted.
    assert cfg["scheduler"]["inference_cfg_scale"] == 6.0


def test_sampling_defaults_fall_back_to_flow_shift() -> None:
    cfg = {"train": {"extra": None}, "scheduler": {"flow_shift": 3.0}}
    defaults = wam_config.sampling_defaults_from_train_config(cfg)
    assert defaults == {"steps": 50, "flow_shift": 3.0, "cfg_scale": 1.0}


def test_policy_config_rejects_other_branches() -> None:
    cfg = wam_config.load_train_config(TRAIN_YAML) if os.path.exists(TRAIN_YAML) else None
    if cfg is None:
        pytest.skip("train yaml not on this host")
    cfg["scheduler"]["pred_sigma"] = True
    with pytest.raises(ValueError, match="pred_sigma"):
        wam_config.policy_config_from_train_config(cfg)


def test_build_policy_model_dtype_and_eval() -> None:
    model = build_policy_model(tiny_policy_config(), dtype=torch.bfloat16, device="cpu")
    assert not model.training
    assert model.dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert list(model.named_buffers()) == []
    assert all(block.attn.fp32_attention for block in model.blocks)


@pytest.mark.parametrize("views", [3, 1])
def test_tiny_forward_shapes(views: int) -> None:
    torch.manual_seed(20260902)
    model = build_policy_model(tiny_policy_config(), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    inputs = tiny_inputs(batch=2, views=views, fps=25.0, seed=11 + views)
    with torch.no_grad():
        out = model(
            inputs["x"],
            inputs["timestep"],
            inputs["y"],
            mask=inputs["mask"],
            data_info=inputs["data_info"],
        )
    steps = (FRAMES - 1) * 8
    assert out["x"].shape == inputs["x"].shape
    if views == 1:
        assert out["x"].shape == (2, IN_CH, FRAMES, VIEW_H, VIEW_W)
    else:
        assert out["x"].shape == (2, IN_CH, FRAMES, 1, views * VIEW_H * VIEW_W)
    assert out["action_pred"].shape == (2, steps, ROBOT_DIM)
    assert torch.isfinite(out["x"]).all() and torch.isfinite(out["action_pred"]).all()
    assert out["action_pred"].abs().sum() > 0
    # Masked action slots are zero in the prediction.
    assert torch.equal(out["action_pred"][:, :, 66:], torch.zeros_like(out["action_pred"][:, :, 66:]))


def test_tiny_forward_is_deterministic() -> None:
    torch.manual_seed(1)
    model = build_policy_model(tiny_policy_config(), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    inputs = tiny_inputs(batch=1, views=3, fps=25.0, seed=7)
    with torch.no_grad():
        a = model(inputs["x"], inputs["timestep"], inputs["y"], mask=inputs["mask"], data_info=inputs["data_info"])
        b = model(inputs["x"], inputs["timestep"], inputs["y"], mask=inputs["mask"], data_info=inputs["data_info"])
    torch.testing.assert_close(a["x"], b["x"], rtol=0, atol=0)
    torch.testing.assert_close(a["action_pred"], b["action_pred"], rtol=0, atol=0)
