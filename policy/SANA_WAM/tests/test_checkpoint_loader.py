"""CPU tests for sana_wam_min.checkpoint.load_policy_weights on a tiny model."""

from __future__ import annotations

import os
import sys
from collections import OrderedDict

import pytest
import torch

ADAPTER_DIR = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/XPolicyLab/policy/SANA_WAM"
if ADAPTER_DIR not in sys.path:
    sys.path.insert(0, ADAPTER_DIR)

from _tiny_policy import CAP_CH, HIDDEN, INPUT_SIZE, MML, tiny_policy_config  # noqa: E402
from sana_wam_min.checkpoint import (  # noqa: E402
    CHECKPOINT_RELATIVE_PATH,
    build_policy_model,
    load_policy_weights,
    resolve_checkpoint_file,
)

UNMODELED_SHAPES = {
    "pos_embed": (1, INPUT_SIZE * INPUT_SIZE, HIDDEN),
    "y_embedder.y_embedding": (MML, CAP_CH),
    "plucker_embed.weight": (HIDDEN, 6, 1, 1, 1),
}


def _randomized_source() -> torch.nn.Module:
    torch.manual_seed(20260902)
    model = build_policy_model(tiny_policy_config(), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    return model


def _fsdp_like_state(model: torch.nn.Module, drift: dict | None = None) -> OrderedDict:
    torch.manual_seed(3)
    state = OrderedDict()
    state["pos_embed"] = torch.randn(UNMODELED_SHAPES["pos_embed"])
    state.update(model.state_dict())
    state["y_embedder.y_embedding"] = torch.randn(UNMODELED_SHAPES["y_embedder.y_embedding"]).to(torch.bfloat16)
    state["plucker_embed.weight"] = torch.zeros(UNMODELED_SHAPES["plucker_embed.weight"])
    for key, shape in (drift or {}).items():
        state[key] = torch.zeros(shape)
    return state


def _write_checkpoint_dir(tmp_path, state: OrderedDict) -> str:
    ckpt_dir = tmp_path / "epoch_0_step_0"
    os.makedirs(ckpt_dir / "model")
    torch.save(state, ckpt_dir / CHECKPOINT_RELATIVE_PATH)
    return str(ckpt_dir)


def test_resolve_checkpoint_file(tmp_path) -> None:
    ckpt_dir = _write_checkpoint_dir(tmp_path, _fsdp_like_state(_randomized_source()))
    expected = os.path.join(ckpt_dir, "model", "pytorch_model_fsdp.bin")
    assert resolve_checkpoint_file(ckpt_dir) == expected
    assert resolve_checkpoint_file(expected) == expected


def test_strict_load_from_checkpoint_dir(tmp_path) -> None:
    source = _randomized_source()
    state = _fsdp_like_state(source)
    ckpt_dir = _write_checkpoint_dir(tmp_path, state)

    target = build_policy_model(tiny_policy_config(), dtype=torch.float32, device="cpu")
    report = load_policy_weights(target, ckpt_dir, device="cpu")

    assert report["tensors_total"] == len(state) == len(source.state_dict()) + 3
    assert report["tensors_loaded"] == len(source.state_dict()) == len(target.state_dict())
    assert report["stripped"] == ["plucker_embed.weight", "pos_embed", "y_embedder.y_embedding"]
    assert report["source"] == os.path.join(ckpt_dir, CHECKPOINT_RELATIVE_PATH)
    assert report["softmax_layer_indices"] == tiny_policy_config().softmax_layer_indices
    assert not target.training
    for key, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], tensor), key


def test_bf16_target_receives_rounded_fp32_weights(tmp_path) -> None:
    source = _randomized_source()
    ckpt_dir = _write_checkpoint_dir(tmp_path, _fsdp_like_state(source))
    target = build_policy_model(tiny_policy_config(), dtype=torch.bfloat16, device="cpu")
    report = load_policy_weights(target, ckpt_dir)
    assert report["dtype"] == "torch.bfloat16"
    assert target.dtype == torch.bfloat16
    for key, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], tensor.to(torch.bfloat16)), key


def test_state_dict_wrapper_and_module_prefix_are_unwrapped(tmp_path) -> None:
    source = _randomized_source()
    inner = OrderedDict((f"module.{k}", v) for k, v in _fsdp_like_state(source).items())
    path = tmp_path / "wrapped.bin"
    torch.save({"state_dict": inner}, path)
    target = build_policy_model(tiny_policy_config(), dtype=torch.float32, device="cpu")
    report = load_policy_weights(target, str(path))
    assert report["tensors_loaded"] == len(source.state_dict())
    assert report["source"] == str(path)
    for key, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], tensor), key


def test_shape_drifted_pos_embed_raises(tmp_path) -> None:
    source = _randomized_source()
    state = _fsdp_like_state(source, drift={"pos_embed": (1, 225, HIDDEN)})
    ckpt_dir = _write_checkpoint_dir(tmp_path, state)
    target = build_policy_model(tiny_policy_config(), dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="pos_embed"):
        load_policy_weights(target, ckpt_dir)


def test_missing_tensor_fails_strict_load(tmp_path) -> None:
    source = _randomized_source()
    state = _fsdp_like_state(source)
    del state["final_layer.linear.bias"]
    ckpt_dir = _write_checkpoint_dir(tmp_path, state)
    target = build_policy_model(tiny_policy_config(), dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError):
        load_policy_weights(target, ckpt_dir)


def test_softmax_block_placement_mismatch_raises(tmp_path) -> None:
    source = _randomized_source()
    state = _fsdp_like_state(source)
    # Moving beta_proj off a GDN block makes the checkpoint look like a different softmax placement.
    state["blocks.1.attn.beta_proj.weight"] = state.pop("blocks.0.attn.beta_proj.weight")
    ckpt_dir = _write_checkpoint_dir(tmp_path, state)
    target = build_policy_model(tiny_policy_config(), dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="softmax blocks"):
        load_policy_weights(target, ckpt_dir)
