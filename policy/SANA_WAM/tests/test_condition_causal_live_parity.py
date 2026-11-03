"""Live parity of the condition-causal memory-less policy (rwm/zekai-merge f6a257d4a + 8e0639b16) against the tree
that defines it (``SANA_CONDITION_REPO``, default the worktree ~/zekail/Sana_zekai_merge_6de2454dc).

At that tree the one policy class is condition-causal with no switch: self-attention runs ``[obs; state] |
[prediction latents; actions]``, the GatedDeltaNet layers carry the chunk recurrence (``gate_proj`` / ``A_log`` /
``dt_bias`` / ``recall_gate``), ``model.memory_gdn_output_normalizer`` false reads the numerator alone and
``model.extra.state_cross_attention`` moves the anchor state into the robot group's caption. The live class and the
mirror (``condition_causal=True``) get identical random weights and one batch; the forwards must agree bit for bit,
the video prediction and the action prediction, on the sana latent strip (V = 3, G = 4) and the one-view sana pixel
canvas (G = 1, unmasked and masked), under both RoPE contracts. The condition never reads the prediction: the frame-0
velocity does not move when the later latents, the actions or their noise level change.

Run on its own like the other live parity files (the first file of a session owns ``dev``).
"""

from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from _tiny_policy import CAP_CH, DEPTH, HEADS, HIDDEN, IN_CH, LHD, MML, NOISY_T, ROBOT_DIM, SHD  # noqa: E402

from sana_wam_min.policy_model.checkpoint import load_policy_state_dict  # noqa: E402
from sana_wam_min.policy_model.condition_chunks import GDN_RECURRENCE_KEYS  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402
from sana_wam_min.policy_model.model import PolicyModel  # noqa: E402

SANA_CONDITION_REPO = os.environ.get(
    "SANA_CONDITION_REPO", os.path.expanduser("~/zekail/Sana_zekai_merge_6de2454dc")
)
FRAMES, STRIDE, STEPS_PER_FRAME = 3, 4, 8          # video_fps 8 of a 25-row tier: every 4th row, 8 action rows each
STRIP_VIEW_HW, CANVAS_GRID = (2, 3), (4, 6)


class _Cfg(SimpleNamespace):
    """Unset fields read as None (dev/rwm/tests/test_policy_model_ownership.py)."""

    def __getattr__(self, name):
        return None


def _live_class():
    if not os.path.isdir(SANA_CONDITION_REPO):
        pytest.skip(f"no Sana checkout at {SANA_CONDITION_REPO} (set SANA_CONDITION_REPO)")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    if SANA_CONDITION_REPO not in sys.path:
        sys.path.insert(0, SANA_CONDITION_REPO)
    module = pytest.importorskip(
        "dev.rwm.diffusion.model.nets.sana_qwennext_action_policy",
        reason="Sana checkout not importable (set SANA_CONDITION_REPO)",
    )
    if not os.path.abspath(module.__file__).startswith(os.path.abspath(SANA_CONDITION_REPO)):
        pytest.skip(
            f"the dev package is already imported from another Sana checkout ({module.__file__}); "
            "run this file on its own for the live parity"
        )
    if not hasattr(module, "resolve_state_cross_attention"):
        pytest.skip("this Sana checkout predates the condition-causal policy's state_cross_attention (8e0639b16)")
    return module.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy


def _twins(
    multiview: str, *, rope: str = "aligned", state_cross: bool = False, normalizer: bool = True, pad: str = "unmasked"
):
    """(live, mirror) with identical random weights."""

    Live = _live_class()
    torch.manual_seed(20261103)
    kwargs = dict(
        depth=DEPTH, hidden_size=HIDDEN, patch_size=(1, 1, 1), num_heads=HEADS, in_channels=IN_CH,
        caption_channels=CAP_CH, model_max_length=MML, linear_head_dim=LHD, softmax_head_dim=SHD, softmax_ratio=0.5,
        attn_res_block_size=2, use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True,
    )
    extra = {"action_dim": ROBOT_DIM, "state_dim": ROBOT_DIM, "rope": rope, "state_cross_attention": state_cross}
    if multiview == "sana_pixel":
        extra["sana_pixel_pad"] = pad
    model = _Cfg(extra=extra)
    if not normalizer:
        model.memory_gdn_output_normalizer = False
    config = _Cfg(model=model, data=_Cfg(extra={"multiview": multiview}), vae=_Cfg(vae_stride=[8, 32, 32]))
    live = Live(config=config, **kwargs)
    assert live.state_cross_attention is state_cross and live.memory_gdn_output_normalizer is normalizer
    if getattr(live, "use_xformers_cross_attention", False):
        pytest.skip("the live model kept xformers cross-attention (imported before DISABLE_XFORMERS=1 took effect)")
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.uniform_(-0.05, 0.05)
    live.eval()
    policy_config = PolicyConfig.from_sana_kwargs(config=config, condition_causal=True, **kwargs)
    if multiview == "sana_latent":  # the robot_sft strip tile (8, 16) the server resolves for these checkpoints
        import dataclasses

        policy_config = dataclasses.replace(policy_config, multiview_spatial_rope_tile_shape=(8, 16)).validate()
    assert (policy_config.condition_causal, policy_config.state_cross_attention) == (True, state_cross)
    assert policy_config.gdn_output_normalizer is normalizer
    mirror = PolicyModel(policy_config)
    state = live.state_dict()
    assert any(key.endswith("attn." + GDN_RECURRENCE_KEYS[0]) for key in state)
    load_policy_state_dict(mirror, state)
    return live, mirror.eval()


def _inputs(multiview: str, *, batch: int = 2, seed: int = 11) -> dict:
    torch.manual_seed(seed)
    steps = (FRAMES - 1) * STEPS_PER_FRAME * STRIDE
    if multiview == "sana_latent":
        views, groups = 3, 4
        height, width = STRIP_VIEW_HW
        x = torch.randn(batch, IN_CH, FRAMES, 1, views * height * width)
        shape, slots = [[[height, width]] * views] * batch, (0, 2, 3)
    else:
        views, groups = 1, 1
        height, width = CANVAS_GRID
        x = torch.randn(batch, IN_CH, FRAMES, height, width)
        shape, slots = [[[height, width]]] * batch, (0,)
    timestep = torch.zeros(batch, 1, FRAMES)
    timestep[:, :, 1:] = NOISY_T
    mask = torch.ones(batch, groups, MML, dtype=torch.int16)
    mask[..., 5:] = 0
    action_mask = torch.ones(batch, steps, ROBOT_DIM, dtype=torch.bool)
    action_mask[:, :, 66:] = False
    state_mask = torch.ones(batch, ROBOT_DIM, dtype=torch.bool)
    state_mask[:, 58:66] = False
    data_info = {
        "rwm_task": "policy",
        "view_count": torch.full((batch,), views, dtype=torch.long),
        "view_latent_shape": torch.tensor(shape),
        "view_slot_ids": torch.tensor(slots),
        "model_fps": torch.full((batch,), 25.0),
        "video_frame_stride": torch.full((batch,), STRIDE, dtype=torch.long),
        "initial_state80": torch.randn(batch, ROBOT_DIM),
        "initial_state_condition_mask80": state_mask,
        "action80": torch.randn(batch, steps, ROBOT_DIM),
        "action_mask80": action_mask,
        "action_timestep": torch.full((batch, steps), NOISY_T),
        "camera_conditioning_enabled": False,
    }
    return {"x": x, "timestep": timestep, "y": torch.randn(batch, groups, 1, MML, CAP_CH), "mask": mask,
            "data_info": data_info}


def _run(model, inputs: dict) -> dict:
    with torch.no_grad():
        return model(
            inputs["x"].clone(), inputs["timestep"].clone(), inputs["y"].clone(), mask=inputs["mask"].clone(),
            data_info=copy.deepcopy(inputs["data_info"]),
        )


def _assert_bitwise(live, mirror, inputs):
    expected, ours = _run(live, inputs), _run(mirror, inputs)
    torch.testing.assert_close(ours["action_pred"], expected["action_pred"], rtol=0, atol=0)
    torch.testing.assert_close(ours["x"], expected["x"], rtol=0, atol=0)
    return ours


@pytest.mark.parametrize("state_cross", [False, True])
@pytest.mark.parametrize("rope", ["aligned", "independent"])
def test_the_sana_latent_strip_forward_is_bitwise(rope, state_cross):
    live, mirror = _twins("sana_latent", rope=rope, state_cross=state_cross)
    _assert_bitwise(live, mirror, _inputs("sana_latent"))


@pytest.mark.parametrize("state_cross", [False, True])
@pytest.mark.parametrize("pad", ["unmasked", "masked"])
def test_the_sana_pixel_canvas_forward_is_bitwise(pad, state_cross):
    live, mirror = _twins("sana_pixel", pad=pad, state_cross=state_cross)
    _assert_bitwise(live, mirror, _inputs("sana_pixel"))


@pytest.mark.parametrize("multiview", ["sana_latent", "sana_pixel"])
def test_the_numerator_only_gdn_readout_is_bitwise(multiview):
    live, mirror = _twins(multiview, normalizer=False)
    _assert_bitwise(live, mirror, _inputs(multiview))


@pytest.mark.parametrize("state_cross", [False, True])
def test_the_condition_never_reads_the_prediction(state_cross):
    """Frame 0 of every view (and the state) is the condition chunk: its velocity is fixed by the condition alone."""

    live, mirror = _twins("sana_latent", state_cross=state_cross)
    inputs = _inputs("sana_latent")
    base = _assert_bitwise(live, mirror, inputs)
    moved = copy.deepcopy(inputs)
    moved["x"][:, :, 1:] = torch.randn_like(moved["x"][:, :, 1:])
    moved["data_info"]["action80"] = torch.randn_like(moved["data_info"]["action80"])
    moved["timestep"][:, :, 1:] = NOISY_T / 2
    moved["data_info"]["action_timestep"] = torch.full_like(moved["data_info"]["action_timestep"], NOISY_T / 2)
    other = _assert_bitwise(live, mirror, moved)
    torch.testing.assert_close(other["x"][:, :, 0], base["x"][:, :, 0], rtol=0, atol=0)
    assert not torch.equal(other["x"][:, :, 1:], base["x"][:, :, 1:])
    assert not torch.equal(other["action_pred"], base["action_pred"])


def _run_cached(model, inputs: dict, cache, prefill: str, target: str) -> dict:
    with torch.no_grad():
        model(
            inputs["x"].clone(), inputs["timestep"].clone(), inputs["y"].clone(), mask=inputs["mask"].clone(),
            data_info=copy.deepcopy(inputs["data_info"]), condition_prefix=(cache, prefill),
        )
        return model(
            inputs["x"].clone(), inputs["timestep"].clone(), inputs["y"].clone(), mask=inputs["mask"].clone(),
            data_info=copy.deepcopy(inputs["data_info"]), condition_prefix=(cache, target),
        )


@pytest.mark.parametrize("state_cross", [False, True])
@pytest.mark.parametrize("multiview", ["sana_latent", "sana_pixel"])
def test_the_prefix_cache_passes_are_bitwise(multiview, state_cross):
    """PREFILL then TARGET (Sana 89eaa78f6): the prediction pass of the mirror equals the live one bit for bit."""

    live_chunks = pytest.importorskip("dev.rwm.diffusion.model.condition_chunks")
    if not hasattr(live_chunks, "ConditionPrefix"):
        pytest.skip("this Sana checkout predates the condition prefix cache (89eaa78f6)")
    from sana_wam_min.policy_model import condition_chunks as mirror_chunks

    live, mirror = _twins(multiview, state_cross=state_cross)
    inputs = _inputs(multiview)
    expected = _run_cached(live, inputs, live_chunks.ConditionPrefix(), live_chunks.PREFILL, live_chunks.TARGET)
    ours = _run_cached(mirror, inputs, mirror_chunks.ConditionPrefix(), mirror_chunks.PREFILL, mirror_chunks.TARGET)
    torch.testing.assert_close(ours["action_pred"], expected["action_pred"], rtol=0, atol=0)
    torch.testing.assert_close(ours["x"][:, :, 1:], expected["x"][:, :, 1:], rtol=0, atol=0)


@pytest.mark.parametrize("normalizer", [True, False])
@pytest.mark.parametrize("state_cross", [False, True])
@pytest.mark.parametrize("multiview", ["sana_latent", "sana_pixel"])
def test_the_prefix_cache_equals_the_full_forward_in_fp64(multiview, state_cross, normalizer):
    """The cached passes are the full forward re-ordered: equal up to rounding (fp64 here), one prefill serving every
    noise level of the prediction."""

    from sana_wam_min.policy_model import condition_chunks as mirror_chunks

    _live, mirror = _twins(multiview, state_cross=state_cross, normalizer=normalizer)
    mirror = mirror.double()
    inputs = _inputs(multiview)
    inputs["x"], inputs["y"] = inputs["x"].double(), inputs["y"].double()
    for key in ("initial_state80", "action80"):
        inputs["data_info"][key] = inputs["data_info"][key].double()
    cache = mirror_chunks.ConditionPrefix()
    with torch.no_grad():
        mirror(inputs["x"], inputs["timestep"], inputs["y"], mask=inputs["mask"],
               data_info=copy.deepcopy(inputs["data_info"]), condition_prefix=(cache, mirror_chunks.PREFILL))
    assert len(cache) == DEPTH
    for level in (NOISY_T, NOISY_T / 3):
        moved = copy.deepcopy(inputs)
        moved["timestep"][:, :, 1:] = level
        moved["data_info"]["action_timestep"] = torch.full_like(moved["data_info"]["action_timestep"], level)
        full = _run(mirror, moved)
        with torch.no_grad():
            cached = mirror(moved["x"], moved["timestep"], moved["y"], mask=moved["mask"],
                            data_info=copy.deepcopy(moved["data_info"]), condition_prefix=(cache, mirror_chunks.TARGET))
        torch.testing.assert_close(cached["action_pred"], full["action_pred"], rtol=0, atol=1e-12)
        torch.testing.assert_close(cached["x"][:, :, 1:], full["x"][:, :, 1:], rtol=0, atol=1e-12)
