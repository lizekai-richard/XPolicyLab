"""``model.extra.sana_pixel_pad: masked`` (rwm/zekai-merge 319d3f666) in the mirror alone: the black top-right quadrant of
the 2x2 sana_pixel canvas leaves the token sequence, the video velocity is 0 there, and nothing the pad holds reaches
the outputs. Bit parity with the live forward: test_sana_pixel_pad_mask_parity.py (run on its own)."""

from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

from _tiny_policy import CAP_CH, IN_CH, MML, NOISY_T, ROBOT_DIM, tiny_policy_config  # noqa: E402

from sana_wam_min.policy_model.geometry import sana_pixel_real_token_index, sana_pixel_real_token_mask  # noqa: E402
from sana_wam_min.policy_model.model import PolicyModel  # noqa: E402

GRID = (4, 6)          # a tiny even canvas grid: 2 x 3 cells per quadrant (the 320x512 canvas is 10 x 16)
FRAMES = 3


def test_the_kept_cells_are_every_quadrant_but_the_top_right_one():
    mask = sana_pixel_real_token_mask(10, 16)
    assert int(mask.sum()) == 120 and not mask[:5, 8:].any() and mask[:5, :8].all() and mask[5:].all()
    index = sana_pixel_real_token_index(2, 10, 16)
    assert index.numel() == 240 and index[:8].tolist() == list(range(8)) and int(index[8]) == 16 and int(index[120]) == 160
    with pytest.raises(ValueError, match="320x512"):
        sana_pixel_real_token_mask(10, 15)          # the 320x480 canvas: the quadrant edge falls inside a cell


def test_the_mask_is_refused_outside_the_one_policy_class():
    config = dataclasses.replace(tiny_policy_config(), sana_pixel_pad="masked")
    assert config.validate().sana_pixel_pad == "masked"
    with pytest.raises(ValueError, match="bidirectional policy only"):
        dataclasses.replace(config, shared_prompt=True).validate()
    with pytest.raises(ValueError, match="sana_pixel_pad"):
        dataclasses.replace(config, sana_pixel_pad="hidden").validate()


def _canvas_inputs(batch: int = 2, seed: int = 5) -> dict:
    torch.manual_seed(seed)
    height, width = GRID
    steps = (FRAMES - 1) * 8
    timestep = torch.zeros(batch, 1, FRAMES)
    timestep[:, :, 1:] = NOISY_T
    action_mask = torch.ones(batch, steps, ROBOT_DIM, dtype=torch.bool)
    action_mask[:, :, 66:] = False
    return {
        "x": torch.randn(batch, IN_CH, FRAMES, height, width),
        "timestep": timestep,
        "y": torch.randn(batch, 1, 1, MML, CAP_CH),
        "mask": torch.ones(batch, 1, MML, dtype=torch.int16),
        "data_info": {
            "rwm_task": "policy",
            "view_count": torch.full((batch,), 1, dtype=torch.long),
            "view_latent_shape": torch.tensor([[[height, width]]] * batch),
            "view_slot_ids": torch.tensor((0,)),
            "model_fps": torch.full((batch,), 25.0),
            "initial_state80": torch.randn(batch, ROBOT_DIM),
            "initial_state_condition_mask80": torch.ones(batch, ROBOT_DIM, dtype=torch.bool),
            "action80": torch.randn(batch, steps, ROBOT_DIM),
            "action_mask80": action_mask,
            "action_timestep": torch.full((batch, steps), NOISY_T),
            "camera_conditioning_enabled": False,
        },
    }


def _run(model, inputs: dict) -> dict:
    with torch.no_grad():
        return model(
            inputs["x"].clone(), inputs["timestep"].clone(), inputs["y"].clone(), mask=inputs["mask"].clone(),
            data_info=copy.deepcopy(inputs["data_info"]),
        )


def test_the_masked_forward_ignores_the_pad_and_moves_it_by_zero():
    torch.manual_seed(0)
    unmasked = PolicyModel(tiny_policy_config()).eval()
    with torch.no_grad():
        for parameter in unmasked.parameters():         # the heads start at zero: randomize every weight
            parameter.uniform_(-0.05, 0.05)
    masked = PolicyModel(dataclasses.replace(tiny_policy_config(), sana_pixel_pad="masked")).eval()
    masked.load_state_dict(unmasked.state_dict())
    inputs = _canvas_inputs()
    poked = copy.deepcopy(inputs)
    pad = ~sana_pixel_real_token_mask(*GRID)
    poked["x"][..., pad] = 10 * torch.randn_like(poked["x"][..., pad])

    out, again = _run(masked, inputs), _run(masked, poked)
    assert torch.count_nonzero(out["action_pred"]) > 0 and torch.count_nonzero(out["x"][..., ~pad]) > 0
    assert torch.count_nonzero(out["x"][..., pad]) == 0
    assert torch.equal(out["action_pred"], again["action_pred"]) and torch.equal(out["x"], again["x"])
    plain, plain_poked = _run(unmasked, inputs), _run(unmasked, poked)
    assert not torch.equal(plain["action_pred"], out["action_pred"])
    assert not torch.equal(plain["action_pred"], plain_poked["action_pred"])     # unmasked, the pad reaches the actions

    strip = copy.deepcopy(inputs)
    strip["data_info"]["view_count"] = torch.full((2,), 3, dtype=torch.long)
    strip["data_info"]["view_latent_shape"] = torch.tensor([[[2, 2]] * 3] * 2)
    strip["x"] = torch.randn(2, IN_CH, FRAMES, 1, 12)
    strip["y"] = torch.randn(2, 4, 1, MML, CAP_CH)
    strip["mask"] = torch.ones(2, 4, MML, dtype=torch.int16)
    with pytest.raises(ValueError, match="one-view"):
        _run(masked, strip)
