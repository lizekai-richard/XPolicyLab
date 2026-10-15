"""The sparse memory's pixel path and causal VAE encode on a tiny random LTX-2.3 encoder (CPU, the real config with small
widths): the prefix [sink | memory rows] encoded alone is the sink and memory latents of the whole sample's encode (so a
decision's prefix is the training's S and M_1..M_n and its world-model reference the training's T), the encode is
Sana's ``encode_prefix`` / the trainer's ``causal_vae_encode`` (live), and the quantization is Sana's."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

import _paths  # noqa: F401
import _live

from sana_wam_sm import sparse_memory as sm
from sana_wam_sm.vae_encode import encode_causal, frames_to_uint8, stack_canvases
from sana_wam_min.ltx2_causal_vae import AutoencoderKLCausalLTX2Video
from sana_wam_min.vae import VaeBundle

CONFIG = os.path.join(_paths.FIXTURES, "ltx23_vae_config.json")
HW = (64, 64)
MEMORIES = {"strided": {"memory_type": "strided", "memory_stride": 25}, "fixed": {"memory_type": "fixed", "memory_frames": 16}}


@pytest.fixture(scope="module")
def bundle():
    with open(CONFIG) as handle:
        cfg = {k: v for k, v in json.load(handle).items() if not k.startswith("_")}
    cfg.update(
        block_out_channels=[8, 16, 16, 16],
        decoder_block_out_channels=[8, 16, 16, 16],
        layers_per_block=[1, 1, 1, 1, 1],
        decoder_layers_per_block=[1, 1, 1, 1, 1],
        latent_channels=4,
    )
    if "upsample_type" in cfg:
        cfg["decoder_upsample_type"] = cfg.pop("upsample_type")
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()}
    torch.manual_seed(3)
    encoder = AutoencoderKLCausalLTX2Video(**cfg).eval()
    encoder.decoder = None
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.normal_(0.0, 0.05)
    encoder = encoder.float()
    g = torch.Generator().manual_seed(4)
    return VaeBundle(
        diffusers_vae=None,
        causal_encoder=encoder,
        latents_mean=torch.randn(4, generator=g),
        latents_std=torch.rand(4, generator=g) + 0.5,
        scaling_factor=0.7,
        temporal_compression=8,
        spatial_compression=32,
    )


def _canvas(row: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + row)
    return torch.randint(0, 256, (3, *HW), generator=g, dtype=torch.uint8)


@pytest.mark.parametrize("memory", MEMORIES.values(), ids=list(MEMORIES))
@pytest.mark.parametrize("obs_row", [0, 33, 230])
def test_the_prefix_alone_is_the_samples_sink_and_memory(bundle, memory, obs_row):
    """Sana ``test_the_sink_and_memory_latents_are_the_encode_of_the_prefix_alone`` on the tiny encoder (bit for bit on one
    device / dtype / kernel path): [S | M_1..M_n] of the prefix == the first n + 1 latents of [S | M_1..M_n | T]."""

    rows = sm.sequence_rows(obs_row, 0, memory)
    n = sm.memory_latents(obs_row, 0, memory)
    sequence = encode_causal(bundle, [_canvas(row) for row in rows])
    prefix = encode_causal(bundle, [_canvas(row) for row in sm.prefix_rows(obs_row, 0, memory)])
    assert sequence.shape[2] == n + 2 and prefix.shape[2] == n + 1
    assert torch.equal(prefix, sequence[:, :, : n + 1])


def test_the_target_latent_is_encoded_in_the_memory_context(bundle):
    """T depends on the frames before it (the causal encoder's cache): the target rows alone do not give it."""

    rows = sm.sequence_rows(100, 0, MEMORIES["fixed"])
    sequence = encode_causal(bundle, [_canvas(row) for row in rows])
    alone = encode_causal(bundle, [_canvas(rows[0])] + [_canvas(row) for row in rows[-8:]])
    assert not torch.equal(sequence[:, :, -1:], alone[:, :, -1:])


def test_the_encode_refuses_misuse(bundle):
    with pytest.raises(ValueError, match="1 \\+ 8k"):
        encode_causal(bundle, [_canvas(0)] * 8)
    with pytest.raises(ValueError, match="uint8"):
        stack_canvases([_canvas(0).float()])


def test_the_encode_is_sanas_encode_prefix(bundle):
    session = _live.live_session()
    canvases = torch.stack([_canvas(row) for row in sm.prefix_rows(230, 0, MEMORIES["strided"])])
    vae = SimpleNamespace(
        causal_encoder=bundle.causal_encoder,
        latents_mean=bundle.latents_mean,
        latents_std=bundle.latents_std,
        config=SimpleNamespace(scaling_factor=bundle.scaling_factor),
    )
    assert torch.equal(encode_causal(bundle, list(canvases)), session.encode_prefix(vae, canvases))


def test_the_pixel_path_is_sanas():
    live = _live.live_sparse_memory()
    g = torch.Generator().manual_seed(0)
    canvas = torch.rand(5, 3, 8, 8, generator=g, dtype=torch.float32) * 2.2 - 1.1
    canvas[0, 0, 0, :4] = torch.tensor([-1.0, 1.0, (0.5 / 127.5) - 1.0, (1.5 / 127.5) - 1.0])   # ties round half to even
    quantized = frames_to_uint8(canvas.clone())
    try:
        from dev.rwm.diffusion.data.datasets.causal_chunk_window_data import frames_to_uint8 as live_quantize
    except Exception:  # noqa: BLE001 - the data stack needs video decoders the policy env may lack
        live_quantize = None
    if live_quantize is not None:
        assert torch.equal(quantized, live_quantize(canvas.clone()))
    expected = ((canvas + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    assert torch.equal(quantized, expected)
    for dtype in (torch.float32, torch.bfloat16):
        assert torch.equal(sm.pixels_from_uint8(quantized[None], dtype), live.pixels_from_uint8(quantized[None], dtype))
