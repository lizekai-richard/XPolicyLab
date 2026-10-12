"""The hierarchical memory's pixel path and causal-VAE streams on a tiny random LTX-2.3 encoder (CPU, the real config with
small widths): a stream fed the first frame and then 8 frames per call returns the one-shot encode's latents bit for bit;
streams do not disturb each other or the one-shot encodes in between; the quantization is Sana's."""

from __future__ import annotations

import json
import os

import pytest
import torch

import _paths  # noqa: F401
import _live

from sana_wam_hm import memory as hm
from sana_wam_hm.vae_stream import LatentStream, encode_once, frames_to_uint8, pixels_from_uint8
from sana_wam_min.ltx2_causal_vae import AutoencoderKLCausalLTX2Video
from sana_wam_min.vae import VaeBundle

CONFIG = os.path.join(_paths.FIXTURES, "ltx23_vae_config.json")
HW = (64, 64)


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
        scaling_factor=1.0,
        temporal_compression=8,
        spatial_compression=32,
    )


def _canvases(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (3, *HW), generator=g, dtype=torch.uint8) for _ in range(n)]


@pytest.mark.parametrize("chunks", [0, 1, 3])
def test_the_dense_stream_is_the_one_shot_encode(bundle, chunks):
    """L_0 = the first frame alone, then one latent per 8 frames == ``encode`` of the 1 + 8n frames (Sana plan 1)."""

    frames = _canvases(1 + 8 * chunks, seed=chunks)
    one_shot = encode_once(bundle, frames)
    stream = LatentStream(bundle)
    streamed = [stream.start(frames[0])]
    for j in range(chunks):
        streamed.append(stream.extend(frames[1 + 8 * j : 9 + 8 * j]))
    assert torch.equal(torch.cat(streamed, dim=2), one_shot)


def test_streams_and_one_shot_encodes_interleave_without_crosstalk(bundle):
    """The dense stream lives across decisions while the observation / sparse encodes run in between."""

    frames = _canvases(25, seed=9)
    reference = encode_once(bundle, frames)
    dense = LatentStream(bundle)
    out = [dense.start(frames[0])]
    encode_once(bundle, _canvases(1, seed=1))                       # an observation encode
    sparse = LatentStream(bundle)
    sparse_out = [sparse.start(frames[3])] + [sparse.extend(frames[4:12])]
    out.append(dense.extend(frames[1:9]))
    encode_once(bundle, _canvases(17, seed=2))                      # another one-shot encode
    sparse_out.append(sparse.extend(frames[12:20]))
    out.append(dense.extend(frames[9:17]))
    out.append(dense.extend(frames[17:25]))
    assert torch.equal(torch.cat(out, dim=2), reference)
    assert torch.equal(torch.cat(sparse_out, dim=2), encode_once(bundle, frames[3:20]))


def test_a_stream_refuses_misuse(bundle):
    stream = LatentStream(bundle)
    with pytest.raises(RuntimeError, match="start the stream"):
        stream.extend(_canvases(8))
    stream.start(_canvases(1)[0])
    with pytest.raises(ValueError, match="exactly 8"):
        stream.extend(_canvases(7))
    with pytest.raises(ValueError, match="1 \\+ 8n"):
        encode_once(bundle, _canvases(8))


def test_the_pixel_path_is_sanas():
    live = _live.live_hm()
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
        ours = pixels_from_uint8(quantized[None], dtype)
        theirs = live.pixels_from_uint8(quantized[None], dtype)
        assert ours.dtype == theirs.dtype and torch.equal(ours, theirs)
    assert hm.FRAMES_PER_LATENT == live.FRAMES_PER_LATENT
