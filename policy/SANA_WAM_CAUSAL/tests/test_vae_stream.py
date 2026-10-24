"""The episode's per-view causal-VAE streams on a tiny random LTX-2.3 encoder (CPU, the real config with small widths):
streams opened at the first frame (the sink, then the frozen first chunk) and fed 8 frames per chunk return the
phase window's one-shot encode bit for bit, per view and packed into the strip; they do not disturb each other or the
one-shot observation encodes in between; and they equal Sana's ``crwm.encode_window_chunks`` (the trainer's and the
phase store's encode) of the window whose head is one whole chunk, observations included (live part: ``SANA_CAUSAL_REPO``).
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

import _paths

from sana_wam_causal.vae_stream import ENCODER_BLOCK, LatentStream, ViewStreams, encode_window
from sana_wam_min.ltx2_causal_vae import AutoencoderKLCausalLTX2Video
from sana_wam_min.multiview import pack_multiview_latents
from sana_wam_min.vae import VaeBundle

CONFIG = os.path.join(_paths.FIXTURES, "ltx23_vae_config.json")
SANA_CAUSAL_REPO = os.path.expanduser(os.environ.get("SANA_CAUSAL_REPO", "~/zekail/Sana_episode_0081aa702"))
HW = (64, 96)
FRAMES_PER_CHUNK = 8   # C / s = 32 / 4


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


def _frames(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(3, *HW, generator=g) * 2 - 1 for _ in range(n)]


def _episode(views, chunks, seed):
    """Per view: its first frame and ``chunks`` executed chunks of 8 frames each."""

    return [(_frames(1, seed=seed + 10 * v)[0], _frames(FRAMES_PER_CHUNK * chunks, seed=seed + 10 * v + 1)) for v in range(views)]


def _window(first, later):
    """The phase window whose head is one whole chunk, at the video frame stride: the first frame, its 8 frozen copies
    (chunk 0), then the executed frames."""

    return [first] + [first] * FRAMES_PER_CHUNK + list(later)


@pytest.mark.parametrize("views", [1, 3])
@pytest.mark.parametrize("chunks", [0, 1, 3])
def test_the_streams_are_the_phase_windows_one_shot_encode(bundle, views, chunks):
    episode = _episode(views, chunks, seed=chunks)
    one_shot = [encode_window(bundle, _window(first, later)) for first, later in episode]   # [1, C, 2 + chunks, h, w]
    streams = ViewStreams(bundle, frames_per_chunk=FRAMES_PER_CHUNK)
    sink, frozen = streams.start([first for first, _ in episode])
    got = [sink, frozen] + [
        streams.encode_chunk([later[FRAMES_PER_CHUNK * j : FRAMES_PER_CHUNK * (j + 1)] for _, later in episode])
        for j in range(chunks)
    ]
    expected = [pack_multiview_latents([latent[:, :, i : i + 1] for latent in one_shot])[0] for i in range(2 + chunks)]
    for i, (a, b) in enumerate(zip(got, expected)):
        assert torch.equal(a, b), f"latent {i}"
    assert streams.chunks == 1 + chunks
    shape = (HW[0] // 32, HW[1] // 32)
    assert streams.view_shapes == ((shape,) * views if views > 1 else (shape,))
    if views > 1:
        assert sink.shape == (1, 4, 1, 1, views * shape[0] * shape[1])
    # chunk 1's observation, its first frame alone, is the sink
    assert torch.equal(pack_multiview_latents([encode_window(bundle, [first]) for first, _ in episode])[0], sink)


@pytest.mark.parametrize("views", [1, 3])
def test_a_first_frame_stream_opens_on_the_sink_alone(bundle, views):
    """A first-frame window's streams (Sana 9fca24879) open on the sink alone and stream chunk 0 on, bit for bit the
    one-shot encode of the window from the first frame. The causal encoder pads with the first frame, so the chunk
    latents equal the padded stream's (the frozen chunk encoded after the sink; the phase store serves the first-frame
    window as the wholly frozen phase's member less its chunk 0) up to float32 round-off: NSC measured the two encodes
    bit-identical, this CPU stream differs by <= 4e-7 on O(1) latents."""

    chunks = 3
    episode = _episode(views, chunks, seed=21)
    firsts = [first for first, _ in episode]

    def chunk_frames(j):
        return [later[FRAMES_PER_CHUNK * j : FRAMES_PER_CHUNK * (j + 1)] for _, later in episode]

    streams = ViewStreams(bundle, frames_per_chunk=FRAMES_PER_CHUNK, frozen_first_chunk=False)
    sink, frozen = streams.start(firsts)
    assert frozen is None and streams.chunks == 0
    got = [streams.encode_chunk(chunk_frames(j)) for j in range(chunks)]
    assert streams.chunks == chunks
    padded = ViewStreams(bundle, frames_per_chunk=FRAMES_PER_CHUNK)
    padded_sink, _ = padded.start(firsts)
    reference = [padded.encode_chunk(chunk_frames(j)) for j in range(chunks)]
    assert torch.equal(sink, padded_sink)
    for j, (a, b) in enumerate(zip(got, reference)):
        torch.testing.assert_close(a, b, rtol=0.0, atol=2e-6, msg=lambda m: f"chunk {j} vs the padded stream: {m}")
    one_shot = [encode_window(bundle, [first] + list(later)) for first, later in episode]
    expected = [pack_multiview_latents([latent[:, :, i : i + 1] for latent in one_shot])[0] for i in range(1 + chunks)]
    assert torch.equal(sink, expected[0])
    for j in range(chunks):
        assert torch.equal(got[j], expected[1 + j]), f"chunk {j} vs the one-shot encode"


def test_streams_and_one_shot_encodes_interleave_without_crosstalk(bundle):
    """The episode's streams live across decisions while the single-frame observation encodes run in between."""

    episode = _episode(3, 3, seed=9)
    reference = [encode_window(bundle, _window(first, later)) for first, later in episode]
    streams = ViewStreams(bundle, frames_per_chunk=FRAMES_PER_CHUNK)
    sink, frozen = streams.start([first for first, _ in episode])
    out = [sink, frozen]
    for j in range(3):
        encode_window(bundle, _frames(1, seed=100 + j))                # an observation encode
        other = LatentStream(bundle)                                    # another stream, e.g. a reopened memory
        other.start(_frames(1, seed=200 + j)[0])
        out.append(streams.encode_chunk([later[8 * j : 8 * (j + 1)] for _, later in episode]))
    expected = [pack_multiview_latents([latent[:, :, i : i + 1] for latent in reference])[0] for i in range(5)]
    assert all(torch.equal(a, b) for a, b in zip(out, expected))


def test_the_streams_refuse_misuse(bundle):
    with pytest.raises(ValueError, match="whole blocks of 8"):
        ViewStreams(bundle, frames_per_chunk=12)
    streams = ViewStreams(bundle, frames_per_chunk=FRAMES_PER_CHUNK)
    with pytest.raises(RuntimeError, match="open the streams"):
        streams.encode_chunk([_frames(8)])
    streams.start(_frames(2))
    with pytest.raises(ValueError, match="2 views stream"):
        streams.encode_chunk([_frames(8)])
    with pytest.raises(ValueError, match="carries 8 frames per view"):
        streams.encode_chunk([_frames(7), _frames(7)])
    stream = LatentStream(bundle)
    with pytest.raises(RuntimeError, match="start the stream"):
        stream.extend(_frames(8))
    stream.start(_frames(1)[0])
    with pytest.raises(ValueError, match="whole blocks of 8"):
        stream.extend(_frames(7))
    with pytest.raises(ValueError, match="1 \\+ 8n"):
        encode_window(bundle, _frames(8))
    assert ENCODER_BLOCK == 8


def test_the_streams_are_sanas_window_encode(bundle):
    """Sana's ``crwm.encode_window_chunks`` with the same encoder over the window whose head is one whole chunk (C = 32,
    stride 4): chunk 0's observation is the sink, its targets the frozen chunk, chunk c >= 1's observation its first
    frame alone (chunk 1: the first frame again) and its targets the streamed latents."""

    if not os.path.isdir(os.path.join(SANA_CAUSAL_REPO, "dev", "rwm")):
        pytest.skip(f"no Sana checkout at {SANA_CAUSAL_REPO} (set SANA_CAUSAL_REPO)")
    if SANA_CAUSAL_REPO not in sys.path:
        sys.path.insert(0, SANA_CAUSAL_REPO)
    crwm = pytest.importorskip("dev.rwm.diffusion.model.utils.causal_rwm_train")
    if not hasattr(crwm, "encode_window_chunks"):
        pytest.skip(f"{SANA_CAUSAL_REPO} predates the window encode (d37817788)")
    chunks = 3
    (first, later), = _episode(1, chunks, seed=21)
    pixels = torch.stack(_window(first, later), dim=1)[None]          # [1, 3, 1 + 8 * (chunks + 1), H, W]

    def encode(clips):
        return torch.cat([encode_window(bundle, list(clip.unbind(1))) for clip in clips], dim=0)

    live = crwm.encode_window_chunks(encode, pixels, chunks + 1, 32, vae_stride=8, frame_stride=4)   # [1, C, 2(M), h, w]
    streams = ViewStreams(bundle, frames_per_chunk=FRAMES_PER_CHUNK)
    sink, frozen = streams.start([first])
    targets = [frozen] + [streams.encode_chunk([later[8 * j : 8 * (j + 1)]]) for j in range(chunks)]
    observations = [sink] + [encode_window(bundle, [first])] + [
        encode_window(bundle, [later[8 * j - 1]]) for j in range(1, chunks)
    ]
    for c in range(chunks + 1):
        assert torch.equal(live[:, :, 2 * c : 2 * c + 1], observations[c]), f"observation {c}"
        assert torch.equal(live[:, :, 2 * c + 1 : 2 * c + 2], targets[c]), f"targets {c}"


@pytest.mark.parametrize("views", [1, 3])
def test_an_observation_is_its_frame_encoded_alone_and_leaves_the_streams_where_they_were(bundle, views):
    """causal_observation (Sana encode_observation): a chunk's observation latent is every view's start frame encoded
    alone, bit for bit the sink's encode of that frame and the one-shot encode of the one frame, packed like the sink;
    a first-frame episode whose streams encode one observation per chunk streams every chunk exactly as without them."""

    chunks = 3
    episode = _episode(views, chunks, seed=33)
    firsts = [first for first, _ in episode]

    def chunk_frames(j):
        return [later[FRAMES_PER_CHUNK * j : FRAMES_PER_CHUNK * (j + 1)] for _, later in episode]

    reference = ViewStreams(bundle, frames_per_chunk=FRAMES_PER_CHUNK, frozen_first_chunk=False)
    reference.start(firsts)
    expected = [reference.encode_chunk(chunk_frames(j)) for j in range(chunks)]
    streams = ViewStreams(bundle, frames_per_chunk=FRAMES_PER_CHUNK, frozen_first_chunk=False)
    sink, _ = streams.start(firsts)
    assert torch.equal(streams.encode_observation(firsts), sink)   # chunk 0's observation: the sink's latent
    got = []
    for j in range(chunks):
        got.append(streams.encode_chunk(chunk_frames(j)))
        starts = [frames[-1] for frames in chunk_frames(j)]       # the next chunk's start frame: this chunk's last
        observation = streams.encode_observation(starts)
        alone = pack_multiview_latents([encode_window(bundle, [frame]) for frame in starts])[0]
        assert torch.equal(observation, alone) and observation.shape[2] == 1
    for j in range(chunks):
        assert torch.equal(got[j], expected[j]), f"chunk {j}: the observation encodes moved the stream"
    with pytest.raises(ValueError, match="observation frames"):
        streams.encode_observation(firsts[:1] * (views + 1))
