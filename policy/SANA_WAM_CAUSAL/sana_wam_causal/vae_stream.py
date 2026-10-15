"""The episode's causal VAE streams: every latent the causal policy's memory holds (Sana ``deploy/causal/view_stream.py``
``CausalViewStream``, episode protocol).

Training encodes each episode window once with the causal LTX-2.3 encoder (Sana ``crwm.encode_window_chunks``, the
phase-window store): per view, the window's first frame alone gives the sink, then every C / s frames (C actions per
chunk, video frame stride s) give one chunk's target latents, each encoded after the window's whole past. A padded window
is the phase whose head is one whole chunk: its chunk 0 is the first frame frozen over the chunk's C / s frames, which
the stream encodes after the sink. A first-frame window (``aligned_phase_window: first_frame``; Sana ``9fca24879``,
``CausalViewStream.start`` returns the sink alone) opens on the sink alone, its chunk 0 the executed one: the causal
encoder pads with the first frame, so its chunk latents equal the padded stream's up to float32 round-off (the phase
store serves that window as the wholly frozen phase's member less its chunk 0, measured bit-identical by Sana's audit;
<= 4e-7 apart on this CPU stream, ``test_vae_stream``).
The encoder is causal, so feeding the same frames as a stream on one persistent feature cache (the first frame on a
fresh cache, then 8-frame blocks) reproduces that window encode latent for latent (Sana ``test_causal_view_stream``),
and a deployment encodes every observed frame once.

``ViewStreams`` keeps one stream per view, each on a feature cache of its own (the encoder's default cache is cleared by
every one-shot ``encode``, e.g. an observation encode in between), one view at a time as the phase store was built
(``--batch-windows 1``, every view encoded on its own). Frames are the float ``[3, H, W]`` model tensors of the
SFT pixel path (``sana_wam_min.pixels.frame_to_model_tensor`` per view, ``sana_pixel_canvas_from_frames`` for the
canvas), cast to the encoder dtype at the encode, as the trainer casts the dataset's float32 pixels.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.ltx2_causal_vae import EncoderCacheManager  # noqa: E402
from sana_wam_min.multiview import pack_multiview_latents  # noqa: E402
from sana_wam_min.vae import VaeBundle  # noqa: E402

# the causal encoder's temporal block: one latent frame per 8 pixel frames after the first
ENCODER_BLOCK = 8


def normalize_latent(bundle: VaeBundle, z: torch.Tensor) -> torch.Tensor:
    """``(mode - latents_mean) * scaling_factor / latents_std`` in the encoder dtype (``encode_video``)."""

    mean = bundle.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    std = bundle.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    return ((z - mean) * bundle.scaling_factor / std).to(bundle.dtype)


def stack_frames(frames: Sequence[torch.Tensor]) -> torch.Tensor:
    """Float ``[3, H, W]`` frames in [-1, 1] -> the encoder input ``[1, 3, F, H, W]``."""

    if not frames:
        raise ValueError("no frame to encode")
    stacked = torch.stack([torch.as_tensor(frame) for frame in frames], dim=1)
    if stacked.ndim != 4 or stacked.shape[0] != 3 or not stacked.is_floating_point():
        raise ValueError(f"frames must be float [3, H, W] tensors, got {tuple(stacked.shape)} {stacked.dtype}")
    return stacked.unsqueeze(0)


class LatentStream:
    """One view's causal-VAE stream: ``start`` with the first frame, then ``extend`` by whole 8-frame blocks."""

    def __init__(self, bundle: VaeBundle) -> None:
        self.bundle = bundle
        self._cache = EncoderCacheManager()
        self.latents = 0

    def _encode(self, frames: Sequence[torch.Tensor], reset: bool) -> torch.Tensor:
        pixels = stack_frames(frames)
        encoder = self.bundle.causal_encoder
        saved = encoder._encoder_cache
        encoder._encoder_cache = self._cache
        try:
            posterior = encoder.streaming_causal_encode(
                pixels.to(device=encoder.device, dtype=encoder.dtype), causal=True, reset_cache=reset
            ).latent_dist
        finally:
            encoder._encoder_cache = saved
        # the streaming entry skips the pending-frame check of ``encode``; every call here is 1 frame or whole
        # blocks, so a leftover would mean a broken grouping (and a silently misaligned next latent)
        self._cache.check_pending_consumed()
        latent = normalize_latent(self.bundle, posterior.mode())
        expected = 1 if reset else len(frames) // ENCODER_BLOCK
        if latent.shape[2] != expected:
            raise RuntimeError(f"a stream step of {len(frames)} frames must yield {expected} latents, got {tuple(latent.shape)}")
        self.latents += expected
        return latent

    @torch.no_grad()
    def start(self, first_frame: torch.Tensor) -> torch.Tensor:
        """Reset the stream and encode its first frame alone -> latent 0 ``[1, C, 1, h, w]``."""

        self.latents = 0
        return self._encode([first_frame], reset=True)

    @torch.no_grad()
    def extend(self, frames: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode the next whole blocks of frames -> their latents ``[1, C, len(frames) / 8, h, w]``."""

        if self.latents == 0:
            raise RuntimeError("start the stream with its first frame before extending it")
        if not frames or len(frames) % ENCODER_BLOCK:
            raise ValueError(f"a stream extends by whole blocks of {ENCODER_BLOCK} frames, got {len(frames)}")
        return self._encode(frames, reset=False)


class ViewStreams:
    """The per-view streams of one episode (Sana ``CausalViewStream``): ``start`` gives the sink and, on a padded
    window (``frozen_first_chunk``), the frozen first chunk; ``encode_chunk`` every executed chunk's target latents, each
    packed as the policy reads it (the one-view grid, or the view-major strip of several views)."""

    def __init__(self, bundle: VaeBundle, *, frames_per_chunk: int, frozen_first_chunk: bool = True) -> None:
        self.bundle = bundle
        self.frames_per_chunk = int(frames_per_chunk)
        self.frozen_first_chunk = bool(frozen_first_chunk)
        if self.frames_per_chunk < 1 or self.frames_per_chunk % ENCODER_BLOCK:
            raise ValueError(f"a chunk spans {frames_per_chunk} frames; the causal encoder streams whole blocks of {ENCODER_BLOCK}")
        self._streams: list[LatentStream] = []
        self.view_shapes: Optional[tuple[tuple[int, int], ...]] = None
        self.chunks = 0

    def _pack(self, latents: Sequence[torch.Tensor]) -> torch.Tensor:
        packed, shapes = pack_multiview_latents(list(latents))
        if self.view_shapes is None:
            self.view_shapes = tuple(shapes)
        elif tuple(shapes) != self.view_shapes:
            raise RuntimeError(f"the views changed shape within an episode: {shapes} vs {self.view_shapes}")
        return packed

    @torch.no_grad()
    def start(self, first_frames: Sequence[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Open every view's stream at the episode's first frame -> (sink ``[1, C, 1, ...]``, frozen first chunk
        ``[1, C, f_c, ...]`` or None): the first frame alone, then on a padded window the first frame repeated over one
        chunk's frames, encoded after it (the phase window whose head is one whole chunk); a first-frame window opens on
        the sink alone."""

        if not first_frames:
            raise ValueError("start needs one first frame per view")
        self._streams = [LatentStream(self.bundle) for _ in first_frames]
        self.view_shapes = None
        sinks = [stream.start(frame) for stream, frame in zip(self._streams, first_frames)]
        if not self.frozen_first_chunk:
            self.chunks = 0
            return self._pack(sinks), None
        frozen = [
            stream.extend([frame] * self.frames_per_chunk) for stream, frame in zip(self._streams, first_frames)
        ]
        self.chunks = 1
        return self._pack(sinks), self._pack(frozen)

    @torch.no_grad()
    def encode_chunk(self, frames_per_view: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
        """Encode the next executed chunk's observed frames (per view, the C / s frames after its observation frame up
        to the next chunk's, i.e. ticks s, 2s, .., C) after everything the streams have seen -> its target latents."""

        if not self._streams:
            raise RuntimeError("open the streams at the episode's first frame (start) before encoding a chunk")
        if len(frames_per_view) != len(self._streams):
            raise ValueError(f"{len(self._streams)} views stream, got frames for {len(frames_per_view)}")
        for frames in frames_per_view:
            if len(frames) != self.frames_per_chunk:
                raise ValueError(f"a chunk carries {self.frames_per_chunk} frames per view, got {len(frames)}")
        targets = [stream.extend(list(frames)) for stream, frames in zip(self._streams, frames_per_view)]
        self.chunks += 1
        return self._pack(targets)


@torch.no_grad()
def encode_window(bundle: VaeBundle, frames: Sequence[torch.Tensor]) -> torch.Tensor:
    """One-shot causal encode of one view's ``1 + 8n`` frames -> ``[1, C, 1 + n, h, w]`` (the store's window encode)."""

    pixels = stack_frames(frames)
    if (pixels.shape[2] - 1) % ENCODER_BLOCK:
        raise ValueError(f"a causal window encode takes 1 + 8n frames, got {pixels.shape[2]}")
    encoder = bundle.causal_encoder
    z = encoder.encode(pixels.to(device=encoder.device, dtype=encoder.dtype)).latent_dist.mode()
    return normalize_latent(bundle, z)


__all__ = ["ENCODER_BLOCK", "LatentStream", "ViewStreams", "encode_window", "normalize_latent", "stack_frames"]
