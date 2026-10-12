"""The hierarchical memory's pixel path and causal-VAE streams.

Pixels: the training data layer quantizes every canvas it decodes (Sana ``causal_chunk_window_data.frames_to_uint8``:
``((x + 1) * 127.5).round().clamp(0, 255)`` with torch's round-half-even) and the trainer turns the uint8 payload back
into the encoder input in float32 before the cast (``hierarchical_memory.pixels_from_uint8``: ``x / 127.5 - 1``). The
deploy keeps exactly that path (the bidirectional SANA_WAM front-end encodes its float canvas directly, which the
hierarchical-memory recipes never trained on).

Latents: every stream of a training sample is ONE causal encode (``AutoencoderKLCausalLTX2Video.encode``: the first frame
alone, then groups of 8, one persistent per-layer feature cache) normalized as ``vae_encode`` does. ``LatentStream``
runs the same calls incrementally through ``streaming_causal_encode`` on a feature cache of its own (the encoder's
default cache is cleared by every one-shot ``encode``), so a stream fed 1 frame and then 8 frames per call returns, bit
for bit on one device / dtype / kernel path, the latents of the one-shot encode of the concatenation (Sana plan
section 1: "streaming == one-shot, maxdiff 0"; the adapter tests check it on a tiny VAE).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.ltx2_causal_vae import EncoderCacheManager  # noqa: E402
from sana_wam_min.vae import VaeBundle  # noqa: E402

from . import memory as hm  # noqa: E402


def frames_to_uint8(frames: torch.Tensor) -> torch.Tensor:
    """Quantize ``[-1, 1]`` canvases to uint8 (Sana ``causal_chunk_window_data.frames_to_uint8``)."""

    return ((frames + 1.0) * 127.5).round_().clamp_(0, 255).to(torch.uint8)


def pixels_from_uint8(frames: torch.Tensor, dtype: torch.dtype, device=None) -> torch.Tensor:
    """uint8 canvases ``[B, F, 3, H, W]`` -> the causal VAE input ``[B, 3, F, H, W]`` in [-1, 1] (Sana
    ``hierarchical_memory.pixels_from_uint8``: float32 normalization, then the cast)."""

    if frames.dtype != torch.uint8 or frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError(f"pixel payloads are uint8 [B, F, 3, H, W] canvases; got {tuple(frames.shape)} {frames.dtype}")
    pixels = frames.to(device=device).permute(0, 2, 1, 3, 4).to(torch.float32)
    return pixels.div_(127.5).sub_(1.0).to(dtype).contiguous()


def stack_canvases(canvases: Sequence[torch.Tensor]) -> torch.Tensor:
    """``[3, H, W]`` uint8 canvases -> ``[1, F, 3, H, W]``."""

    if not canvases:
        raise ValueError("no canvas to encode")
    frames = torch.stack([torch.as_tensor(c) for c in canvases], dim=0)
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"canvases must be uint8 [3, H, W], got {tuple(frames.shape)} {frames.dtype}")
    return frames.unsqueeze(0)


def normalize_latent(bundle: VaeBundle, z: torch.Tensor) -> torch.Tensor:
    """``(mode - latents_mean) * scaling_factor / latents_std`` in the encoder dtype (``vae_encode`` / ``encode_video``)."""

    mean = bundle.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    std = bundle.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    return ((z - mean) * bundle.scaling_factor / std).to(bundle.dtype)


@torch.no_grad()
def encode_once(bundle: VaeBundle, canvases: Sequence[torch.Tensor]) -> torch.Tensor:
    """One-shot causal encode of ``1 + 8n`` uint8 canvases -> normalized ``[1, C, 1 + n, h, w]`` (Sana
    ``causal_vae_encode`` of ``pixels_from_uint8``)."""

    pixels = pixels_from_uint8(stack_canvases(canvases), bundle.dtype, bundle.device)
    if (pixels.shape[2] - 1) % hm.FRAMES_PER_LATENT:
        raise ValueError(f"a causal encode takes 1 + 8n frames, got {pixels.shape[2]}")
    encoder = bundle.causal_encoder
    z = encoder.encode(pixels.to(device=encoder.device, dtype=encoder.dtype), causal=True).latent_dist.mode()
    return normalize_latent(bundle, z)


class LatentStream:
    """One causal-VAE stream encoded incrementally: ``start`` with the first frame, then ``extend`` by 8 frames per call
    (each call returns its one latent), on a feature cache of its own."""

    def __init__(self, bundle: VaeBundle) -> None:
        self.bundle = bundle
        self._cache = EncoderCacheManager()
        self.latents = 0

    def _encode(self, canvases: Sequence[torch.Tensor], reset: bool) -> torch.Tensor:
        pixels = pixels_from_uint8(stack_canvases(canvases), self.bundle.dtype, self.bundle.device)
        encoder = self.bundle.causal_encoder
        saved = encoder._encoder_cache
        encoder._encoder_cache = self._cache
        try:
            posterior = encoder.streaming_causal_encode(
                pixels.to(device=encoder.device, dtype=encoder.dtype), causal=True, reset_cache=reset
            ).latent_dist
        finally:
            encoder._encoder_cache = saved
        # the streaming entry skips the pending-state check of ``encode``; every call here is 1 or 8 frames, so a
        # leftover would mean a broken grouping (and a silently misaligned next latent)
        self._cache.check_pending_consumed()
        latent = normalize_latent(self.bundle, posterior.mode())
        if latent.shape[2] != 1:
            raise RuntimeError(f"a stream step must yield one latent frame, got {tuple(latent.shape)}")
        self.latents += 1
        return latent

    @torch.no_grad()
    def start(self, first_canvas: torch.Tensor) -> torch.Tensor:
        """Reset the stream and encode its first frame alone -> latent 0 ``[1, C, 1, h, w]``."""

        self.latents = 0
        return self._encode([first_canvas], reset=True)

    @torch.no_grad()
    def extend(self, canvases: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode the next 8 frames -> the stream's next latent ``[1, C, 1, h, w]``."""

        if self.latents == 0:
            raise RuntimeError("start the stream with its first frame before extending it")
        if len(canvases) != hm.FRAMES_PER_LATENT:
            raise ValueError(f"a stream extends by exactly {hm.FRAMES_PER_LATENT} frames, got {len(canvases)}")
        return self._encode(canvases, reset=False)


def canvas_uint8(canvas_float: torch.Tensor) -> torch.Tensor:
    """A float ``[3, H, W]`` canvas in [-1, 1] (``sana_pixel_canvas_from_frames``) quantized as training stores it."""

    if canvas_float.ndim != 3 or canvas_float.shape[0] != 3:
        raise ValueError(f"a canvas is [3, H, W], got {tuple(canvas_float.shape)}")
    return frames_to_uint8(canvas_float.float().clone())


def as_uint8_hwc(canvas: torch.Tensor) -> np.ndarray:
    """A uint8 ``[3, H, W]`` canvas as an HxWx3 array (debug dumps)."""

    return canvas.permute(1, 2, 0).cpu().numpy()


__all__ = [
    "LatentStream",
    "as_uint8_hwc",
    "canvas_uint8",
    "encode_once",
    "frames_to_uint8",
    "normalize_latent",
    "pixels_from_uint8",
    "stack_canvases",
]
