"""The sparse memory's pixel path and its causal VAE encode.

Pixels: the training data layer quantizes every canvas it decodes (Sana ``causal_chunk_window_data.frames_to_uint8``:
``((x + 1) * 127.5).round().clamp(0, 255)`` with torch's round-half-even) and the trainer turns the uint8 payload back
into the encoder input in float32 before the cast (``sparse_memory.pixels_from_uint8``: ``x / 127.5 - 1``). The deploy
keeps exactly that path (the bidirectional SANA_WAM front-end encodes its float canvas directly, which the
sparse-memory recipes never trained on).

Latents: a sample's [S | M_1..M_n | T] is ONE causal encode of its 1 + 8n + 8 canvases (the trainer's
``causal_vae_encode``: ``AutoencoderKLCausalLTX2Video.encode`` with ``causal=True``, the first frame alone, then groups
of 8, normalized as ``vae_encode`` does). A decision encodes its prefix [sink | memory rows] alone the same way (Sana
``sparse_memory_session.encode_prefix``): the causal encoder gives a frame group nothing of the later ones, so the
prefix alone yields the training's S and M_1..M_n (Sana ``test_sparse_memory_vae.py``). The memory grid moves with t,
so two decisions share no latent and nothing is streamed across decisions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.vae import VaeBundle  # noqa: E402

from . import sparse_memory as sm  # noqa: E402


def frames_to_uint8(frames: torch.Tensor) -> torch.Tensor:
    """Quantize ``[-1, 1]`` canvases to uint8 (Sana ``causal_chunk_window_data.frames_to_uint8``)."""

    return ((frames + 1.0) * 127.5).round_().clamp_(0, 255).to(torch.uint8)


def stack_canvases(canvases: Sequence[torch.Tensor]) -> torch.Tensor:
    """``[3, H, W]`` uint8 canvases -> ``[F, 3, H, W]``."""

    if not canvases:
        raise ValueError("no canvas to encode")
    frames = torch.stack([torch.as_tensor(c) for c in canvases], dim=0)
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"canvases must be uint8 [3, H, W], got {tuple(frames.shape)} {frames.dtype}")
    return frames


@torch.no_grad()
def encode_causal(bundle: VaeBundle, canvases: Sequence[torch.Tensor]) -> torch.Tensor:
    """One causal encode of ``1 + 8k`` uint8 canvases -> normalized ``[1, C, 1 + k, h, w]`` in the encoder dtype (Sana
    ``encode_prefix`` / the trainer's ``causal_vae_encode`` of ``pixels_from_uint8``)."""

    frames = stack_canvases(canvases)
    if (frames.shape[0] - 1) % sm.FRAMES_PER_LATENT:
        raise ValueError(f"a causal encode takes 1 + 8k frames, got {frames.shape[0]}")
    encoder = bundle.causal_encoder
    pixels = sm.pixels_from_uint8(frames[None], encoder.dtype, encoder.device)
    latents = encoder.encode(pixels, causal=True).latent_dist.mode()
    mean = bundle.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = bundle.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - mean) * bundle.scaling_factor / std


def canvas_uint8(canvas_float: torch.Tensor) -> torch.Tensor:
    """A float ``[3, H, W]`` canvas in [-1, 1] (``sana_pixel_canvas_from_frames``) quantized as training stores it."""

    if canvas_float.ndim != 3 or canvas_float.shape[0] != 3:
        raise ValueError(f"a canvas is [3, H, W], got {tuple(canvas_float.shape)}")
    return frames_to_uint8(canvas_float.float().clone())


def as_uint8_hwc(canvas: torch.Tensor) -> np.ndarray:
    """A uint8 ``[3, H, W]`` canvas as an HxWx3 array (debug dumps)."""

    return canvas.permute(1, 2, 0).cpu().numpy()


__all__ = ["as_uint8_hwc", "canvas_uint8", "encode_causal", "frames_to_uint8", "stack_canvases"]
