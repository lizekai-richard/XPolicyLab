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

Views: a row's frames are uint8 ``[V, 3, H, W]``, the one canvas (V = 1) or the sana_latent views; every view runs the
same causal encode with the views folded into the batch, and the views are packed into the strip
``[1, C, F, 1, sum(h_v * w_v)]`` (Sana ``encode_prefix`` / ``pack_multiview_latents``; one view keeps its grid).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.multiview import ViewShapes, pack_multiview_latents  # noqa: E402
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


def stack_views(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """One uint8 ``[V, 3, H, W]`` per row (a ``[3, H, W]`` canvas reads as V = 1) -> ``[V, F, 3, H, W]``."""

    if not rows:
        raise ValueError("no frame to encode")
    frames = [torch.as_tensor(row) for row in rows]
    frames = [row[None] if row.ndim == 3 else row for row in frames]
    if any(row.dtype != torch.uint8 or row.ndim != 4 or row.shape[1] != 3 for row in frames):
        raise ValueError(f"every row's frames must be uint8 [V, 3, H, W], got {[(tuple(r.shape), r.dtype) for r in frames]}")
    if len({tuple(row.shape) for row in frames}) != 1:
        raise ValueError(f"the rows' frames differ in shape: {sorted({tuple(row.shape) for row in frames})}")
    return torch.stack(frames, dim=1)


@torch.no_grad()
def encode_views(bundle: VaeBundle, rows: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ViewShapes]:
    """One causal encode of ``1 + 8k`` rows of uint8 frames -> (normalized latents in the encoder dtype, the views'
    latent shapes): ``[1, C, 1 + k, h, w]`` for one view, the strip ``[1, C, 1 + k, 1, sum(h_v * w_v)]`` for several
    (Sana ``encode_prefix`` / the trainer's ``causal_vae_encode`` of ``pixels_from_uint8``)."""

    frames = stack_views(rows)
    if (frames.shape[1] - 1) % sm.FRAMES_PER_LATENT:
        raise ValueError(f"a causal encode takes 1 + 8k frames, got {frames.shape[1]}")
    encoder = bundle.causal_encoder
    pixels = sm.pixels_from_uint8(frames, encoder.dtype, encoder.device)
    latents = encoder.encode(pixels, causal=True).latent_dist.mode()
    mean = bundle.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = bundle.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - mean) * bundle.scaling_factor / std
    packed, shapes = pack_multiview_latents(list(latents.unflatten(0, (1, latents.shape[0])).unbind(1)))
    return packed, tuple(shapes)


@torch.no_grad()
def encode_causal(bundle: VaeBundle, canvases: Sequence[torch.Tensor]) -> torch.Tensor:
    """One causal encode of ``1 + 8k`` uint8 ``[3, H, W]`` canvases -> normalized ``[1, C, 1 + k, h, w]`` (one view of
    ``encode_views``)."""

    return encode_views(bundle, list(stack_canvases(canvases)))[0]


def views_uint8(views_float: Sequence[torch.Tensor]) -> torch.Tensor:
    """Float ``[3, H, W]`` views in [-1, 1] (``frame_to_model_tensor``) quantized as training stores them ->
    ``[V, 3, H, W]`` uint8 (Sana ``layout_frames``)."""

    if any(view.ndim != 3 or view.shape[0] != 3 for view in views_float):
        raise ValueError(f"a view is [3, H, W], got {[tuple(view.shape) for view in views_float]}")
    return frames_to_uint8(torch.stack([view.float() for view in views_float]))


def canvas_uint8(canvas_float: torch.Tensor) -> torch.Tensor:
    """A float ``[3, H, W]`` canvas in [-1, 1] (``sana_pixel_canvas_from_frames``) quantized as training stores it."""

    if canvas_float.ndim != 3 or canvas_float.shape[0] != 3:
        raise ValueError(f"a canvas is [3, H, W], got {tuple(canvas_float.shape)}")
    return frames_to_uint8(canvas_float.float().clone())


def as_uint8_hwc(canvas: torch.Tensor) -> np.ndarray:
    """A uint8 ``[3, H, W]`` canvas as an HxWx3 array (debug dumps)."""

    return canvas.permute(1, 2, 0).cpu().numpy()


__all__ = [
    "as_uint8_hwc",
    "canvas_uint8",
    "encode_causal",
    "encode_views",
    "frames_to_uint8",
    "stack_canvases",
    "stack_views",
    "views_uint8",
]
