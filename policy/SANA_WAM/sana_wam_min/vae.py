"""LTX-2.3 VAE bundle: the diffusers decoder/buffers plus the vendored causal encoder used for training latents.

Port of the ``LTX2VAE_diffusers`` branches of Sana's ``diffusion/model/builder.py``
(``get_vae`` / ``_attach_ltx2_causal_encoder`` / ``vae_encode`` / ``vae_decode``). Encoding
always goes through :class:`AutoencoderKLCausalLTX2Video` (the module that produced every
training latent); the diffusers model supplies ``decode``, ``latents_mean``/``latents_std`` and
``config.scaling_factor``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
from diffusers.models.autoencoders import AutoencoderKLLTX2Video

from .ltx2_causal_vae import AutoencoderKLCausalLTX2Video

LATENT_BUFFER_KEYS = {"latents_mean", "latents_std"}
DECODE_TILING = {
    "tile_sample_min_height": 256,
    "tile_sample_min_width": 256,
    "tile_sample_stride_height": 192,
    "tile_sample_stride_width": 192,
}


@dataclass
class VaeBundle:
    """The loaded VAE pieces the session needs: decoder model, causal encoder, normalization buffers."""

    diffusers_vae: AutoencoderKLLTX2Video
    causal_encoder: AutoencoderKLCausalLTX2Video
    latents_mean: torch.Tensor
    latents_std: torch.Tensor
    scaling_factor: float
    temporal_compression: int = 8
    spatial_compression: int = 32
    load_info: dict = field(default_factory=dict)

    @property
    def dtype(self) -> torch.dtype:
        return self.causal_encoder.dtype

    @property
    def device(self) -> torch.device:
        return self.causal_encoder.device


def causal_config_from_diffusers(vae: AutoencoderKLLTX2Video) -> dict:
    """Return the diffusers VAE config as kwargs of ``AutoencoderKLCausalLTX2Video`` (lists -> tuples)."""

    cfg = {k: v for k, v in dict(vae.config).items() if not k.startswith("_")}
    # The causal class names the decoder field decoder_upsample_type; encoder weights are unaffected.
    if "upsample_type" in cfg:
        cfg["decoder_upsample_type"] = cfg.pop("upsample_type")
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()}


def build_causal_encoder(
    vae: AutoencoderKLLTX2Video, device, dtype: torch.dtype
) -> tuple[AutoencoderKLCausalLTX2Video, dict]:
    """Build the encoder-only causal VAE from ``vae``'s config and copy its ``encoder.*`` weights and latent buffers."""

    causal_vae = AutoencoderKLCausalLTX2Video(**causal_config_from_diffusers(vae)).eval()
    causal_vae.decoder = None
    encoder_state = {
        key: value
        for key, value in vae.state_dict().items()
        if key.startswith("encoder.") or key in LATENT_BUFFER_KEYS
    }
    result = causal_vae.load_state_dict(encoder_state, strict=False)
    missing_encoder = [k for k in result.missing_keys if k.startswith("encoder.") or k in LATENT_BUFFER_KEYS]
    if result.unexpected_keys or missing_encoder:
        raise RuntimeError(
            f"causal encoder state mismatch: unexpected={list(result.unexpected_keys)} missing={missing_encoder}"
        )
    causal_vae.to(device=device, dtype=dtype)
    causal_vae.requires_grad_(False)
    load_info = {
        "loaded_keys": len(encoder_state),
        "missing_keys": len(result.missing_keys),
        "unexpected_keys": len(result.unexpected_keys),
        "temporal_compression_ratio": int(causal_vae.temporal_compression_ratio),
    }
    return causal_vae, load_info


def resolve_vae_subfolder(vae_path: str) -> str | None:
    """Return the diffusers ``subfolder`` for ``vae_path``: ``None`` when it is the VAE folder itself, else ``"vae"``.

    Hub ids and repo roots (local dirs holding a ``vae/`` subdir) load ``subfolder="vae"``. A local
    dir is the VAE folder itself when its basename is ``vae`` or its own ``config.json`` declares
    ``_class_name == "AutoencoderKLLTX2Video"``.
    """

    path = str(vae_path)
    if os.path.isdir(path):
        if os.path.basename(os.path.normpath(path)) == "vae":
            return None
        config_path = os.path.join(path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as handle:
                if json.load(handle).get("_class_name") == "AutoencoderKLLTX2Video":
                    return None
    return "vae"


def load_vae(vae_path: str, device="cuda", dtype: torch.dtype = torch.bfloat16) -> VaeBundle:
    """Load ``AutoencoderKLLTX2Video`` from ``vae_path`` (hub id, repo root or VAE folder) and attach the causal encoder."""

    subfolder = resolve_vae_subfolder(vae_path)
    vae = (
        AutoencoderKLLTX2Video.from_pretrained(vae_path, subfolder=subfolder, torch_dtype=dtype)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    causal_encoder, load_info = build_causal_encoder(vae, device=device, dtype=dtype)
    return VaeBundle(
        diffusers_vae=vae,
        causal_encoder=causal_encoder,
        latents_mean=vae.latents_mean,
        latents_std=vae.latents_std,
        scaling_factor=float(vae.config.scaling_factor),
        temporal_compression=int(vae.config.temporal_compression_ratio),
        spatial_compression=int(vae.config.spatial_compression_ratio),
        load_info=load_info,
    )


@torch.no_grad()
def encode_video(bundle: VaeBundle, video_bcthw_minus1_1: torch.Tensor) -> torch.Tensor:
    """Encode ``[B, 3, T, H, W]`` pixels in ``[-1, 1]`` to the normalized latent ``[B, C, (T-1)//8+1, H/32, W/32]``.

    ``z = (mode - latents_mean) * scaling_factor / latents_std``, computed in the encoder dtype
    (bf16) from the posterior mode, and returned in the bundle dtype.
    """

    encoder = bundle.causal_encoder
    posterior = encoder.encode(video_bcthw_minus1_1.to(device=encoder.device, dtype=encoder.dtype)).latent_dist
    z = posterior.mode()
    mean = bundle.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    std = bundle.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    return ((z - mean) * bundle.scaling_factor / std).to(bundle.dtype)


@torch.no_grad()
def decode_latent(bundle: VaeBundle, latent: torch.Tensor, tiled: bool = True) -> torch.Tensor:
    """Decode a normalized ``[B, C, F, h, w]`` latent to ``[B, 3, (F-1)*8+1, 32h, 32w]`` pixels in ``[-1, 1]`` (diagnostic).

    ``tiled=True`` reproduces the validation decode tiling (256/192 px tiles, 64 px crossfade).
    """

    vae = bundle.diffusers_vae
    was_tiling = bool(getattr(vae, "use_tiling", False))
    if tiled:
        vae.enable_tiling(**DECODE_TILING)
    try:
        mean = bundle.latents_mean.view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        std = bundle.latents_std.view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        z = (latent * std / bundle.scaling_factor + mean).to(vae.dtype)
        return vae.decode(z, temb=None, return_dict=False)[0]
    finally:
        if tiled and not was_tiling:
            vae.disable_tiling()


def video_to_uint8(video_f3hw: torch.Tensor) -> torch.Tensor:
    """Map ``[-1, 1]`` pixels to uint8 on CPU: ``clamp(127.5 * x + 127.5, 0, 255)``."""

    return torch.clamp(127.5 * video_f3hw + 127.5, 0, 255).cpu().to(torch.uint8)


__all__ = [
    "DECODE_TILING",
    "VaeBundle",
    "build_causal_encoder",
    "causal_config_from_diffusers",
    "decode_latent",
    "encode_video",
    "load_vae",
    "resolve_vae_subfolder",
    "video_to_uint8",
]
