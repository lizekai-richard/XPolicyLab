"""CPU tests for sana_wam_min.vae: causal-encoder construction from the diffusers config, without downloading weights."""

from __future__ import annotations

import inspect
import json
import os
import sys

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLLTX2Video

ADAPTER_DIR = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/XPolicyLab/policy/SANA_WAM"
if ADAPTER_DIR not in sys.path:
    sys.path.insert(0, ADAPTER_DIR)

from sana_wam_min import vae as vae_mod  # noqa: E402
from sana_wam_min.ltx2_causal_vae import AutoencoderKLCausalLTX2Video  # noqa: E402

VAE_ROOT = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/models/Sana/LTX-2.3-Diffusers"
VAE_CONFIG = os.path.join(VAE_ROOT, "vae", "config.json")

EXPECTED_CONFIG = {
    "block_out_channels": [256, 512, 1024, 1024],
    "decoder_block_out_channels": [256, 512, 512, 1024],
    "decoder_causal": False,
    "encoder_causal": True,
    "latent_channels": 128,
    "patch_size": 4,
    "patch_size_t": 1,
    "scaling_factor": 1.0,
    "spatial_compression_ratio": 32,
    "temporal_compression_ratio": 8,
    "upsample_type": ["spatiotemporal", "spatiotemporal", "temporal", "spatial"],
}


def _real_config() -> dict:
    if not os.path.isfile(VAE_CONFIG):
        pytest.skip(f"local VAE snapshot missing: {VAE_CONFIG}")
    with open(VAE_CONFIG) as f:
        return json.load(f)


def _tiny_config() -> dict:
    cfg = {k: v for k, v in _real_config().items() if not k.startswith("_")}
    cfg.update(
        block_out_channels=[16, 32, 32, 32],
        decoder_block_out_channels=[16, 32, 32, 32],
        layers_per_block=[1, 1, 1, 1, 1],
        decoder_layers_per_block=[1, 1, 1, 1, 1],
        latent_channels=4,
    )
    return cfg


HUB_ID = "Efficient-Large-Model/LTX-2.3-Diffusers"


def test_subfolder_resolution(tmp_path):
    assert vae_mod.resolve_vae_subfolder(HUB_ID) == "vae"
    assert vae_mod.resolve_vae_subfolder(VAE_ROOT) == "vae"
    assert vae_mod.resolve_vae_subfolder(os.path.join(VAE_ROOT, "vae")) is None
    # A renamed copy of the VAE folder is recognized through its own config.json.
    renamed = tmp_path / "ltx_vae_weights"
    renamed.mkdir()
    (renamed / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLLTX2Video"}))
    assert vae_mod.resolve_vae_subfolder(str(renamed)) is None
    # A local dir with neither marker is treated like a repo root.
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"_class_name": "SomethingElse"}))
    assert vae_mod.resolve_vae_subfolder(str(other)) == "vae"


def test_load_vae_passes_subfolder_to_from_pretrained(monkeypatch):
    calls: list[dict] = []
    tiny = _tiny_config()

    class FakeAutoencoder:
        @staticmethod
        def from_pretrained(path, subfolder=None, torch_dtype=None):
            calls.append({"path": path, "subfolder": subfolder, "torch_dtype": torch_dtype})
            return AutoencoderKLLTX2Video.from_config(tiny)

    monkeypatch.setattr(vae_mod, "AutoencoderKLLTX2Video", FakeAutoencoder)
    for path, expected in ((HUB_ID, "vae"), (VAE_ROOT, "vae"), (os.path.join(VAE_ROOT, "vae"), None)):
        bundle = vae_mod.load_vae(path, device="cpu", dtype=torch.float32)
        assert calls[-1] == {"path": path, "subfolder": expected, "torch_dtype": torch.float32}, path
        assert bundle.temporal_compression == 8 and bundle.causal_encoder.decoder is None
    assert len(calls) == 3


def test_real_config_matches_spec():
    cfg = _real_config()
    assert cfg["_class_name"] == "AutoencoderKLLTX2Video"
    for key, value in EXPECTED_CONFIG.items():
        assert cfg[key] == value, key
    assert cfg["decoder_causal"] is False and cfg["encoder_causal"] is True


def test_causal_config_keys_map_onto_causal_class_init():
    vae = AutoencoderKLLTX2Video.from_config(_tiny_config())
    cfg = vae_mod.causal_config_from_diffusers(vae)
    assert not any(k.startswith("_") for k in cfg)
    assert "upsample_type" not in cfg and cfg["decoder_upsample_type"] == (
        "spatiotemporal",
        "spatiotemporal",
        "temporal",
        "spatial",
    )
    assert all(not isinstance(v, list) for v in cfg.values())
    params = inspect.signature(AutoencoderKLCausalLTX2Video.__init__).parameters
    unknown = sorted(set(cfg) - set(params))
    assert unknown == [], unknown
    # Every real-config key (minus the two '_' keys, after the rename) is an explicit kwarg.
    real_keys = {("decoder_upsample_type" if k == "upsample_type" else k) for k in _real_config() if not k.startswith("_")}
    assert real_keys <= set(params)


def test_tiny_causal_encoder_state_copy_and_encode_shapes():
    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video.from_config(_tiny_config()).eval()
    with torch.no_grad():
        vae.latents_mean.copy_(torch.linspace(-0.5, 0.4, 4))
        vae.latents_std.copy_(torch.linspace(0.1, 0.9, 4))
    enc, info = vae_mod.build_causal_encoder(vae, device="cpu", dtype=torch.float32)
    assert info["unexpected_keys"] == 0 and info["missing_keys"] == 0
    assert info["temporal_compression_ratio"] == 8
    assert enc.decoder is None
    assert info["loaded_keys"] == sum(1 for k in vae.state_dict() if k.startswith("encoder.")) + 2
    for key, value in vae.state_dict().items():
        if key.startswith("encoder.") or key in {"latents_mean", "latents_std"}:
            assert torch.equal(enc.state_dict()[key], value), key
    assert not any(p.requires_grad for p in enc.parameters())

    bundle = vae_mod.VaeBundle(
        diffusers_vae=vae,
        causal_encoder=enc,
        latents_mean=vae.latents_mean,
        latents_std=vae.latents_std,
        scaling_factor=float(vae.config.scaling_factor),
        temporal_compression=8,
        spatial_compression=32,
        load_info=info,
    )
    assert bundle.dtype == torch.float32 and bundle.device.type == "cpu"

    clip = torch.rand(1, 3, 25, 64, 64) * 2 - 1
    z = vae_mod.encode_video(bundle, clip)
    assert z.shape == (1, 4, 4, 2, 2)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()
    with torch.no_grad():
        mode = enc.encode(clip).latent_dist.mode()
    expected = (mode - vae.latents_mean.view(1, -1, 1, 1, 1)) / vae.latents_std.view(1, -1, 1, 1, 1)
    assert torch.equal(z, expected)

    # Frame 0 of the 25-frame chunked encode equals the standalone 1-frame encode (fresh cache, same ops).
    z1 = vae_mod.encode_video(bundle, clip[:, :, :1])
    assert z1.shape == (1, 4, 1, 2, 2)
    assert torch.allclose(z1, z[:, :, :1], atol=1e-5, rtol=1e-5)

    pixels = vae_mod.decode_latent(bundle, z1, tiled=False)
    assert pixels.shape == (1, 3, 1, 64, 64)
    pixels_tiled = vae_mod.decode_latent(bundle, z1, tiled=True)
    assert pixels_tiled.shape == (1, 3, 1, 64, 64)
    assert vae.use_tiling is False
    u8 = vae_mod.video_to_uint8(pixels[0].permute(1, 0, 2, 3))
    assert u8.shape == (1, 3, 64, 64) and u8.dtype == torch.uint8


def test_full_size_causal_encoder_construction_from_real_config():
    """Build the real-size encoder from the local config (random init) and check the 86-tensor state copy."""

    vae = AutoencoderKLLTX2Video.from_config(_real_config()).eval()
    enc, info = vae_mod.build_causal_encoder(vae, device="cpu", dtype=torch.bfloat16)
    assert info == {"loaded_keys": 86, "missing_keys": 0, "unexpected_keys": 0, "temporal_compression_ratio": 8}
    assert enc.dtype == torch.bfloat16
    assert enc.latents_mean.shape == (128,) and enc.latents_std.shape == (128,)
    assert enc.encoder.conv_out.conv.weight.shape[0] == 129


@pytest.mark.skipif(
    os.environ.get("SANA_WAM_TEST_REAL_VAE") != "1",
    reason="loads the 1.4 GB LTX-2.3 VAE weights and runs a CPU bf16 encode; set SANA_WAM_TEST_REAL_VAE=1",
)
def test_load_vae_real_weights_cpu():
    bundle = vae_mod.load_vae(VAE_ROOT, device="cpu", dtype=torch.bfloat16)
    assert bundle.load_info == {"loaded_keys": 86, "missing_keys": 0, "unexpected_keys": 0, "temporal_compression_ratio": 8}
    assert bundle.scaling_factor == 1.0
    assert bundle.temporal_compression == 8 and bundle.spatial_compression == 32
    assert bundle.latents_mean.dtype == torch.bfloat16
    assert abs(float(bundle.latents_mean[0]) - 0.022216796875) < 1e-6
    assert abs(float(bundle.latents_std[0]) - 0.23828125) < 1e-6
    frame = torch.full((1, 3, 1, 256, 320), -1.0, dtype=torch.bfloat16)
    z = vae_mod.encode_video(bundle, frame)
    assert z.shape == (1, 128, 1, 8, 10) and z.dtype == torch.bfloat16
    assert torch.isfinite(z).all()
    assert float(z.float().abs().mean()) < 10.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="frame-0 parity on real weights needs a GPU (bf16 encode)")
def test_frame0_parity_real_weights_gpu():
    bundle = vae_mod.load_vae(VAE_ROOT, device="cuda", dtype=torch.bfloat16)
    clip = (torch.rand(1, 3, 25, 256, 320, device="cuda") * 2 - 1).to(torch.bfloat16)
    z25 = vae_mod.encode_video(bundle, clip)
    z1 = vae_mod.encode_video(bundle, clip[:, :, :1])
    assert z25.shape == (1, 128, 4, 8, 10) and z1.shape == (1, 128, 1, 8, 10)
    assert float((z25[:, :, :1].float() - z1.float()).abs().max()) <= 1e-2
