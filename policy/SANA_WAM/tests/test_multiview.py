"""CPU tests for the vendored pack/unpack multiview latent helpers."""

from __future__ import annotations

import sys

import torch

ADAPTER_DIR = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/XPolicyLab/policy/SANA_WAM"
if ADAPTER_DIR not in sys.path:
    sys.path.insert(0, ADAPTER_DIR)

from sana_wam_min.multiview import (  # noqa: E402
    pack_multiview_latents,
    pack_spatial_views,
    unpack_multiview_latents,
    unpack_spatial_views,
)


def _three_views() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    return [torch.randn(1, 128, 4, 8, 10, generator=g, dtype=torch.float32).to(torch.bfloat16) for _ in range(3)]


def test_pack_three_views_to_strip_and_back():
    views = _three_views()
    strip, shapes = pack_multiview_latents(views)
    assert strip.shape == (1, 128, 4, 1, 240)
    assert strip.dtype == torch.bfloat16
    assert shapes == ((8, 10), (8, 10), (8, 10))
    back = unpack_multiview_latents(strip, shapes)
    assert len(back) == 3
    for a, b in zip(back, views, strict=True):
        assert a.shape == (1, 128, 4, 8, 10)
        assert torch.equal(a, b)


def test_strip_token_order_is_view_major_then_row_major():
    views = _three_views()
    strip, _ = pack_multiview_latents(views)
    # view v occupies columns [80v, 80v+80); inside a view the order is (y, x) row-major.
    for v, view in enumerate(views):
        assert torch.equal(strip[..., 0, 80 * v : 80 * (v + 1)], view.flatten(-2))
        assert torch.equal(strip[0, :, :, 0, 80 * v + 3 * 10 + 7], view[0, :, :, 3, 7])


def test_single_view_keeps_native_grid():
    view = _three_views()[0]
    packed, shapes = pack_multiview_latents([view])
    assert packed is view
    assert shapes == ((8, 10),)
    (back,) = unpack_multiview_latents(packed, shapes)
    assert back is view


def test_mixed_shapes_round_trip():
    a = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 2, 3, 4, 5)
    b = torch.arange(2 * 3 * 2 * 7, dtype=torch.float32).reshape(1, 2, 3, 2, 7) + 1000
    strip, shapes = pack_spatial_views([a, b])
    assert strip.shape == (1, 2, 3, 1, 20 + 14)
    assert shapes == ((4, 5), (2, 7))
    ra, rb = unpack_spatial_views(strip, shapes)
    assert torch.equal(ra, a) and torch.equal(rb, b)


def test_package_import_paths():
    import importlib

    a = importlib.import_module("sana_wam_min.multiview")
    b = importlib.import_module("XPolicyLab.policy.SANA_WAM.sana_wam_min.multiview")
    view = torch.zeros(1, 2, 1, 2, 2)
    assert a.pack_multiview_latents([view, view])[0].shape == b.pack_multiview_latents([view, view])[0].shape
