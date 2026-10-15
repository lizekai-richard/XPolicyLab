"""The attention probe (deploy key ``attention_dump_dir``): read-only, and its aggregates are the layers' own reads (CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import _paths  # noqa: F401

from sana_wam_causal.attention_probe import GDN_PARTS, QUERY_GROUPS, SOFTMAX_OUT_PARTS, AttentionProbe
from sana_wam_causal.contract import MEMORY_VIDEO_AND_ACTION, MEMORY_VIDEO_ONLY
from sana_wam_causal.layers import CausalGDNAttention, CausalSoftmaxAttention
from sana_wam_causal.session import Caption
from test_contract_and_memory import _open, _session, _tiny_model

CHUNKS = 4
WINDOW = 2


def _episode(model, probe=None, tmp_path=None):
    """The opening (sink, frozen chunk 0) and chunks 1..4 of one deterministic episode; returns the generated (action,
    video) per chunk and the dump paths."""

    g = torch.Generator().manual_seed(7)
    caption = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    null = Caption(torch.randn(1, 1, 1, 7, 16, generator=g), torch.ones(1, 1, 7, dtype=torch.int16))
    sink, frozen = torch.randn(1, 4, 1, 3, 4, generator=g), torch.randn(1, 4, 1, 3, 4, generator=g)
    clean = [torch.randn(1, 4, 1, 3, 4, generator=g) for _ in range(CHUNKS + 1)]
    mask = torch.zeros(1, 32, 80, dtype=torch.bool)
    mask[..., 7:17] = True
    mask[..., 36:46] = True
    state = torch.randn(1, 80, generator=g)
    session = _session(model, caption, null, window=WINDOW)
    session.probe = probe
    _open(session, sink, frozen, state, mask)
    outs, paths = [], []
    for c in range(1, CHUNKS + 1):
        action, video = session.generate_chunk(
            anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask,
            generator=torch.Generator().manual_seed(100 + c),
        )
        outs.append((action, video))
        if probe is not None:
            paths.append(probe.write(str(tmp_path), episode=1, extra={"note": "test"}))
        session.commit_chunk(clean[c], action, anchor_state=state, anchor_state_mask=mask[0, 0], action_mask=mask)
    return outs, paths


@pytest.mark.parametrize("memory_content", [MEMORY_VIDEO_AND_ACTION, MEMORY_VIDEO_ONLY], ids=["video_and_action", "video_only"])
def test_the_probe_changes_nothing_and_its_masses_are_distributions(memory_content, tmp_path):
    model = _tiny_model(window=WINDOW, memory_content=memory_content)
    plain, _ = _episode(model)
    probe = AttentionProbe(model)
    probed, paths = _episode(model, probe, tmp_path)
    for (a1, v1), (a2, v2) in zip(plain, probed):
        assert torch.equal(a1, a2) and torch.equal(v1, v2)
    assert not probe.active
    robot = memory_content == MEMORY_VIDEO_AND_ACTION
    groups = list(QUERY_GROUPS[1:])   # no observation group: a chunk is its targets and robot rows
    for c, path in enumerate(paths, start=1):
        assert path.endswith(f"ep001_chunk{c:03d}.npz")
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        read = list(range(max(0, c - WINDOW), c))
        assert meta["chunk"] == c and meta["read_ids"] == read and meta["note"] == "test" and meta["sink"]
        per_chunk = ["video", "state", "action"] if robot else ["video"]
        assert meta["query_groups"] == groups
        assert meta["key_labels"] == ["sink.video"] + [f"mem{i}.{part}" for i in read for part in per_chunk] + [f"cur.{q}" for q in groups]
        assert meta["query_sizes"] == [12, 1, 32]
        assert meta["memory_keys"] == 12 + len(read) * (12 + (33 if robot else 0))
        mass, out, gdn = data["softmax_mass"], data["softmax_out"], data["gdn"]
        steps, streams = 3, 2
        assert mass.shape == (steps, streams, len(meta["softmax_layers"]), meta["softmax_heads"], len(groups), len(meta["key_labels"]))
        assert out.shape[-1] == len(SOFTMAX_OUT_PARTS) and gdn.shape[-1] == len(GDN_PARTS)
        np.testing.assert_allclose(mass.sum(-1), 1.0, atol=1e-5)
        memory_mass = mass[..., : len(meta["key_labels"]) - len(groups)].sum(-1)
        assert np.all(mass[..., 0] > 0)                               # every chunk reads the sink
        assert np.all(memory_mass > 0) and np.all(out[..., 0] > 0) and np.all(gdn[..., 0] > 0)
        assert np.all(np.isfinite(gdn)) and np.all(np.isfinite(data["timesteps"]))


def test_the_softmax_mass_is_the_kernels_own_read(tmp_path, monkeypatch):
    """Cross-check through the attention kernel itself: SDPA with one-hot key-group VALUES returns, per query, the
    attention mass of each key group; its query-group means must equal the probe's aggregates."""

    model = _tiny_model(window=WINDOW)
    probe = AttentionProbe(model)
    reference = {}
    original = CausalSoftmaxAttention._run_attention

    def run_attention(self, q, k, v):
        if probe.active:
            info = probe._chunk
            key_m, query_m = probe._matrices(q.device)
            onehot = key_m[None, :, None, :].expand(1, -1, q.shape[2], -1).transpose(1, 2).to(q.dtype)   # [1, H, S, G]
            mass = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), onehot)[0]          # [H, T, G]
            reference[(probe._position, self.probe_layer, info["chunk"])] = torch.einsum("htg,tq->hqg", mass, query_m)
        return original(self, q, k, v)

    monkeypatch.setattr(CausalSoftmaxAttention, "_run_attention", run_attention)
    _, paths = _episode(model, probe, tmp_path)
    checked = 0
    for (position, layer, chunk), expected in reference.items():
        data = np.load(paths[chunk - 1])   # the dumps start at chunk 1
        slot = probe.softmax_layers.index(layer)
        got = data["softmax_mass"][position[0], position[1], slot]
        np.testing.assert_allclose(got, expected.numpy(), rtol=1e-5, atol=1e-6)
        checked += 1
    assert checked == CHUNKS * 3 * 2 * len(probe.softmax_layers)


def test_the_gdn_parts_add_up_to_the_layers_readout(tmp_path):
    """past + window readout of the probe = the GDN layer's ``num / den`` (the o_norm input), per query-group mean."""

    model = _tiny_model(window=WINDOW)
    probe = AttentionProbe(model)
    captured = {}
    handles = []
    for index, block in enumerate(model.blocks):
        if isinstance(block.attn, CausalGDNAttention):
            def hook(module, args, index=index):
                if probe.active:
                    out = args[0]                                   # [B, H, Dv, T] (permuted before o_norm)
                    norms = out[0].double().norm(dim=1)             # [H, T]
                    _, query_m = probe._matrices(out.device)
                    captured[(probe._position, index, probe._chunk["chunk"])] = (norms @ query_m.double()).float()
            handles.append(block.attn.o_norm.register_forward_pre_hook(hook))
    try:
        _, paths = _episode(model, probe, tmp_path)
    finally:
        for handle in handles:
            handle.remove()
    assert captured
    for (position, layer, chunk), expected in captured.items():
        got = np.load(paths[chunk - 1])["gdn"][position[0], position[1], probe.gdn_layers.index(layer)][..., 3]
        np.testing.assert_allclose(got, expected.numpy(), rtol=1e-4, atol=1e-6)


def test_a_probe_refuses_an_unexpected_layout_and_incomplete_chunks(tmp_path):
    model = _tiny_model(window=WINDOW)
    probe = AttentionProbe(model)
    with pytest.raises(RuntimeError, match="no chunk to write"):
        probe.write(str(tmp_path), 1)
    probe.begin_chunk(chunk=0, read_ids=[], origin=0, tokens_per_frame=12, latent_frames_per_chunk=1, chunk_actions=32,
                      robot_memory=True, steps=2, streams=2)
    probe.set_forward(0, 0, 1000.0, 1000.0)
    with pytest.raises(RuntimeError, match="layout expects"):
        probe.softmax(1, torch.zeros(1, 10, 2, 32), torch.zeros(1, 10, 2, 32), torch.zeros(1, 10, 2, 32))
    probe.pause()
    with pytest.raises(RuntimeError, match="forwards seen"):
        probe.write(str(tmp_path), 1)
    AttentionProbe.uninstall(model)
    assert all(block.attn.probe is None for block in model.blocks)


def test_the_probe_computes_in_fp32_under_the_generations_bf16_autocast(tmp_path):
    """On CUDA the generation runs under bf16 autocast; the probe must neither inherit it (a bf16 matmul, and
    ``.numpy()`` refuses bf16) nor differ from its autocast-free result."""

    model = _tiny_model(window=WINDOW)
    probe = AttentionProbe(model)
    g = torch.Generator().manual_seed(3)
    q = torch.randn(1, 57, 2, 32, generator=g)
    k = torch.randn(1, 12 + 45 + 57, 2, 32, generator=g)      # the sink, chunk 0 (video_and_action), the window
    v = torch.randn(1, 12 + 45 + 57, 2, 32, generator=g)
    gdn = [torch.randn(1, 2, 57, 32, generator=g), torch.randn(1, 2, 32, 32, generator=g), torch.randn(1, 2, 32, 32, generator=g),
           torch.rand(1, 2, 57, 32, generator=g), torch.rand(1, 2, 32, generator=g), torch.rand(1, 2, 32, generator=g)]
    results = []
    for autocast in (False, True):
        probe.begin_chunk(chunk=1, read_ids=[0], origin=0, tokens_per_frame=12, latent_frames_per_chunk=1, chunk_actions=32,
                          robot_memory=True, steps=1, streams=1)
        probe.set_forward(0, 0, 1000.0, 1000.0)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
            probe.softmax(probe.softmax_layers[0], q, k, v)
            probe.gdn(probe.gdn_layers[0], *gdn, 1e-8)
        probe.pause()
        data = np.load(probe.write(str(tmp_path / str(autocast)), 1))
        results.append((data["softmax_mass"], data["softmax_out"], data["gdn"]))
    for plain, autocast in zip(*results):
        np.testing.assert_array_equal(plain, autocast)
