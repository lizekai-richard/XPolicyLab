"""Opt-in diagnostics of how much a chunk's generation reads the memory (deploy key ``attention_dump_dir``).

The probe watches every forward of ``CausalPolicySession.generate_chunk`` (each denoising step, each text stream) and
aggregates, per chunk, what the layers read; it never changes a forward (it only reads the tensors the layer already
computed) and the commit forwards are not watched.

* Softmax layers: the attention weights ``softmax(q k^T / sqrt(D))`` the kernel uses (same q / k after q/k-norm, RoPE
  and the memory concatenation, no mask), summed over the keys of each KEY GROUP and averaged over the queries of each
  QUERY GROUP. Key groups: the sink (the episode's first observation, ``sink.video``), every committed chunk read (its
  target latents, and under video_and_action its state row and its action rows) and the window's own observation /
  targets / state / actions. Also the norm of the attention output split into the part read from the memory keys (the
  sink included) and the part read from the window keys.
* GDN layers (no per-key weights): the readout ``(q S + q K^T B V) / (q_den z + q_den K_den^T beta)`` split into the past
  (the decayed state of the committed chunks) and this window: the past's share of the normalizer (the linear-attention
  analogue of the softmax mass on the memory) and the norms of the two parts of the output.

Query groups (the window's tokens): ``obs``, ``target``, ``state``, ``action`` (no ``obs`` for a model without chunk
observations; ``meta["query_groups"]`` names them). One ``.npz`` per chunk, written by the adapter:
``ep{E:03d}_chunk{C:03d}.npz`` with the arrays below plus a JSON ``meta`` string (labels, sizes, read plan).
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np
import torch

QUERY_GROUPS = ("obs", "target", "state", "action")
SOFTMAX_OUT_PARTS = ("memory", "window", "total")
GDN_PARTS = ("past_share_of_normalizer", "past_out_norm", "window_out_norm", "total_out_norm")


def _group_matrix(sizes: Sequence[int], device: torch.device) -> torch.Tensor:
    """One-hot ``[sum(sizes), len(sizes)]`` (fp32) mapping each token to its group."""

    total = int(sum(sizes))
    ids = torch.repeat_interleave(torch.arange(len(sizes), device=device), torch.as_tensor(list(sizes), device=device))
    out = torch.zeros(total, len(sizes), device=device, dtype=torch.float32)
    out[torch.arange(total, device=device), ids] = 1.0
    return out


class AttentionProbe:
    """Per-chunk aggregates of the memory read; ``install`` it on a ``CausalPolicyModel``, the session drives it."""

    def __init__(self, model) -> None:
        if getattr(model, "causal_observation", False):
            raise NotImplementedError(
                "attention_dump_dir: the probe does not model a causal_observation chunk's observation prefill"
            )
        if any(getattr(block.attn, "output_normalizer", True) is False for block in model.blocks):
            raise NotImplementedError(
                "attention_dump_dir: the probe's GDN parts are shares of the readout's denominator, which a numerator-only "
                "checkpoint (model.memory_gdn_output_normalizer false) does not divide by"
            )
        self.kinds = [block.attn_type for block in model.blocks]
        self.softmax_layers = [i for i, kind in enumerate(self.kinds) if kind != "GatedDeltaNet"]
        self.gdn_layers = [i for i, kind in enumerate(self.kinds) if kind == "GatedDeltaNet"]
        self._sm_slot = {layer: n for n, layer in enumerate(self.softmax_layers)}
        self._gdn_slot = {layer: n for n, layer in enumerate(self.gdn_layers)}
        for index, block in enumerate(model.blocks):
            block.attn.probe = self
            block.attn.probe_layer = index
        self.active = False
        self._chunk: Optional[dict] = None
        self._position: tuple[int, int] = (0, 0)

    @staticmethod
    def uninstall(model) -> None:
        for block in model.blocks:
            block.attn.probe = None

    # -- driven by the session -----------------------------------------------------------------------------------------

    def begin_chunk(
        self,
        *,
        chunk: int,
        read_ids: Sequence[int],
        origin: int,
        tokens_per_frame: int,
        latent_frames_per_chunk: int,
        chunk_actions: int,
        robot_memory: bool,
        steps: int,
        streams: int,
        sink: bool = True,
        observation: bool = True,
    ) -> None:
        """Lay out the key / query groups of chunk ``chunk``'s denoising window and allocate its aggregates."""

        video_entry = int(latent_frames_per_chunk) * int(tokens_per_frame)
        key_labels, key_sizes = [], []
        if sink:
            key_labels.append("sink.video")
            key_sizes.append(int(tokens_per_frame))
        for chunk_id in read_ids:
            key_labels.append(f"mem{chunk_id}.video")
            key_sizes.append(video_entry)
            if robot_memory:
                key_labels += [f"mem{chunk_id}.state", f"mem{chunk_id}.action"]
                key_sizes += [1, int(chunk_actions)]
        query_groups = list(QUERY_GROUPS if observation else QUERY_GROUPS[1:])
        query_sizes = ([int(tokens_per_frame)] if observation else []) + [video_entry, 1, int(chunk_actions)]
        key_labels += [f"cur.{name}" for name in query_groups]
        key_sizes += query_sizes
        self._chunk = {
            "chunk": int(chunk),
            "read_ids": [int(i) for i in read_ids],
            "origin": int(origin),
            "robot_memory": bool(robot_memory),
            "sink": bool(sink),
            "query_groups": query_groups,
            "key_labels": key_labels,
            "key_sizes": key_sizes,
            "query_sizes": query_sizes,
            "memory_keys": int(sum(key_sizes) - sum(query_sizes)),
            "steps": int(steps),
            "streams": int(streams),
            "softmax_heads": None,
            "gdn_heads": None,
            "softmax_mass": None,
            "softmax_out": None,
            "gdn": None,
            "timesteps": np.full((int(steps), 2), np.nan, dtype=np.float32),
            "seen": np.zeros((int(steps), int(streams)), dtype=np.int32),
        }
        self._key_matrix = None
        self._query_matrix = None

    def set_forward(self, step: int, stream: int, video_t: float, action_t: float) -> None:
        self._position = (int(step), int(stream))
        self._chunk["timesteps"][int(step)] = (float(video_t), float(action_t))
        self._chunk["seen"][int(step), int(stream)] += 1
        self.active = True

    def pause(self) -> None:
        self.active = False

    # -- called from the layers -----------------------------------------------------------------------------------------

    def _matrices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._key_matrix is None or self._key_matrix.device != device:
            self._key_matrix = _group_matrix(self._chunk["key_sizes"], device)
            query = _group_matrix(self._chunk["query_sizes"], device)
            self._query_matrix = query / query.sum(dim=0, keepdim=True)   # column means
        return self._key_matrix, self._query_matrix

    @torch.no_grad()
    def softmax(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """``q`` [1, T, H, D] (this window), ``k`` / ``v`` [1, M + T, H, D] (memory entries then the window), exactly
        what the attention kernel receives."""

        info = self._chunk
        tokens, total = q.shape[1], k.shape[1]
        if tokens != sum(info["query_sizes"]) or total != sum(info["key_sizes"]):
            raise RuntimeError(
                f"attention probe: layer {layer} has {tokens} queries / {total} keys, the layout expects "
                f"{sum(info['query_sizes'])} / {sum(info['key_sizes'])}"
            )
        heads = q.shape[2]
        if info["softmax_mass"] is None:
            info["softmax_heads"] = heads
            shape = (info["steps"], info["streams"], len(self.softmax_layers), heads, len(info["query_groups"]))
            info["softmax_mass"] = np.zeros(shape + (len(info["key_sizes"]),), dtype=np.float32)
            info["softmax_out"] = np.zeros(shape + (len(SOFTMAX_OUT_PARTS),), dtype=np.float32)
        with torch.autocast(device_type=q.device.type, enabled=False):   # the generation runs under bf16 autocast
            mass, out = self._softmax_aggregates(q, k, v)
        step, stream = self._position
        slot = self._sm_slot[layer]
        info["softmax_mass"][step, stream, slot] = mass.float().cpu().numpy()
        info["softmax_out"][step, stream, slot] = out.float().cpu().numpy()

    def _softmax_aggregates(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        info = self._chunk
        key_m, query_m = self._matrices(q.device)
        qh = q[0].permute(1, 0, 2).float()                     # [H, T, D]
        kh = k[0].permute(1, 0, 2).float()                     # [H, S, D]
        vh = v[0].permute(1, 0, 2).float()
        weights = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) * (qh.shape[-1] ** -0.5), dim=-1)   # [H, T, S]
        mass = torch.matmul(torch.matmul(weights, key_m).transpose(-1, -2), query_m).transpose(-1, -2)    # [H, Q, G]
        memory = info["memory_keys"]
        out_memory = torch.matmul(weights[..., :memory], vh[:, :memory])
        out_window = torch.matmul(weights[..., memory:], vh[:, memory:])
        norms = torch.stack(
            (out_memory.norm(dim=-1), out_window.norm(dim=-1), (out_memory + out_window).norm(dim=-1)), dim=-1
        )                                                       # [H, T, 3]
        return mass, torch.einsum("htp,tq->hqp", norms, query_m)

    @torch.no_grad()
    def gdn(
        self,
        layer: int,
        q: torch.Tensor,
        past_state: torch.Tensor,
        window_state: torch.Tensor,
        q_den: torch.Tensor,
        past_z: torch.Tensor,
        window_z: torch.Tensor,
        eps: float,
    ) -> None:
        """The GDN readout's two parts: ``q`` / ``q_den`` [1, H, T, D], states [1, H, D, D], normalizers [1, H, D]."""

        info = self._chunk
        heads = q.shape[1]
        if info["gdn"] is None:
            info["gdn_heads"] = heads
            info["gdn"] = np.zeros(
                (info["steps"], info["streams"], len(self.gdn_layers), heads, len(info["query_groups"]), len(GDN_PARTS)),
                dtype=np.float32,
            )
        with torch.autocast(device_type=q.device.type, enabled=False):   # the generation runs under bf16 autocast
            parts = self._gdn_parts(q, past_state, window_state, q_den, past_z, window_z, eps)
        step, stream = self._position
        info["gdn"][step, stream, self._gdn_slot[layer]] = parts.float().cpu().numpy()

    def _gdn_parts(self, q, past_state, window_state, q_den, past_z, window_z, eps) -> torch.Tensor:
        _, query_m = self._matrices(q.device)
        q64 = q[0].double()
        den_past = torch.einsum("htd,hd->ht", q_den[0].double(), past_z[0].double())
        den_window = torch.einsum("htd,hd->ht", q_den[0].double(), window_z[0].double())
        den = den_past + den_window + eps
        num_past = torch.einsum("htd,hdv->htv", q64, past_state[0].double()) / den[..., None]
        num_window = torch.einsum("htd,hdv->htv", q64, window_state[0].double()) / den[..., None]
        parts = torch.stack(
            (
                den_past / den,
                num_past.norm(dim=-1),
                num_window.norm(dim=-1),
                (num_past + num_window).norm(dim=-1),
            ),
            dim=-1,
        ).float()                                               # [H, T, 4]
        return torch.einsum("htp,tq->hqp", parts, query_m)

    # -- written by the adapter -----------------------------------------------------------------------------------------

    def write(self, directory: str, episode: int, extra: Optional[dict] = None) -> str:
        """Save the finished chunk as ``ep{E:03d}_chunk{C:03d}.npz``; returns the path."""

        info = self._chunk
        if info is None:
            raise RuntimeError("attention probe: no chunk to write")
        if not bool((info["seen"] == 1).all()):
            raise RuntimeError(f"attention probe: forwards seen per (step, stream) {info['seen'].tolist()}, expected 1 each")
        meta = {
            key: info[key]
            for key in ("chunk", "read_ids", "origin", "robot_memory", "sink", "query_groups", "key_labels", "key_sizes",
                        "query_sizes", "memory_keys", "steps", "streams", "softmax_heads", "gdn_heads")
        }
        meta.update(
            episode=int(episode),
            softmax_out_parts=list(SOFTMAX_OUT_PARTS),
            gdn_parts=list(GDN_PARTS),
            softmax_layers=self.softmax_layers,
            gdn_layers=self.gdn_layers,
            streams_order=["cond", "uncond"][: info["streams"]],
        )
        meta.update(extra or {})
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"ep{int(episode):03d}_chunk{info['chunk']:03d}.npz")
        arrays = {
            "softmax_mass": info["softmax_mass"],
            "softmax_out": info["softmax_out"],
            "gdn": info["gdn"],
            "timesteps": info["timesteps"],
        }
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, meta=np.asarray(json.dumps(meta)), **arrays)
        os.replace(tmp, path)
        self._chunk = None
        return path


__all__ = ["AttentionProbe", "GDN_PARTS", "QUERY_GROUPS", "SOFTMAX_OUT_PARTS"]
