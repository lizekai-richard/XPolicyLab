"""Condition-causal self-attention of the memory-less policy: ``[obs; state] | [prediction latents; actions]``.

Inference mirror of Sana ``dev/rwm/diffusion/model/condition_chunks.py`` (rwm/zekai-merge f6a257d4a, 8e0639b16; the
policy class swaps every block's self-attention for it, no switch). Chunk 0 holds the clean observation latent
(latent frame 0 of every view) and the state row (the frame-0 latents alone under ``state_cross_attention``); chunk 1
holds every later latent frame and the action rows. Each chunk attends to itself bidirectionally and to the earlier
chunk, so the condition never reads the noisy prediction.

The model keeps its view-major token order; every self-attention layer gathers its inputs into chunk-major order,
runs the chunk-causal computation and scatters the result back. Every operation below is the live one in the same
order and precision (copied, inference only), so a forward is bit-identical to the live class's.

``ConditionPass`` / ``ConditionPrefix`` mirror the live prefix cache (Sana 89eaa78f6): the condition never reads the
prediction and its timestep is 0, so a decision can run the condition chunk once (``PREFILL``) and every denoising
step the prediction chunk alone (``TARGET``) against the cached reads; up to rounding it equals the full forward.
"""

from __future__ import annotations

import torch

CONDITION_CHUNK = 0
PREDICTION_CHUNK = 1
PREFILL, TARGET = "prefill", "target"
# the GatedDeltaNet recurrence tensors a condition-causal checkpoint carries per linear-attention block
GDN_RECURRENCE_KEYS = ("gate_proj.weight", "gate_proj.bias", "A_log", "dt_bias", "recall_gate")
DT_BIAS_INIT = -5.0
A_LOG_INIT = 0.0


class ConditionChunkLayout:
    """Chunk ids of the model's token order with the stable chunk-major permutation (``order``) and its inverse."""

    def __init__(self, chunk_ids: torch.Tensor) -> None:
        cpu_ids = chunk_ids.detach().to("cpu", torch.long)
        sizes = torch.bincount(cpu_ids, minlength=PREDICTION_CHUNK + 1)
        if sizes.numel() != PREDICTION_CHUNK + 1 or bool((sizes == 0).any()):
            raise ValueError(f"the condition layout needs tokens in both chunks, got chunk sizes {sizes.tolist()}")
        order = torch.argsort(cpu_ids, stable=True)
        self.num_tokens = int(cpu_ids.numel())
        self.bounds = (0, *torch.cumsum(sizes, 0).tolist())
        self.order_host = order
        self.order = order.to(chunk_ids.device)
        self.restore = torch.argsort(order).to(chunk_ids.device)

    def slices(self) -> list[tuple[int, int]]:
        return list(zip(self.bounds[:-1], self.bounds[1:]))


class ConditionPrefix:
    """The condition chunk's reads of one decision and caption stream, per self-attention layer: a softmax layer's
    keys and values, a GatedDeltaNet layer's state after the condition's write (and its denominator state)."""

    def __init__(self) -> None:
        self._layers = {}

    def store(self, module, value) -> None:
        self._layers[id(module)] = value

    def load(self, module):
        try:
            return self._layers[id(module)]
        except KeyError as error:
            raise RuntimeError("a prediction pass reads the condition's prefill over the same layers first") from error

    def __len__(self) -> int:
        return len(self._layers)


class ConditionPass(ConditionChunkLayout):
    """The tokens of one prefix-cache pass: the condition chunk's (``PREFILL``) or the prediction chunk's (``TARGET``)
    in the model's token order, one chunk to the layers, with the cache they write or read. ``keep`` indexes the
    pass's tokens in the full layout (ascending; ``keep_host`` its host copy)."""

    def __init__(self, layout: ConditionChunkLayout, phase: str, cache: ConditionPrefix) -> None:
        if phase not in (PREFILL, TARGET):
            raise ValueError(f"a condition pass is {PREFILL!r} or {TARGET!r}, got {phase!r}")
        chunk = CONDITION_CHUNK if phase == PREFILL else PREDICTION_CHUNK
        lo, hi = layout.bounds[chunk], layout.bounds[chunk + 1]
        self.keep = layout.order[lo:hi]
        self.keep_host = layout.order_host[lo:hi]
        self.full_tokens = layout.num_tokens
        self.num_tokens = hi - lo
        self.bounds = (0, self.num_tokens)
        self.order = torch.arange(self.num_tokens, device=self.keep.device)
        self.restore = self.order
        self.phase = phase
        self.cache = cache

    def kept(self, offset: int, length: int) -> tuple[int, int]:
        """The pass's (offset, length) of the full layout's token span [offset, offset + length)."""
        start = int((self.keep_host < int(offset)).sum())
        return start, int((self.keep_host < int(offset) + int(length)).sum()) - start


def condition_chunk_layout(
    frame_ids: torch.Tensor, action_rows: int, state_row: bool = True, device=None
) -> ConditionChunkLayout:
    """Layout of video tokens with latent frame ids ``frame_ids`` (model token order) followed by the robot rows
    ``[state, actions]`` (``[actions]`` when ``state_row`` is False: ``state_cross_attention``)."""

    video = (torch.as_tensor(frame_ids, dtype=torch.long) > 0).long()
    robot = torch.full((int(state_row) + int(action_rows),), PREDICTION_CHUNK, dtype=torch.long)
    if state_row:
        robot[0] = CONDITION_CHUNK
    return ConditionChunkLayout(torch.cat((video.cpu(), robot)).to(device))


def require_condition_layout(chunk_layout) -> ConditionChunkLayout:
    if not isinstance(chunk_layout, ConditionChunkLayout):
        raise TypeError(
            "condition-causal attention needs the policy's ConditionChunkLayout, "
            f"got {type(chunk_layout).__name__}"
        )
    return chunk_layout


def chunk_causal_attention(attention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: ConditionChunkLayout):
    """Run ``attention`` on ``[B, N, heads, dim]`` inputs in chunk-major order; chunk c reads the key prefix of
    chunks <= c. Returns the output in the model's token order."""

    if q.shape[1] != layout.num_tokens:
        raise ValueError(f"attention has {q.shape[1]} tokens, the condition layout covers {layout.num_tokens}")
    q, k, v = (tensor.index_select(1, layout.order) for tensor in (q, k, v))
    out = torch.cat([attention(q[:, lo:hi], k[:, :hi], v[:, :hi]) for lo, hi in layout.slices()], dim=1)
    return out.index_select(1, layout.restore)


def chunk_means(x: torch.Tensor, slices: list[tuple[int, int]]) -> torch.Tensor:
    """Per-chunk token means ``[B, M, C]`` of chunk-major ``[B, N, C]`` tokens (the live ``_ChunkMeans`` forward)."""

    return torch.stack([x[:, lo:hi].mean(dim=1) for lo, hi in slices], dim=1)


def gated_delta_condition_pass(
    q_num, k_num, v, q_den, k_den, beta, decay, eps: float, normalizer: bool = True, state: tuple | None = None
):
    """One chunk of :func:`gated_delta_chunk_scan` on its own: the condition chunk (``state`` None), returning its
    output and the state it writes, or the prediction chunk reading that ``state`` decayed by its own ``decay``
    (``[B, H, 1]``), returning its output. Same operations and precision as the scan's chunk."""

    dtype = torch.float64 if q_num.dtype == torch.float64 else torch.float32
    if not normalizer:
        q_den = k_den = None
    q_num, k_num, v, q_den, k_den, beta, decay = (
        None if tensor is None else tensor.to(dtype) for tensor in (q_num, k_num, v, q_den, k_den, beta, decay)
    )
    scale = (q_num.shape[-1] * q_num.shape[2]) ** -0.5
    k_c = k_num * scale
    beta_c = beta[..., None]
    read = (k_c * beta_c).transpose(-1, -2) @ v
    read_z = None
    if normalizer:
        read_z = (k_den * scale * beta_c).sum(dim=2)
    if state is None:
        written = (read, read_z)
    else:
        gate = decay[:, :, 0, None]
        read = read + state[0] * gate[..., None]
        if normalizer:
            read_z = read_z + state[1] * gate
    numerator = q_num @ read
    out = numerator / ((q_den @ read_z[..., None]) + eps) if normalizer else numerator
    return (out, written) if state is None else out


def gated_delta_chunk_scan(
    q_num, k_num, v, q_den, k_den, beta, decay, slices: list[tuple[int, int]], eps: float, normalizer: bool = True
) -> torch.Tensor:
    """Gated delta-rule scan over contiguous chunks (rwm/yuyang ``gated_delta_chunk_scan``, Sana condition_chunks).

    Inputs are chunk-major ``[B, H, N, D]`` (RoPE on the numerator path, ReLU on the denominator path), ``beta`` is
    ``[B, H, N]`` and ``decay`` ``[B, H, M]``. Chunk c reads ``q (g_c S + K_c^T diag(beta) V_c)``, divided by
    ``q_den (g_c z + K_den_c^T beta) + eps`` when ``normalizer`` is set, then writes
    ``S' = g_c S + K_c^T diag(beta) (V_c - K_c g_c S)`` (z likewise towards 1). Keys are scaled by
    ``1 / sqrt(D * |chunk|)``; the last chunk's write is never read. Computes in float32 (float64 for float64 inputs).
    """

    dtype = torch.float64 if q_num.dtype == torch.float64 else torch.float32
    if not normalizer:
        q_den = k_den = None
    q_num, k_num, v, q_den, k_den, beta, decay = (
        None if tensor is None else tensor.to(dtype) for tensor in (q_num, k_num, v, q_den, k_den, beta, decay)
    )
    dim = q_num.shape[-1]
    total = q_num.shape[2]
    state = state_z = None
    outputs = []
    for chunk, (lo, hi) in enumerate(slices):
        scale = (dim * (hi - lo)) ** -0.5
        k_c = k_num[:, :, lo:hi] * scale
        beta_c = beta[:, :, lo:hi, None]
        v_c = v[:, :, lo:hi]
        read = (k_c * beta_c).transpose(-1, -2) @ v_c
        if normalizer:
            k_den_c = k_den[:, :, lo:hi] * scale
            read_z = (k_den_c * beta_c).sum(dim=2)
        if state is not None:
            gate = decay[:, :, chunk, None]
            state = state * gate[..., None]
            read = read + state
            if normalizer:
                state_z = state_z * gate
                read_z = read_z + state_z
        numerator = q_num[:, :, lo:hi] @ read
        if normalizer:
            outputs.append(numerator / ((q_den[:, :, lo:hi] @ read_z[..., None]) + eps))
        else:
            outputs.append(numerator)
        if hi == total:
            break
        if state is None:
            state = read
            if normalizer:
                state_z = read_z
        else:
            correction = (v_c - k_c @ state) * beta_c
            state = state + k_c.transpose(-1, -2) @ correction
            if normalizer:
                z_correction = (1.0 - (k_den_c @ state_z[..., None])) * beta_c
                state_z = state_z + (k_den_c.transpose(-1, -2) @ z_correction)[..., 0]
    return torch.cat(outputs, dim=2)


__all__ = [
    "A_LOG_INIT",
    "CONDITION_CHUNK",
    "ConditionChunkLayout",
    "ConditionPass",
    "ConditionPrefix",
    "DT_BIAS_INIT",
    "GDN_RECURRENCE_KEYS",
    "PREDICTION_CHUNK",
    "PREFILL",
    "TARGET",
    "chunk_causal_attention",
    "chunk_means",
    "condition_chunk_layout",
    "gated_delta_chunk_scan",
    "gated_delta_condition_pass",
    "require_condition_layout",
]
