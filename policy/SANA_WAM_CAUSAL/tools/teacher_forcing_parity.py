"""Real-weight teacher-forcing parity: Sana's two-stream TRAINING forward vs the adapter's KV-cache inference, on the
episode canvas in its single form (no chunk observations; the live tree must be at or after Sana 4bd6eca83). A
``causal_observation`` checkpoint (Sana 764ee2113, causal_state afc9874b6; a live tree at or after them) runs with every
chunk's own observation latent -- its start frame encoded alone -- on both sides: [o_c | targets] on the canvas, the
session's observation prefill before the chunk's window on the chain.

The same checkpoint and the same inputs go through two implementations of one prediction:

(A) Sana's chunk-causal policy class, built and loaded exactly as Sana's causal validator does
    (``model_video_init_config`` + ``build_model``, ``pos_embed`` dropped), ONE forward over the training canvas
    [sink | clean targets 0..L-2 | noisy targets 0..L-1] with ``two_stream_num_chunks = L`` -- the trainer's assembly,
    the sink (the first frame) and its StreamingLLM position, the softmax sliding window and ``memory_content`` rows all;
(B) the adapter's causal inference: the sink committed (and a padded window's frozen chunk 0), then per emitted chunk
    k (from 1 after a padded window's chunk 0, from 0 on a first-frame window) the denoising window [noisy targets_k]
    against the KV cache (the forward ``generate_chunk`` runs at every sampler step, built with the session's own
    helpers) followed by ``commit_chunk`` of clean chunk k.

Both get the same caption (Gemma), the same latents from the adapter's own front-end (the per-view causal VAE streams
of the episode: the sink, a padded window's frozen first chunk and every executed chunk's targets), clean actions /
states and noisy chunks (flow mixtures at fixed per-chunk video and action timesteps). Per emitted noisy chunk the video
velocity and the action velocity (EEF-only slots) are compared. Runs (A) and (B) in fp32 (TF32 off: the implementation check, expect ~1e-5 relative) and in bf16
(the deployment precision), and (A) bf16 against (A) fp32 as the bf16 noise floor of the training forward;
``--precisions float64`` (attention and the GDN rotary product in fp64 too) separates rounding from logic. As a
by-product it reports sink_gap, the first frame encoded alone vs the stream's sink (expected 0).

Inputs are VAE encodes of smooth synthetic camera frames and smooth normalized EEF trajectories (no simulator needed);
``--frames-from`` takes recorded RGB frames instead (an .npz with ``frames`` [T, 3, 480, 640, 3] uint8, the three
cameras head / left wrist / right wrist).

Usage (on a GPU node, from policy/SANA_WAM_CAUSAL, PYTHONPATH=<XPolicyLab parent>):
  python tools/teacher_forcing_parity.py --checkpoint-dir DIR --sana-repo SANA_TREE --text-encoder-path P --vae-path P
         [--chunks 30] [--allow-donor-checkpoint] [--out DIR]
  # fp64 (the encoders stay on the GPU; CUDA's fp64 attention returns outputs the shared cross-attention cannot view):
  OMP_NUM_THREADS=32 python tools/teacher_forcing_parity.py ... --chunks 8 --precisions float64 --model-device cpu
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR))

from sana_wam_causal import shared  # noqa: E402,F401
from sana_wam_causal.checkpoint import load_causal_weights  # noqa: E402
from sana_wam_causal.contract import bidirectional_view, resolve_causal_contract  # noqa: E402
from sana_wam_causal.model import build_causal_policy  # noqa: E402
from sana_wam_causal.runtime import CausalRuntime  # noqa: E402
from sana_wam_causal.session import Caption, CausalPolicySession  # noqa: E402
from sana_wam_min.config import load_train_config, policy_config_from_train_config  # noqa: E402
from sana_wam_min.session import find_train_config  # noqa: E402

EEF_SLOTS = list(range(7, 17)) + list(range(36, 46))


# ---- flash_attn stand-in for the live tree (the nodes carry no flash_attn) ------------------------------------------------

def _sdpa_varlen(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0,
                 softmax_scale=None, causal=False, **_):
    """flash_attn_varlen_func on packed (T, H, D): one (query span, key span) pair at a time, torch SDPA."""

    assert dropout_p == 0.0 and not causal
    q_bounds = [int(b) for b in cu_seqlens_q.tolist()]
    k_bounds = [int(b) for b in cu_seqlens_k.tolist()]
    outs = []
    for (q_lo, q_hi), (k_lo, k_hi) in zip(zip(q_bounds, q_bounds[1:]), zip(k_bounds, k_bounds[1:])):
        qs = q[q_lo:q_hi].transpose(0, 1).unsqueeze(0)
        ks = k[k_lo:k_hi].transpose(0, 1).unsqueeze(0)
        vs = v[k_lo:k_hi].transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(qs, ks, vs, scale=softmax_scale)
        outs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _sdpa_dense(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale,
                                         is_causal=causal)
    return out.transpose(1, 2)


def import_live_tree(sana_repo: str):
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.setdefault(key, "0" if key != "WORLD_SIZE" else "1")
    sys.path.insert(0, sana_repo)
    from diffusion.model.builder import build_model  # noqa: F401
    from diffusion.utils.config import SanaVideoConfig, model_video_init_config  # noqa: F401
    import dev.rwm.diffusion.model.nets.sana_qwennext_policy_causal  # noqa: F401  (registry)
    import dev.rwm.diffusion.model.utils.causal_rwm_train as crwm
    if not hasattr(crwm, "refuse_chunk_observation"):
        raise SystemExit(f"{sana_repo} predates the single form without chunk observations (Sana 4bd6eca83)")
    module = types.ModuleType("flash_attn")   # bound only after the import: diffusers probes the real package's spec
    module.flash_attn_varlen_func = _sdpa_varlen
    module.flash_attn_func = _sdpa_dense
    sys.modules["flash_attn"] = module
    return crwm


def build_live(sana_repo: str, ckpt_dir: str, dtype: torch.dtype, device):
    import pyrallis
    from diffusion.model.builder import build_model
    from diffusion.utils.config import SanaVideoConfig, model_video_init_config

    config = pyrallis.parse(config_class=SanaVideoConfig, config_path=os.path.join(ckpt_dir, "config.yaml"), args=[])
    config.work_dir = None   # the frozen yaml's cluster work_dir: the model would open its train log there
    model = build_model(config.model.model, **model_video_init_config(config))
    state = torch.load(os.path.join(ckpt_dir, "model", "pytorch_model_fsdp.bin"), map_location="cpu", weights_only=True, mmap=True)
    state = dict(state.get("state_dict", state))
    state.pop("pos_embed", None)  # the trainer / validator load drops it
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"live load: unexpected keys {result.unexpected_keys[:8]}")
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, result.missing_keys


def reference_forms(model) -> list[str]:
    """The live training forward in its reference forms: the recipe's training-only speed switches off (Sana 02aebea6e
    gdn_fused_two_stream, c4d1d246d attn_res_fast_attend / attn_res_compiled_sum: Triton / cached forms of the same
    math, the GDN one with bf16 matmuls) and the two-stream softmax through Sana's per-segment ``_two_stream_spans``,
    which the FlexAttention call equals up to summation order (its compiled kernel at head dim 256 needs 139 KB of shared
    memory, more than an sm_86 board has). Returns what was changed."""

    changed = []
    if getattr(model, "gdn_fused_two_stream", False):
        model.gdn_fused_two_stream = False
        changed.append("gdn_fused_two_stream")
    for name in ("fast_training_attend", "compiled_source_sum"):
        if getattr(model.attn_res, name, False):
            setattr(model.attn_res, name, False)
            changed.append(f"attn_res.{name}")

    def spans(self, q, k, v, bounds, num_clean, window=None, sink=0, sink_phases=None, observation_spans=()):
        return self._two_stream_spans(q, k, v, bounds, num_clean, window=window, sink=sink, sink_phases=sink_phases,
                                      observation_spans=observation_spans)

    for block in model.blocks:
        attn = block.attn
        if getattr(attn, "fused_two_stream", False):
            attn.fused_two_stream = False
        if hasattr(attn, "_two_stream_attention") and hasattr(attn, "_two_stream_spans"):
            attn._two_stream_attention = types.MethodType(spans, attn)
            if "two_stream_spans" not in changed:
                changed.append("two_stream_spans")
    return changed


def gdn_rope_in_float64() -> None:
    """fp64 runs: both implementations rotate the GDN numerator in complex64 (Sana ``chunk_causal_gdn._rope_bhnd``, the
    port's ``layers.rope_bhnd_complex64``), i.e. at fp32 precision whatever the activations; at real sizes a last-bit
    difference then crosses fp32 rounding boundaries and cascades through the layers (~1e-6). Lift the product to
    complex128 over complex64-rounded phases on BOTH sides, so an fp64 comparison measures the implementations."""

    import dev.rwm.diffusion.model.layers.chunk_causal_gdn as live_gdn
    import sana_wam_causal.layers as port_layers

    def rope(x, freqs, _fallback):
        if x.dtype != torch.float64:
            return _fallback(x, freqs)
        rotated = torch.view_as_complex(x.contiguous().unflatten(-1, (-1, 2)))
        phases = (freqs.to(torch.complex64) if freqs.is_complex() else freqs.float()).to(torch.complex128)
        return torch.view_as_real(rotated * phases).flatten(-2).type_as(x)

    for module, name in ((live_gdn, "_rope_bhnd"), (port_layers, "rope_bhnd_complex64")):
        original = getattr(module, name)
        setattr(module, name, lambda x, freqs, _o=original: rope(x, freqs, _o))


def attention_in_working_precision(model) -> None:
    """fp64 runs: the recipe's fp32_attention would round the softmax attention of both sides to fp32; run it in fp64
    (as the CPU parity tests do), so an fp64 comparison measures the implementations, not that rounding."""

    for block in model.blocks:
        block.attn.fp32_attention = False


# ---- inputs -----------------------------------------------------------------------------------------------------------------

def smooth_frames(seed: int, ticks) -> dict:
    """{tick: [3 x (480, 640, 3) uint8]}: per camera a low-frequency colour field drifting with time (a moving blob)."""

    g = torch.Generator().manual_seed(seed)
    fields = torch.rand(3, 3, 12, 16, generator=g)
    blobs = torch.rand(3, 2, generator=g)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 480), torch.linspace(0, 1, 640), indexing="ij")
    out = {}
    for t in ticks:
        out[t] = [None] * 3
        for cam in range(3):
            phase = 0.004 * t * (cam + 1)
            field = torch.roll(fields[cam], shifts=int(phase * 16) % 16, dims=-1)
            img = F.interpolate(field[None], size=(480, 640), mode="bilinear", align_corners=False)[0]
            cy, cx = (blobs[cam] + torch.tensor([0.3 * math.sin(phase * 7), 0.3 * math.cos(phase * 5)])) % 1.0
            blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 0.01)
            img = (0.75 * img + 0.25 * blob[None]).clamp(0, 1)
            out[t][cam] = (img.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    return out


def smooth_rows(seed: int, chunks: int, chunk: int) -> torch.Tensor:
    """[chunks * chunk + 1, 80] normalized rows: a smooth random walk on the EEF-only slots, zeros elsewhere."""

    g = torch.Generator().manual_seed(seed + 1)
    steps = chunks * chunk + 1
    walk = torch.cumsum(0.03 * torch.randn(steps, len(EEF_SLOTS), generator=g, dtype=torch.float64), dim=0)
    walk = torch.tanh(walk + 0.3 * torch.randn(1, len(EEF_SLOTS), generator=g, dtype=torch.float64))
    rows = torch.zeros(steps, 80, dtype=torch.float64)
    rows[:, EEF_SLOTS] = walk
    return rows


# ---- (A) the training forward -------------------------------------------------------------------------------------------------

def two_stream(live, crwm, inp, L, f_c, chunk, memory_content, first, autocast_dtype):
    sink, clean, noisy = inp["sink"], inp["clean"], inp["noisy"]
    dev = next(live.parameters()).device
    views, slots = inp["view_shapes"], inp["view_slots"]
    width = sum(h * w for h, w in views)
    # causal_observation: noisy chunk c is [o_c | targets], o_c clean (Sana noisy_chunk_frames)
    observations = inp.get("observations")
    lead = int(observations is not None)
    noisy = [torch.cat([observations[c], noisy[c]], dim=2) if lead else noisy[c] for c in range(L)]
    x = torch.cat([sink] + clean[: L - 1] + noisy, dim=2)
    strip = x.reshape(*x.shape[:3], 1, width)
    frames = x.shape[2]
    noisy_start = 1 + (L - 1) * f_c
    t_video = torch.zeros(1, frames, dtype=torch.float32)
    for c in range(L):
        lo = noisy_start + c * (lead + f_c) + lead
        t_video[:, lo : lo + f_c] = inp["t_video"][c]
    memory = crwm.memory_row_chunks(L, memory_content)
    rows = torch.cat(inp["a_clean"][:memory] + inp["a_noisy"][:L], dim=1)
    row_t = torch.cat([torch.zeros(1, memory * chunk)] + [torch.full((1, chunk), inp["t_action"][c]) for c in range(L)], dim=1)
    anchors = inp["states"][:memory] + inp["states"][:L]
    grid = len(anchors)
    mask_rows = inp["action_mask"][:, :1].expand(1, grid * chunk, 80)
    info = {
        "model_fps": 25.0,
        "view_count": torch.tensor([len(views)]),
        "view_latent_shape": torch.tensor([[list(v) for v in views]]),
        "video_frame_stride": torch.tensor([inp["stride"]]),
        "view_slot_ids": torch.tensor(list(slots)),
        "rwm_task": "policy",
        "action80": crwm.insert_state_slots(rows, grid, chunk),
        "action_mask80": crwm.insert_state_slots(mask_rows, grid, chunk),
        "action_timestep": crwm.insert_state_slots(row_t, grid, chunk),
        "initial_state80": anchors[0],
        "initial_state_condition_mask80": inp["state_mask"],
        "two_stream_num_chunks": L,
    }
    if grid > 1:
        info["chunk_states80"] = torch.stack(anchors[1:], dim=1)
        info["chunk_states_mask80"] = inp["state_mask"][:, None].expand(1, grid - 1, 80)
    info = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in info.items()}
    dtype = next(live.parameters()).dtype
    with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
        out = live(strip.to(dev, dtype), t_video.to(dev), inp["y"].to(dev, dtype), mask=inp["mask"].to(dev), data_info=info)
    positions = crwm.action_row_positions(grid, chunk)
    preds = []
    for c in range(first, L):
        lo = noisy_start + c * (lead + f_c) + lead
        v = out["x"][:, :, lo : lo + f_c].reshape(1, -1, f_c, *clean[0].shape[-2:])
        a = out["action_pred"].index_select(1, positions[(memory + c) * chunk : (memory + c + 1) * chunk].to(dev))
        preds.append((v.float().cpu(), a.float().cpu()))
    return preds


# ---- (B) the adapter's KV-cache inference --------------------------------------------------------------------------------------

def kv_chain(mirror, inp, L, window, first, autocast_dtype):
    dev = next(mirror.parameters()).device
    dtype = next(mirror.parameters()).dtype
    chunk = mirror.chunk_actions
    strip = len(inp["view_shapes"]) > 1
    session = CausalPolicySession(
        mirror, caption=Caption(inp["y"].to(dev, dtype), inp["mask"].to(dev)), unconditional_caption=None, steps=1,
        flow_shift=5.0, action_flow_shift=1.0, video_cfg_scale=1.0, action_cfg_scale=1.0, model_fps=25.0,
        window=window, view_shapes=inp["view_shapes"] if strip else None, view_slot_ids=inp["view_slots"],
        autocast_dtype=autocast_dtype,
    )
    caption, memory = session.streams[0]
    amask = inp["action_mask"].to(dev)
    smask = inp["state_mask"][0].to(dev)
    f_c = session.f_chunk

    def commit(k):
        session.commit_chunk(
            inp["clean"][k].to(dev, dtype), inp["a_clean"][k].to(dev, dtype), anchor_state=inp["states"][k].to(dev, dtype),
            anchor_state_mask=smask, action_mask=amask,
        )

    session.commit_sink(inp["sink"].to(dev, dtype))
    if first == 1:
        commit(0)   # a padded window's frozen first chunk, memory only
    preds, reads = [], []
    for k in range(first, L):
        video_t = torch.full((1, f_c), float(inp["t_video"][k]), device=dev)
        action_t = torch.full((1, chunk), float(inp["t_action"][k]), device=dev)
        robot = {
            "state80": inp["states"][k].to(dev, dtype), "state_mask80": smask[None],
            "action80": inp["a_noisy"][k].to(dev, dtype), "action_mask80": amask, "action_timesteps": action_t,
        }
        reads.append((session.read_chunk_ids(), session.softmax_origin()))
        with torch.no_grad(), session._autocast(dev):
            if session.observation:   # the chunk's observation prefill (Sana forward_observation), read by its window
                caches = session.prefill_observation(
                    caption, memory, inp["observations"][k].to(dev, dtype), state=robot["state80"],
                    state_mask=robot["state_mask80"],
                )
            else:
                caches = session._caches(memory, commit=False)
            frame_offset, step_offset = session.window_offsets()
            robot, robot_form = session.denoise_robot(robot)
            out = session._forward(
                inp["noisy"][k].to(dev, dtype), video_t, caption, memory, frame_offset=frame_offset,
                step_offset=step_offset, robot=robot, caches=caches, robot_form=robot_form,
            )
        preds.append((out["x"].float().cpu(), out["action_pred"].float().cpu()))
        if k < L - 1:
            commit(k)
    return preds, reads


# ---- comparison ---------------------------------------------------------------------------------------------------------------

def compare(a_preds, b_preds, action_mask):
    rows = []
    mask = action_mask[0, 0].cpu()
    for (va, aa), (vb, ab) in zip(a_preds, b_preds):
        dv, da = vb - va, (ab - aa)[..., mask]
        rows.append({
            "video_rel": float(dv.norm() / va.norm().clamp_min(1e-12)), "video_max": float(dv.abs().max()),
            "action_rel": float(da.norm() / aa[..., mask].norm().clamp_min(1e-12)), "action_max": float(da.abs().max()),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--sana-repo", required=True)
    p.add_argument("--text-encoder-path", required=True)
    p.add_argument("--vae-path", required=True)
    p.add_argument("--chunks", type=int, default=30)
    p.add_argument("--allow-donor-checkpoint", action="store_true",
                   help="an SFT donor dir served as the causal step-0 model (recurrence initialized on both sides)")
    p.add_argument("--instruction", default="Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames-from", default=None)
    p.add_argument("--precisions", default="float32,bfloat16")
    p.add_argument("--model-device", default="cuda",
                   help="device of the two policy models (A and B); cpu serves float64 runs, whose CUDA attention kernels "
                        "return non-contiguous outputs the shared SANA_WAM cross-attention cannot view")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = torch.device("cuda")
    model_device = torch.device(args.model_device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.perf_counter()

    cfg = load_train_config(str(find_train_config(args.checkpoint_dir)))
    contract = resolve_causal_contract(cfg)
    L, f_c, chunk, stride = args.chunks, contract.latent_frames_per_chunk, contract.actions_per_chunk, contract.video_frame_stride
    window = contract.sliding_window_chunks
    first = contract.first_chunk   # 1: a padded window's chunk 0 is frozen (memory only); 0: a first-frame window
    print(f"[tf-parity] contract {contract.describe()} | L={L} window={window}", flush=True)

    # 1. inputs through the adapter's own front-end (canvas, VAE, Gemma prompt)
    runtime = CausalRuntime.from_paths(args.checkpoint_dir, text_encoder_path=args.text_encoder_path, vae_path=args.vae_path,
                                       device=device, steps=10, cfg_scale=6.0, action_cfg_scale=1.0,
                                       allow_donor_checkpoint=args.allow_donor_checkpoint)
    # emitted chunk k (from `first` on) starts at episode tick (k - first) * C; its targets are the frames after it up to
    # the next decision
    ticks = list(range(0, (L - first) * chunk + 1, stride))
    if args.frames_from:
        recorded = np.load(args.frames_from)["frames"]
        if recorded.shape[0] <= ticks[-1]:
            raise ValueError(f"{args.frames_from} holds {recorded.shape[0]} ticks, need {ticks[-1] + 1}")
        frames = {t: list(recorded[t]) for t in ticks}
    else:
        frames = smooth_frames(args.seed, ticks)
    views = {t: runtime.view_frames(frames[t]) for t in ticks}
    streams = runtime.new_streams()
    sink, frozen = streams.start(views[0])
    clean = [] if frozen is None else [frozen.float().cpu()]
    for k in range(first, L):
        per_tick = [views[(k - first) * chunk + t] for t in range(stride, chunk + 1, stride)]
        clean.append(streams.encode_chunk([[v[i] for v in per_tick] for i in range(len(per_tick[0]))]).float().cpu())
    observations = None
    if contract.causal_observation:
        # every chunk's own observation latent: its start frame encoded alone (chunk `first` starts at the first frame,
        # so its observation is the sink's latent; a padded window's frozen chunk 0 takes the sink too)
        observations = [sink.float().cpu()] * first + [
            streams.encode_observation(views[(k - first) * chunk]).float().cpu() for k in range(first, L)
        ]
        obs_gap = float((observations[first] - sink.float().cpu()).abs().max())
        print(f"[tf-parity] causal_observation: {L} observation latents (chunk {first}'s vs the sink: max {obs_gap:.1e})",
              flush=True)
    sink = sink.float().cpu()
    first_frame, _ = runtime.encode_sink(views[0])
    sink_gap = float(((first_frame.float().cpu() - sink) ** 2).mean().sqrt() / (sink ** 2).mean().sqrt().clamp_min(1e-8))
    print(f"[tf-parity] sink_gap (the first frame encoded alone vs the stream's sink, relative RMSE): {sink_gap:.2e}", flush=True)
    view_shapes = streams.view_shapes if runtime.strip else ((int(clean[0].shape[-2]), int(clean[0].shape[-1])),)
    view_slots = runtime.view_slot_ids
    y, mask, _, _ = runtime.frontend.encode_instruction(args.instruction)
    y, mask = y.float().cpu(), mask.cpu()
    del runtime
    gc.collect(); torch.cuda.empty_cache()

    rows = smooth_rows(args.seed, L, chunk).float()
    g = torch.Generator().manual_seed(args.seed + 2)
    sig_v = [0.15 + 0.7 * ((7 * c) % 10) / 9 for c in range(L)]       # a spread of noise levels, per chunk
    sig_a = [0.15 + 0.7 * ((3 * c + 4) % 10) / 9 for c in range(L)]
    action_mask = torch.zeros(1, chunk, 80, dtype=torch.bool)
    action_mask[..., EEF_SLOTS] = True
    state_mask = torch.zeros(1, 80, dtype=torch.bool)
    state_mask[:, EEF_SLOTS] = True
    inp = {
        "sink": sink, "clean": clean, "y": y, "mask": mask, "stride": stride, "action_mask": action_mask, "state_mask": state_mask,
        "view_shapes": view_shapes, "view_slots": view_slots,
        "states": [rows[k * chunk][None] for k in range(L)],
        "a_clean": [rows[k * chunk + 1 : (k + 1) * chunk + 1][None] for k in range(L)],
        "t_video": [1000.0 * s for s in sig_v], "t_action": [1000.0 * s for s in sig_a],
    }
    if observations is not None:
        inp["observations"] = observations
    inp["noisy"] = [(1 - s) * x0 + s * torch.randn(x0.shape, generator=g) for s, x0 in zip(sig_v, clean)]
    inp["a_noisy"] = [((1 - s) * a0 + s * torch.randn(a0.shape, generator=g)).masked_fill(~action_mask, 0)
                      for s, a0 in zip(sig_a, inp["a_clean"])]
    print(f"[tf-parity] inputs ready ({time.perf_counter() - t0:.0f}s): latents {tuple(sink.shape)} per frame, {L} chunks "
          f"({'chunk 0 the frozen opening' if first else 'chunk 0 at the first frame'}), views {view_shapes}", flush=True)

    # 2. (A) and (B) per precision, one model on the GPU at a time
    crwm = import_live_tree(args.sana_repo)
    view = bidirectional_view(cfg)
    if "float64" in args.precisions.split(","):
        gdn_rope_in_float64()
    results = {}
    for prec in [s for s in args.precisions.split(",") if s]:
        dtype = getattr(torch, prec)
        autocast = None if dtype in (torch.float32, torch.float64) else dtype
        live, missing = build_live(args.sana_repo, args.checkpoint_dir, dtype, model_device)
        print(f"[tf-parity] {prec} live reference forms: {reference_forms(live)}", flush=True)
        if dtype == torch.float64:
            attention_in_working_precision(live)
        t1 = time.perf_counter()
        a_preds = two_stream(live, crwm, inp, L, f_c, chunk, contract.memory_content, first, autocast)
        print(f"[tf-parity] {prec} (A) two-stream forward {time.perf_counter() - t1:.1f}s; live missing keys {len(missing)}: {missing[:6]}", flush=True)
        del live
        gc.collect(); torch.cuda.empty_cache()
        mirror = build_causal_policy(policy_config_from_train_config(view), contract, dtype=dtype, device="cpu")
        load_causal_weights(mirror, args.checkpoint_dir, device=model_device,
                            allow_missing_recurrence=args.allow_donor_checkpoint)
        if dtype == torch.float64:
            attention_in_working_precision(mirror)
        t1 = time.perf_counter()
        b_preds, reads = kv_chain(mirror, inp, L, window, first, autocast)
        print(f"[tf-parity] {prec} (B) KV-cache chain {time.perf_counter() - t1:.1f}s", flush=True)
        del mirror
        gc.collect(); torch.cuda.empty_cache()
        results[prec] = {"A": a_preds, "B": b_preds, "reads": reads}

    # 3. report
    report = {}
    for prec, r in results.items():
        report[f"B_vs_A_{prec}"] = compare(r["A"], r["B"], action_mask)
    if "float32" in results and "bfloat16" in results:
        report["A_bf16_vs_A_fp32"] = compare(results["float32"]["A"], results["bfloat16"]["A"], action_mask)
        report["B_bf16_vs_A_fp32"] = compare(results["float32"]["A"], results["bfloat16"]["B"], action_mask)
    reads = next(iter(results.values()))["reads"]
    print("\n[tf-parity] per noisy chunk (relative L2 error = ||X - A|| / ||A||; video velocity | action velocity on the 20 EEF slots)")
    keys = list(report)
    print("chunk  read(origin)      " + "  ".join(f"{k:>26s}" for k in keys))
    for c in range(first, L):
        ids, origin = reads[c - first]
        rd = f"[{ids[0]}..{ids[-1]}]({origin})" if ids else "empty"
        print(f"{c:5d}  {rd:16s} " + "  ".join(f"{report[k][c - first]['video_rel']:12.2e} | {report[k][c - first]['action_rel']:10.2e}" for k in keys))
    print("\n[tf-parity] summary: max relative error over chunks before / after the window boundary (chunk > window)")
    for k in keys:
        pre = [x for c, x in enumerate(report[k], start=first) if window is None or c <= window]
        post = [x for c, x in enumerate(report[k], start=first) if window is not None and c > window]
        fmt = lambda xs, f: f"{max(x[f] for x in xs):.2e}" if xs else "-"
        print(f"  {k:22s} video {fmt(pre, 'video_rel')} / {fmt(post, 'video_rel')}   action {fmt(pre, 'action_rel')} / {fmt(post, 'action_rel')}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        json.dump({"args": vars(args), "reads": reads, "report": report, "sink_gap": sink_gap},
                  open(os.path.join(args.out, "tf_parity.json"), "w"), indent=1)
        torch.save({k: v for k, v in inp.items()}, os.path.join(args.out, "tf_parity_inputs.pt"))
        print(f"[tf-parity] wrote {args.out}")
    print(f"[tf-parity] done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
