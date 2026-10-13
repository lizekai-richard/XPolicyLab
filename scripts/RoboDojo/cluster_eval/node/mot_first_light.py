#!/usr/bin/env python
"""First light of a SANA MoT checkpoint through the XPolicyLab SANA_MOT adapter (no simulator).

Builds the adapter's MoTInferenceSession from an assembled checkpoint snapshot (config.yaml +
model/pytorch_model_fsdp.bin + normalization/*.json), i.e. the real load path of the policy server:
train-yaml -> MoTConfig -> bf16 model -> STRICT weight load -> LTX-2.3 VAE -> Gemma text encoder.
Then runs one `predict` on synthetic RGB frames and reports the chunk it produced.

Env / args:
  --ckpt        checkpoint snapshot dir (required)
  --xpl-root    XPolicyLab checkout holding policy/SANA_MOT and policy/SANA_WAM (default $CE_REMOTE_XPL, else ~/xpl_zekai/XPolicyLab)
  --text-encoder / --vae   encoder paths (default: $SANA_REPO/output/pretrained_models/...)
  --device cuda|cpu        default cuda when available
  --steps N                Euler steps for the probe (default 10, the evaluated setting)
  --cfg / --action-cfg     CFG scales (default 6 / 1, the best-scoring SANA setting)
  --skip-forward           load only, no sampling
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--xpl-root", default=os.environ.get("CE_REMOTE_XPL", os.path.expanduser("~/xpl_zekai/XPolicyLab")))
    parser.add_argument("--text-encoder", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--action-cfg", type=float, default=1.0)
    parser.add_argument("--flow-shift", type=float, default=3.5)
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("DISABLE_XFORMERS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    adapter_dir = os.path.join(args.xpl_root, "policy", "SANA_MOT")
    if not os.path.isdir(adapter_dir):
        print(f"FATAL: no SANA_MOT adapter at {adapter_dir}")
        return 2
    sys.path.insert(0, adapter_dir)

    import torch

    from sana_mot_min.config import load_train_config, mot_config_from_train_config
    from sana_mot_min.session import MoTInferenceSession, find_train_config

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sana_repo = os.environ.get("SANA_REPO", os.path.expanduser("~/Sana"))
    text_encoder = args.text_encoder or os.path.join(sana_repo, "output/pretrained_models/text_encoder/gemma-2-2b-it")
    vae = args.vae or os.path.join(sana_repo, "output/pretrained_models/LTX-2.3-Diffusers")
    print(f"ckpt        : {args.ckpt}")
    print(f"adapter     : {adapter_dir}")
    print(f"device      : {device}   torch {torch.__version__}")
    print(f"text encoder: {text_encoder}")
    print(f"vae         : {vae}")

    cfg_path = find_train_config(args.ckpt)
    train_cfg = load_train_config(str(cfg_path))
    from sana_mot_min.checkpoint import read_mot_state_dict
    from sana_mot_min.mot_model.checkpoint import detect_context_layout

    layout = detect_context_layout(read_mot_state_dict(args.ckpt)[0])
    mot_cfg = mot_config_from_train_config(train_cfg, context_layout=layout)
    print(f"\n== architecture resolved from {cfg_path} ==")
    print(f"  factory            : {train_cfg['model']['model']}   (video layout / text layout: resolved at load from the weights)")
    print(f"  video trunk        : depth {mot_cfg.video.depth} hidden {mot_cfg.video.hidden_size} heads {mot_cfg.video.num_heads}"
          f" softmax layers {mot_cfg.video.softmax_layer_indices}")
    print(f"  action expert      : hidden {mot_cfg.action_hidden_size} cross-heads {mot_cfg.action_cross_attn_heads}"
          f" mlp {mot_cfg.action_mlp_ratio} attn_res {mot_cfg.action_attn_res_block_size}")
    print(f"  state_as_context   : {mot_cfg.action_state_as_context}")
    print(f"  joint_target_mode  : {train_cfg['data']['extra']['joint_target_mode']}")

    t0 = time.perf_counter()
    session = MoTInferenceSession.from_paths(
        checkpoint_dir=args.ckpt,
        text_encoder_path=text_encoder,
        vae_path=vae,
        device=device,
        steps=args.steps,
        cfg_scale=args.cfg,
        action_cfg_scale=args.action_cfg,
        flow_shift=args.flow_shift,
    )
    load_s = time.perf_counter() - t0
    report = session.load_report
    model = session.model
    total = sum(p.numel() for p in model.parameters())
    video = sum(p.numel() for n, p in model.named_parameters() if n.startswith("video_dit.") or ".video_block." in n)
    print(f"\n== load OK in {load_s:.1f} s ==")
    print(f"  tensors total/loaded : {report['tensors_total']} / {report['tensors_loaded']}   stripped {report['stripped']}")
    print(f"  softmax layers       : {report['softmax_layer_indices']}   state_as_context {report['state_as_context']}")
    print(f"  checkpoint layout    : text {report.get('context_layout')}   video {report.get('video_layout')}")
    print(f"  parameters           : {total/1e9:.3f} B total = video {video/1e9:.3f} B + action {(total-video)/1e9:.3f} B  ({model.dtype})")
    print(f"  normalization        : sha256 {session.normalization.sha256}")
    print(f"     joint_target_mode {session.normalization.joint_target_mode}  num_frames {session.normalization.num_frames}"
          f"  model_fps {session.normalization.model_fps}")
    print(f"  sampling             : steps {session.steps} cfg {session.cfg_scale} video_cfg {session.video_cfg_scale}"
          f" action_cfg {session.action_cfg_scale} flow_shift {session.flow_shift}")
    if device == "cuda":
        print(f"  gpu memory           : {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated,"
              f" {torch.cuda.max_memory_allocated()/2**30:.2f} GiB peak")

    if args.skip_forward:
        print("\nFIRST LIGHT (load only) OK")
        return 0

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(3)]
    state80 = np.zeros(80, dtype=np.float32)
    state80[0:6] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    state80[29:35] = [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6]
    state80[16] = state80[45] = 0.0
    mask80 = np.zeros(80, dtype=bool)
    mask80[list(range(0, 6)) + [16] + list(range(29, 35)) + [45]] = True
    instruction = "Stack the three bowls together."

    for run in ("cold", "warm"):
        t0 = time.perf_counter()
        result = session.predict(frames, state80, mask80, instruction, torch.Generator(device=session.device).manual_seed(20260802))
        dt = time.perf_counter() - t0
        rows = result.action80_raw_absolute
        joints = rows[:, list(range(0, 6)) + list(range(29, 35))]
        print(f"\n== predict ({run}) in {dt:.2f} s at {session.steps} steps ==")
        print(f"  action80 rows        : {tuple(rows.shape)}   finite {bool(rows.isfinite().all())}")
        print(f"  joint targets        : min {joints.min():+.4f} max {joints.max():+.4f} mean |step| "
              f"{(joints[1:] - joints[:-1]).abs().mean():.5f} rad")
        print(f"  gripper closedness   : left {rows[:, 16].min():.3f}..{rows[:, 16].max():.3f}"
              f"  right {rows[:, 45].min():.3f}..{rows[:, 45].max():.3f}")
        print(f"  video latent         : {tuple(result.video_latent.shape)}")
        print(f"  row 0 left arm       : {[round(float(v), 4) for v in rows[0, 0:6]]}")
        print(f"  row 23 left arm      : {[round(float(v), 4) for v in rows[23, 0:6]]}")
        print(f"  receipt              : { {k: v for k, v in result.receipt.items() if k not in ('checkpoint',)} }")
    if device == "cuda":
        print(f"\n  gpu peak after sampling: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("\nFIRST LIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
