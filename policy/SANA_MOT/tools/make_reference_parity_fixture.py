#!/usr/bin/env python
"""Record the MoT forward of a PAST Sana rwm/mot revision on a tiny model, as a parity fixture for the mirror.

Two revisions trained checkpoints that the branch tip no longer reproduces verbatim, so their outputs are recorded
once here and ``tests/test_mot_reference_parity.py`` compares the mirror against them:

* ``606e48dd9`` -- the LEGACY text layout (``action_dit.context_mlp``, the video expert's own ``y_embedder`` +
  ``attention_y_norm``), canvas-only. Checkpoints of runs launched before the ContextEmbedder refactor carry it (the
  NSC canvas run's epoch-5 snapshot ``mot_canvas_nsc_s15k``). Cases: the base text layout and G = 2 token groups.
* ``20c4d63d9`` -- the ``context_embedder`` layout while the OpenWAM canvas was still the ONLY video layout (8fd95e219
  .. 33b220373). Runs launched in that window have canvas yamls without ``model.extra.video_layout`` -- the NSC f25
  canvas runs at step 36,250 (``logits/sft_robodojo_mot_jointabs_f25_openwam_nsc`` and ``..._state_as_context_true``)
  among them. Under state-as-context the state token joins BOTH experts' text (one shared key mask). Cases: the base
  text layout and the G = 1 token group, batch 2 with per-sample text masks.

Per state-conditioning mode the fixture holds the tiny model's weights (rounded to fp32 so the file stays small; the
reference forward ran on exactly those values in fp64), the inputs, and the fp64 outputs.

Usage (any environment with torch; the checkout only needs its ``dev`` and ``diffusion`` trees):
    git -C <sana checkout> archive <rev> dev diffusion | tar -x -C /tmp/sana_<rev>
    DISABLE_XFORMERS=1 python tools/make_reference_parity_fixture.py --reference <rev> --repo /tmp/sana_<rev> \\
        --out tests/fixtures/<fixture name below>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

DIMS = dict(
    hidden_size=32,
    depth=2,
    num_heads=2,
    input_size=2,
    in_channels=4,
    caption_channels=16,
    model_max_length=6,
    mlp_ratio=2.0,
    linear_head_dim=16,
    softmax_head_dim=16,
    softmax_layer_indices=[1],
    attn_res_block_size=1,
    action_hidden_size=16,
    action_cross_attn_heads=2,
)
F, H, W = 2, 2, 2
MODES = (("state_token", False), ("state_context", True))
REFERENCES = {
    "606e48dd9": dict(fixture="legacy_606e48d_parity.pt", context_layout="legacy_action_mlp", batch=1),
    "20c4d63d9": dict(fixture="canvas_20c4d63d9_parity.pt", context_layout="context_embedder", batch=2),
}


def _build(policy_cls, state_as_context: bool):
    return policy_cls(
        action_hidden_size=DIMS["action_hidden_size"],
        action_cross_attn_heads=DIMS["action_cross_attn_heads"],
        action_attn_res_block_size=DIMS["attn_res_block_size"],
        action_state_as_context=state_as_context,
        video_donor=None,
        config=SimpleNamespace(work_dir=None, model=SimpleNamespace(extra={"action_dim": 80}), vae=SimpleNamespace(vae_stride=(8, 32, 32))),
        input_size=DIMS["input_size"],
        patch_size=(1, 1, 1),
        in_channels=DIMS["in_channels"],
        hidden_size=DIMS["hidden_size"],
        depth=DIMS["depth"],
        num_heads=DIMS["num_heads"],
        mlp_ratio=DIMS["mlp_ratio"],
        caption_channels=DIMS["caption_channels"],
        model_max_length=DIMS["model_max_length"],
        qk_norm=True,
        cross_norm=True,
        y_norm=True,
        class_dropout_prob=0.1,
        linear_attn_type="GatedDeltaNet",
        softmax_attn_type="GatedSoftmaxAttention",
        softmax_layer_indices=DIMS["softmax_layer_indices"],
        ffn_type="SwiGLU",
        use_pe=True,
        pos_embed_type="wan_rope",
        linear_head_dim=DIMS["linear_head_dim"],
        softmax_head_dim=DIMS["softmax_head_dim"],
        use_attn_res=True,
        attn_res_block_size=DIMS["attn_res_block_size"],
        use_time_conditioning=False,
        pred_sigma=False,
    ).double()


def _data_info(torch, batch: int, g):
    steps = (F - 1) * 8
    data_info = {
        "model_fps": 25.0,
        "view_count": 1,
        "action80": torch.randn(batch, steps, 80, dtype=torch.float64, generator=g),
        "action_mask80": torch.ones(batch, steps, 80, dtype=torch.bool),
        "action_timestep": torch.full((batch, steps), 700.0, dtype=torch.float64),
        "initial_state80": torch.randn(batch, 80, dtype=torch.float64, generator=g),
        "initial_state_condition_mask80": torch.ones(batch, 80, dtype=torch.bool),
    }
    data_info["action_mask80"][:, :, 60:] = False
    return data_info


def _legacy_606e48dd9(torch, model, state_as_context: bool):
    """The recorded 606e48dd9 fixture, draw for draw (regenerating it must reproduce the committed file)."""

    assert hasattr(model.action_dit, "context_mlp") and not hasattr(model, "context_embedder"), "not the legacy layout"
    with torch.no_grad():
        torch.nn.init.normal_(model.action_dit.action_head.linear.weight, std=0.05)
        torch.nn.init.normal_(model.action_dit.action_head.linear.bias, std=0.05)
        for expert in (model.video_dit, model.action_dit):
            for proj in (expert.attn_res.attn_proj, expert.attn_res.mlp_proj, expert.attn_res.final_proj):
                torch.nn.init.normal_(proj.weight, std=0.5)
        for embedder in [model.action_dit.action_embed] + ([model.action_dit.state_context_embed] if state_as_context else [model.action_dit.state_embed]):
            torch.nn.init.normal_(embedder.proj.weight, std=0.05)
        for param in model.parameters():
            param.copy_(param.float().double())
    model.eval()

    batch = REFERENCES["606e48dd9"]["batch"]
    g = torch.Generator().manual_seed(7)
    x = torch.randn(batch, DIMS["in_channels"], F, H, W, dtype=torch.float64, generator=g)
    timestep = torch.zeros(batch, 1, F, dtype=torch.float64)
    timestep[:, :, 1:] = 700.0
    data_info = _data_info(torch, batch, g)
    y = torch.randn(batch, 1, DIMS["model_max_length"], DIMS["caption_channels"], dtype=torch.float64, generator=g)
    mask = torch.ones(batch, 1, 1, DIMS["model_max_length"], dtype=torch.float64)
    mask[..., -2:] = 0
    y_groups = torch.stack((torch.randn_like(y, dtype=torch.float64), y), dim=1)
    mask_groups = torch.stack((mask, mask), dim=1)
    return x, timestep, data_info, (("base", y, mask), ("token_groups_g2", y_groups, mask_groups))


def _canvas_20c4d63d9(torch, model, state_as_context: bool):
    """20c4d63d9: ContextEmbedder, canvas-only. Every zero-initialized projection gets random values, the GDN beta
    projections and output-gate biases included, so no path of the forward collapses to its init."""

    assert hasattr(model, "context_embedder") and not hasattr(model.action_dit, "context_mlp"), "not the context_embedder layout"
    assert not hasattr(model, "video_layout"), "this revision already has the video_layout switch (use a pre-33b220373 one)"
    assert (model.context_embedder.state_proj is not None) == state_as_context
    with torch.no_grad():
        torch.nn.init.normal_(model.action_dit.action_head.linear.weight, std=0.05)
        torch.nn.init.normal_(model.action_dit.action_head.linear.bias, std=0.05)
        for expert in (model.video_dit, model.action_dit):
            for proj in (expert.attn_res.attn_proj, expert.attn_res.mlp_proj, expert.attn_res.final_proj):
                torch.nn.init.normal_(proj.weight, std=0.5)
        state_module = model.context_embedder.state_proj if state_as_context else model.action_dit.state_embed
        for embedder in (model.action_dit.action_embed, state_module):
            torch.nn.init.normal_(embedder.proj.weight, std=0.05)
            torch.nn.init.normal_(embedder.proj.bias, std=0.05)
        for name, param in model.named_parameters():
            if name.endswith("beta_proj.weight") or name.endswith("output_gate.bias"):
                torch.nn.init.normal_(param, std=0.05)
        for param in model.parameters():
            param.copy_(param.float().double())
    model.eval()

    batch = REFERENCES["20c4d63d9"]["batch"]
    length, channels = DIMS["model_max_length"], DIMS["caption_channels"]
    g = torch.Generator().manual_seed(11)
    x = torch.randn(batch, DIMS["in_channels"], F, H, W, dtype=torch.float64, generator=g)
    timestep = torch.zeros(batch, 1, F, dtype=torch.float64)
    timestep[:, :, 1:] = 700.0
    data_info = _data_info(torch, batch, g)
    y = torch.randn(batch, 1, length, channels, dtype=torch.float64, generator=g)
    mask = torch.ones(batch, 1, 1, length, dtype=torch.float64)
    mask[0, ..., -2:] = 0
    mask[1, ..., -1:] = 0
    y_group = torch.randn(batch, 1, 1, length, channels, dtype=torch.float64, generator=g)
    mask_group = torch.ones(batch, 1, 1, 1, length, dtype=torch.float64)
    mask_group[0, ..., -3:] = 0
    return x, timestep, data_info, (("base", y, mask), ("token_group_g1", y_group, mask_group))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, choices=sorted(REFERENCES))
    parser.add_argument("--repo", required=True, help="a Sana checkout (or source copy) at that rwm/mot revision")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ.setdefault("DISABLE_XFORMERS", "1")
    repo = str(Path(args.repo).resolve())
    sys.path.insert(0, repo)
    import torch

    from dev.rwm.diffusion.model.nets import sana_qwennext_mot_policy as module

    assert str(Path(module.__file__).resolve()).startswith(repo), module.__file__
    policy_cls = module.SanaRWMMoTAttnResPolicy
    spec = REFERENCES[args.reference]
    make_case = _legacy_606e48dd9 if args.reference == "606e48dd9" else _canvas_20c4d63d9

    record = {
        "meta": {
            "source": f"Sana rwm/mot @ {args.reference}",
            "context_layout": spec["context_layout"],
            "dims": dict(DIMS),
            "shape": (spec["batch"], F, H, W),
        },
        "modes": {},
    }
    for name, state_as_context in MODES:
        torch.manual_seed(20260918)
        model = _build(policy_cls, state_as_context)
        x, timestep, data_info, text_cases = make_case(torch, model, state_as_context)

        cases = []
        for case_name, text, text_mask in text_cases:
            with torch.no_grad():
                out = model(x, timestep, text, mask=text_mask, data_info=data_info)
            cases.append(
                {
                    "name": case_name,
                    "y": text.clone(),
                    "mask": text_mask.clone(),
                    "x_out": out["x"].clone(),
                    "action_pred": out["action_pred"].clone(),
                }
            )
            print(f"{name:13s} {case_name:16s} |x| {out['x'].abs().mean():.4f} |action| {out['action_pred'].abs().mean():.4f}")
        record["modes"][name] = {
            "state_as_context": state_as_context,
            "state_dict": {k: v.float().clone() for k, v in model.state_dict().items()},
            "x": x,
            "timestep": timestep,
            "data_info": data_info,
            "cases": cases,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, out)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
