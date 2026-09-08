"""CPU tests for sana_wam_min.text: byte-exact prompt rendering and the encoder tensor contract."""

from __future__ import annotations

import glob
import inspect
import json
import os
import sys

import pytest
import torch

ADAPTER_DIR = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/XPolicyLab/policy/SANA_WAM"
if ADAPTER_DIR not in sys.path:
    sys.path.insert(0, ADAPTER_DIR)

from sana_wam_min import text  # noqa: E402

MANIFEST_GLOB = (
    "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/sana_wam_runs/output/"
    "VAL_SFT_RoboDojo_ArxX5_320px_unified_joint_only_holdout_s35000/log_vis/rwm_validation/"
    "policy/joint_only/step_35000/sample_*/manifest.json"
)
GEMMA_DIR = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/models/Sana/text_encoder/gemma-2-2b-it"
REFERENCE_OUT = (
    "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/zekail/xpolicylab_sana_wam_port_20260908/"
    "logs/text_cpu_reference_sample000.pt"
)


def _manifests() -> list[str]:
    return sorted(glob.glob(MANIFEST_GLOB))


def _instruction_from_manifest(manifest: dict) -> str:
    """Instruction as rendered in the manifest's own conditional robot-group row (last group, last line)."""

    robot_row = manifest["prompt"]["conditional"][-1]
    last_line = robot_row.split("\n")[-1]
    assert last_line.startswith("Instruction: ")
    return last_line[len("Instruction: "):]


def test_package_importable_both_ways():
    import XPolicyLab.policy.SANA_WAM.sana_wam_min.text as via_repo

    assert via_repo.ROBODOJO_VIEW_ORDER == text.ROBODOJO_VIEW_ORDER == ("cam_head", "cam_left_wrist", "cam_right_wrist")


def test_prompt_field_normalizes_trailing_periods_and_keeps_quotes():
    assert text.prompt_field("Instruction", ' Arrange the letters to spell "RoboDojo" in a row... ') == (
        'Instruction: Arrange the letters to spell "RoboDojo" in a row.'
    )
    with pytest.raises(ValueError):
        text.prompt_field("Instruction", " ... ")


def test_render_rows_shape_and_uncond_drop():
    cond = text.render_token_group_rows("Stack the three bowls together.")
    uncond = text.render_token_group_rows("Stack the three bowls together.", include_instruction=False)
    assert len(cond) == len(uncond) == 4
    assert all(row.split("\n")[0].startswith("Embodiment Type: ") for row in cond)
    assert all(row.split("\n")[1].startswith("Action Mode: ") for row in cond)
    assert all(row.split("\n")[-1].startswith("Instruction: ") for row in cond)
    assert [len(row.split("\n")) for row in cond] == [4, 4, 4, 3]
    assert text.unconditional_rows_from_conditional(cond) == uncond
    assert text.split_policy_prompt(text.render_policy_prompt("Stack the three bowls together.")) == cond


@pytest.mark.skipif(not _manifests(), reason="validation manifests not present")
def test_render_rows_byte_exact_against_all_validation_manifests():
    paths = _manifests()
    assert len(paths) == 35
    instructions = set()
    for path in paths:
        with open(path) as f:
            manifest = json.load(f)
        assert manifest["view_keys"] == list(text.ROBODOJO_VIEW_ORDER)
        assert manifest["action_mode"] == "joint_only" and manifest["joint_target_mode"] == "anchor_delta"
        instruction = _instruction_from_manifest(manifest)
        instructions.add(instruction)
        payload = text.token_group_prompt_payload(instruction)
        assert list(payload["conditional"]) == manifest["prompt"]["conditional"], path
        assert list(payload["unconditional"]) == manifest["prompt"]["unconditional"], path
        assert list(text.unconditional_rows_from_conditional(manifest["prompt"]["conditional"])) == (
            manifest["prompt"]["unconditional"]
        )
    assert len(instructions) == 35
    assert "Pick up two slices of bread, place them into the toaster, and press the lever down." in instructions


class _StubBatch:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class _StubTokenizer:
    """Deterministic tokenizer: token i of prompt = hash of (prompt, i); length = len(prompt) // 8 + 1 (BOS)."""

    def __init__(self):
        self.calls = []

    def encode(self, prompt: str) -> list[int]:
        return [2] + [7] * (len(prompt) // 8)

    def __call__(self, prompts, max_length, padding, truncation, return_tensors):
        self.calls.append(dict(max_length=max_length, padding=padding, truncation=truncation, return_tensors=return_tensors))
        ids = torch.zeros(len(prompts), max_length, dtype=torch.int64)
        mask = torch.zeros(len(prompts), max_length, dtype=torch.int64)
        for row, prompt in enumerate(prompts):
            n = min(len(self.encode(prompt)), max_length)
            ids[row, 0] = 2
            ids[row, 1:n] = torch.tensor([(hash((prompt, i)) % 1000) + 10 for i in range(1, n)])
            mask[row, :n] = 1
        return _StubBatch(ids, mask)


class _StubEncoder(torch.nn.Module):
    """Deterministic encoder: hidden[b, t, :] = f(input_ids[b, t]) in bf16, returned as a tuple like HF."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1], dtype=torch.float32)[None, :, None]
        chan = torch.arange(self.channels, dtype=torch.float32)[None, None, :]
        hidden = (input_ids.float()[:, :, None] * 0.001 + pos * 0.01 + chan * 0.1) * attention_mask[:, :, None]
        return (hidden.to(torch.bfloat16),)


def test_encode_prompt_rows_contract_with_stubs():
    tokenizer, encoder = _StubTokenizer(), _StubEncoder(channels=16)
    rows = [text.render_token_group_rows("Stack the three bowls together."), text.render_token_group_rows("Fold the clothes neatly.")]
    y, mask = text.encode_prompt_rows(rows, tokenizer, encoder, device="cpu", max_length=64)

    assert tokenizer.calls == [dict(max_length=64, padding="max_length", truncation=True, return_tensors="pt")]
    assert y.shape == (2, 4, 1, 64, 16) and y.dtype == torch.bfloat16
    assert mask.shape == (2, 4, 1, 1, 64) and mask.dtype == torch.int64

    # Row-major flattening: y[b, g] is the encoding of rows[b][g], and the identity selection keeps positions.
    flat = [p for r in rows for p in r]
    tokens = tokenizer(flat, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    ref = encoder(tokens.input_ids, tokens.attention_mask)[0]
    assert torch.equal(y.reshape(8, 64, 16), ref)
    assert torch.equal(mask.reshape(8, 64), tokens.attention_mask)
    assert mask[..., 0].eq(1).all()
    lengths = mask.reshape(8, 64).sum(-1)
    assert (lengths == torch.tensor([len(tokenizer.encode(p)) for p in flat])).all()


def test_encode_prompt_rows_gemma_selection_with_system_prompt():
    """With a chi_prompt the tokenizer pads to a longer length and [0] + last (L-1) is a real re-selection."""

    tokenizer, encoder = _StubTokenizer(), _StubEncoder(channels=8)
    rows = [("a" * 40, "b" * 60)]
    system = "s" * 20
    y, mask = text.encode_prompt_rows(rows, tokenizer, encoder, device="cpu", max_length=30, chi_prompt=[system])
    max_length_all = len(tokenizer.encode(system)) + 30 - 2
    assert tokenizer.calls[0]["max_length"] == max_length_all == 31

    prompts = [system + p for p in rows[0]]
    tokens = tokenizer(prompts, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt")
    ref = encoder(tokens.input_ids, tokens.attention_mask)[0]
    selected = [0] + list(range(-29, 0))
    assert y.shape == (1, 2, 1, 30, 8)
    assert torch.equal(y[0, :, 0], ref[:, selected])
    assert torch.equal(mask[0, :, 0, 0], tokens.attention_mask[:, selected])


def test_encode_conditional_and_unconditional_cfg_gate():
    tokenizer, encoder = _StubTokenizer(), _StubEncoder(channels=8)
    cond = text.render_token_group_rows("Fold the clothes neatly.")
    y, m, yu, mu = text.encode_conditional_and_unconditional(cond, tokenizer, encoder, "cpu", cfg_scale=1.0, max_length=64)
    assert y.shape == (1, 4, 1, 64, 8) and m.shape == (1, 4, 1, 1, 64) and yu is None and mu is None
    y, m, yu, mu = text.encode_conditional_and_unconditional(cond, tokenizer, encoder, "cpu", cfg_scale=6.0, max_length=64)
    assert y.shape == yu.shape == (1, 4, 1, 64, 8) and m.shape == mu.shape == (1, 4, 1, 1, 64)
    assert (mu.sum(-1) < m.sum(-1)).all()


def test_load_text_encoder_signature():
    params = list(inspect.signature(text.load_text_encoder).parameters)
    assert params == ["path", "device"]
    assert text.CAPTION_MAX_LENGTH == 300 and text.CAPTION_CHANNELS == 2304


@pytest.mark.skipif(
    os.environ.get("SANA_WAM_TEXT_SKIP_GEMMA") == "1" or not os.path.isdir(GEMMA_DIR),
    reason="Gemma-2-2B dir missing or SANA_WAM_TEXT_SKIP_GEMMA=1 (CPU load+encode ~30 s, ~5 GB RAM)",
)
def test_gemma_cpu_reference_sample000():
    tokenizer, encoder = text.load_text_encoder(GEMMA_DIR, device="cpu")
    assert tokenizer.padding_side == "right"
    assert type(encoder).__name__ == "Gemma2Model"
    assert not encoder.training
    assert next(encoder.parameters()).dtype == torch.bfloat16

    with open(_manifests()[0]) as f:
        manifest = json.load(f)
    rows = [tuple(manifest["prompt"]["conditional"])]
    y, mask = text.encode_prompt_rows(rows, tokenizer, encoder, device="cpu")
    assert y.shape == (1, 4, 1, 300, 2304) and y.dtype == torch.bfloat16
    assert mask.shape == (1, 4, 1, 1, 300) and mask.dtype == torch.int64
    lengths = mask.reshape(4, 300).sum(-1)
    assert (lengths >= 57).all() and (lengths <= 112).all()
    assert torch.isfinite(y.float()).all()
    os.makedirs(os.path.dirname(REFERENCE_OUT), exist_ok=True)
    torch.save(
        {"y": y, "y_mask": mask, "rows": rows, "manifest": _manifests()[0], "token_lengths": lengths.tolist()},
        REFERENCE_OUT,
    )
