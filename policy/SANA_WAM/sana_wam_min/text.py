"""Language conditioning of the policy: token-group prompt rendering and Gemma encoding.

Ports of ``dev/rwm/diffusion/data/token_group_prompts.py`` (field renderer and
``build_token_group_prompts``), the gemma branch of
``diffusion/model/builder.py::get_tokenizer_and_text_encoder``, and
``dev/rwm/train_utils/text.py::encode_token_group_prompt_rows``. The scene is conditioned
by G = V + 1 independent prompt strings (one per camera view in token-group order, then the
robot/action group); each string is encoded separately and the model routes group g's keys
to token group g. The CFG unconditional row is the same strings minus the trailing
``Instruction:`` line, which is what training-time instruction dropout produced. The
``Action Mode`` sentence depends on the line's action mode and joint / EEF target modes
(``action_mode_text``); the OpenWAM canvas line's single shared row lives in ``openwam_canvas``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

# Frozen RoboDojo policy view order; the token-group axis of ``y`` follows it, then the robot group.
ROBODOJO_VIEW_ORDER = ("cam_head", "cam_left_wrist", "cam_right_wrist")

# Training-side view descriptor texts. ``cam_head`` is described as an overhead external
# camera (not "robot head-mounted camera"); the checkpoint was trained on this wording.
VIEW_PROMPT_TEXT = {
    "cam_head": "overhead external camera",
    "cam_left_wrist": "left wrist-mounted camera",
    "cam_right_wrist": "right wrist-mounted camera",
}

ROBODOJO_EMBODIMENT = "dual-arm RoboDojo ARX-X5 robot with parallel grippers"

# ``_ACTION_MODE_TEXT["anchor_delta"]["joint_only"]``; the field renderer appends the period.
ACTION_MODE_TEXT = (
    "joint_only; joint motion is relative to the first state, gripper "
    "targets are absolute future targets, and no end-effector action is "
    "supervised"
)

# Every Action Mode sentence of ``build_token_group_prompts`` (token_group_prompts.py ``_ACTION_MODE_TEXT`` and
# ``_EEF_ABSOLUTE_ACTION_MODE_TEXT``), keyed by (joint_target_mode, eef_target_mode) then the action mode. The
# training prompt always carries the sentence of ITS line, so the served prompt must be rendered from the
# checkpoint's own modes (``action_mode_text``); qwen_canonical has no absolute-EEF form.
ACTION_MODE_TEXTS = {
    ("anchor_delta", "anchor_delta"): {
        "qwen_canonical": (
            "qwen_canonical; joint and end-effector motion are relative to the "
            "first state, end-effector motion is expressed in the camera-aligned "
            "canonical frame, and gripper targets are absolute future targets"
        ),
        "robot_base_eef": (
            "robot_base_eef; end-effector motion is relative to the first state "
            "and expressed in the robot base frame, joint motion is relative to "
            "the first state, and gripper targets are absolute future targets"
        ),
        "joint_only": ACTION_MODE_TEXT,
    },
    ("absolute", "anchor_delta"): {
        "qwen_canonical": (
            "qwen_canonical; end-effector motion is relative to the first state "
            "and expressed in the camera-aligned canonical frame, while joint "
            "and gripper targets are absolute future targets"
        ),
        "robot_base_eef": (
            "robot_base_eef; end-effector motion is relative to the first state "
            "and expressed in the robot base frame, while joint and gripper "
            "targets are absolute future targets"
        ),
        "joint_only": (
            "joint_only; joint and gripper targets are absolute future targets "
            "and no end-effector action is supervised"
        ),
    },
    ("anchor_delta", "absolute"): {
        "robot_base_eef": (
            "robot_base_eef; end-effector targets are absolute future poses in the "
            "robot base frame, joint motion is relative to the first state, and "
            "gripper targets are absolute future targets"
        ),
        "joint_only": ACTION_MODE_TEXT,
    },
    ("absolute", "absolute"): {
        "robot_base_eef": (
            "robot_base_eef; end-effector targets are absolute future poses in the "
            "robot base frame, and joint and gripper targets are absolute future "
            "targets"
        ),
        "joint_only": (
            "joint_only; joint and gripper targets are absolute future targets "
            "and no end-effector action is supervised"
        ),
    },
}


def action_mode_text(
    action_mode: str = "joint_only",
    joint_target_mode: str = "anchor_delta",
    eef_target_mode: str = "anchor_delta",
) -> str:
    """The Action Mode sentence of a line (without the trailing period the field renderer adds)."""

    try:
        table = ACTION_MODE_TEXTS[(str(joint_target_mode), str(eef_target_mode))]
    except KeyError as error:
        raise ValueError(
            f"unsupported target modes joint={joint_target_mode!r} eef={eef_target_mode!r}"
        ) from error
    try:
        return table[str(action_mode)]
    except KeyError as error:
        raise ValueError(
            f"unsupported action mode {action_mode!r} for joint_target_mode {joint_target_mode!r} / "
            f"eef_target_mode {eef_target_mode!r}"
        ) from error


# Wire encoding of the four groups carried in ``obs.instruction`` by the deploy client.
TOKEN_GROUP_SEPARATOR = "\n\n"

CAPTION_MAX_LENGTH = 300
CAPTION_CHANNELS = 2304


def prompt_field(name: str, value: str) -> str:
    """Render one ``Name: value.`` prompt line byte-identically to the training ``_field``."""

    value = str(value).strip().rstrip(".")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return f"{name}: {value}."


def render_token_group_rows(
    instruction: str,
    include_instruction: bool = True,
    embodiment: str = ROBODOJO_EMBODIMENT,
    action_mode_text: str = ACTION_MODE_TEXT,
    view_order: Sequence[str] = ROBODOJO_VIEW_ORDER,
) -> tuple[str, ...]:
    """Render the G = V + 1 token-group prompt strings for one scene.

    View groups carry ``Embodiment Type`` / ``Action Mode`` / ``Observation View`` /
    ``Instruction`` lines joined by ``"\\n"``; the trailing robot group omits the
    ``Observation View`` line. ``include_instruction=False`` yields the CFG unconditional row
    (every group minus its final ``Instruction`` line).
    """

    common = (
        prompt_field("Embodiment Type", embodiment),
        prompt_field("Action Mode", action_mode_text),
    )
    instruction_field = (prompt_field("Instruction", instruction),) if include_instruction else ()
    view_prompts = tuple(
        "\n".join((*common, prompt_field("Observation View", VIEW_PROMPT_TEXT[view]), *instruction_field))
        for view in view_order
    )
    robot_action_prompt = "\n".join((*common, *instruction_field))
    return (*view_prompts, robot_action_prompt)


def token_group_prompt_payload(instruction: str) -> dict[str, tuple[str, ...]]:
    """Return ``{"conditional": rows, "unconditional": rows}`` as stored in validation manifests."""

    return {
        "conditional": render_token_group_rows(instruction, include_instruction=True),
        "unconditional": render_token_group_rows(instruction, include_instruction=False),
    }


def unconditional_rows_from_conditional(rows: Sequence[str]) -> tuple[str, ...]:
    """Drop the last (``Instruction:``) line of each rendered group, as the deploy session does."""

    return tuple("\n".join(group.split("\n")[:-1]) for group in rows)


def render_policy_prompt(instruction: str, embodiment: str = ROBODOJO_EMBODIMENT) -> str:
    """Render the deploy wire encoding: the conditional groups joined by a blank line."""

    groups = render_token_group_rows(instruction, include_instruction=True, embodiment=embodiment)
    for index, group in enumerate(groups):
        if TOKEN_GROUP_SEPARATOR in group:
            raise ValueError(f"token group {index} contains a blank line; wire encoding is not reversible")
    return TOKEN_GROUP_SEPARATOR.join(groups)


def split_policy_prompt(policy_prompt: str, num_views: int = len(ROBODOJO_VIEW_ORDER)) -> tuple[str, ...]:
    """Recover the ``num_views + 1`` token-group texts from one wire-encoded prompt."""

    groups = tuple(policy_prompt.split(TOKEN_GROUP_SEPARATOR))
    if len(groups) != num_views + 1 or not all(groups):
        raise ValueError(f"policy_prompt must split into {num_views + 1} non-empty groups, got {len(groups)}")
    return groups


def load_text_encoder(path: str, device: str | torch.device = "cuda"):
    """Load the Gemma tokenizer and the bf16 decoder stack (no lm_head) used as the text encoder.

    Mirrors the gemma branch of Sana's ``get_tokenizer_and_text_encoder``: right padding,
    ``AutoModelForCausalLM(...).get_decoder()`` in bf16 with the default (sdpa) attention.
    Returns ``(tokenizer, encoder)`` with the encoder in eval mode.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "right"
    encoder = (
        AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        .get_decoder()
        .to(device)
        .eval()
    )
    return tokenizer, encoder


@torch.no_grad()
def encode_prompt_rows(
    rows: Sequence[Sequence[str]],
    tokenizer,
    encoder,
    device: str | torch.device,
    max_length: int = CAPTION_MAX_LENGTH,
    chi_prompt: Optional[Sequence[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode sample-major ``B x G`` prompt rows into ``y [B,G,1,L,C]`` and ``mask [B,G,1,1,L]``.

    Each of the ``B * G`` strings is a separate tokenizer sequence (BOS added, no EOS,
    right-padded to ``max_length``, truncated). ``y`` is the encoder's last hidden state in
    the encoder dtype; ``mask`` is the int64 attention mask (1 = BOS + text, 0 = pad). The
    gemma ``[0] + last (L-1)`` token selection is applied literally; it is the identity
    unless ``chi_prompt`` prepends a system prompt.
    """

    rows = tuple(tuple(row) for row in rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("prompt rows must form a non-empty B x G grid")
    batch, groups = len(rows), len(rows[0])
    prompts = [prompt for row in rows for prompt in row]
    if chi_prompt:
        system_prompt = "\n".join(chi_prompt)
        prompts = [system_prompt + prompt for prompt in prompts]
        max_length_all = len(tokenizer.encode(system_prompt)) + max_length - 2
    else:
        max_length_all = max_length
    tokens = tokenizer(
        prompts,
        max_length=max_length_all,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    caption = encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
    text_mask = tokens.attention_mask
    selected_tokens = [0] + list(range(-max_length + 1, 0))
    caption = caption[:, selected_tokens]
    text_mask = text_mask[:, selected_tokens]
    caption = caption.reshape(batch, groups, caption.shape[-2], caption.shape[-1])
    text_mask = text_mask.reshape(batch, groups, text_mask.shape[-1])
    return caption.unsqueeze(2), text_mask[:, :, None, None]


def encode_conditional_and_unconditional(
    conditional_rows: Sequence[str],
    tokenizer,
    encoder,
    device: str | torch.device,
    cfg_scale: float = 1.0,
    max_length: int = CAPTION_MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Encode one scene's rows; the unconditional pair is ``None`` unless ``cfg_scale > 1``.

    Under CFG both rows go through one batched encoder call, as the deploy session does,
    and are returned as ``(y_cond, mask_cond, y_uncond, mask_uncond)`` with B = 1 each.
    """

    rows = [tuple(conditional_rows)]
    if cfg_scale > 1:
        rows.append(unconditional_rows_from_conditional(conditional_rows))
    y, mask = encode_prompt_rows(rows, tokenizer, encoder, device, max_length=max_length)
    if cfg_scale > 1:
        return y[:1], mask[:1], y[1:2], mask[1:2]
    return y, mask, None, None
