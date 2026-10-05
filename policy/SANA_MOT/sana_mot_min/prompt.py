"""The MoT policy's language conditioning, in every video layout.

Ports of ``build_token_group_prompts`` (``dev/rwm/diffusion/data/token_group_prompts.py``) as the MoT datasets call it:

* multiview (``multiview: sana_latent``): G = V + 1 rows -- one per camera view in view order (``Embodiment Type`` /
  ``Action Mode`` / ``Observation View`` / ``Instruction`` joined by ``"\\n"``), then the robot row without the
  ``Observation View`` line. The video expert reads the view rows, the action expert the robot row.
* canvas (``openwam`` / ``sana_pixel``): the composite view's row -- ONE row read by both experts (before 2026-09-21,
  and since 2026-09-22 through the shared caption embedder), or that row plus the robot row (G = 2, 2026-09-21..22) --
  with the era's layout descriptor (``sana_wam_min.openwam_canvas`` / ``sana_pixel_canvas``).

The ``Action Mode`` sentence follows the checkpoint's action mode and target modes (``sana_wam_min.text``; the MoT
RoboDojo joint lines train ``absolute`` joint targets, the eefabs line EEF-only absolute EEF targets). The CFG
unconditional rows drop the ``Instruction`` line, which is what training-time instruction dropout produced.
"""

from __future__ import annotations

from typing import Optional, Sequence

from sana_wam_min.openwam_canvas import render_canvas_prompt_rows as render_shared_canvas_rows
from sana_wam_min.robot80 import JOINT_TARGET_ABSOLUTE, JOINT_TARGET_ANCHOR_DELTA, validate_joint_target_mode
from sana_wam_min.robodojo_io import ROBODOJO_VIEW_ORDER
from sana_wam_min.text import ROBODOJO_EMBODIMENT, VIEW_PROMPT_TEXT, prompt_field

COMPOSITE_VIEW_DESCRIPTOR = "openwam_composite"
COMPOSITE_VIEW_TEXT = "a composite view combining the head camera above the left and right wrist cameras"

# ``_ACTION_MODE_TEXT[<joint_target_mode>][JOINT_ONLY]``; the field renderer appends the period.
ACTION_MODE_TEXT = {
    JOINT_TARGET_ABSOLUTE: (
        "joint_only; joint and gripper targets are absolute future targets "
        "and no end-effector action is supervised"
    ),
    JOINT_TARGET_ANCHOR_DELTA: (
        "joint_only; joint motion is relative to the first state, gripper "
        "targets are absolute future targets, and no end-effector action is "
        "supervised"
    ),
}


def render_canvas_prompt_rows(
    instruction: str,
    include_instruction: bool = True,
    joint_target_mode: str = JOINT_TARGET_ABSOLUTE,
    embodiment: str = ROBODOJO_EMBODIMENT,
    view_text: str = COMPOSITE_VIEW_TEXT,
    groups: int = 1,
    action_mode_text: Optional[str] = None,
) -> tuple[str, ...]:
    """Render the canvas rows: the composite view's row (G = 1), or that row and the robot row (G = 2).

    ``action_mode_text`` (the checkpoint's own sentence, ``sana_wam_min.text.action_mode_text``) wins over the joint_only
    sentence of ``joint_target_mode``."""

    action_text = action_mode_text or ACTION_MODE_TEXT[validate_joint_target_mode(joint_target_mode)]
    return render_shared_canvas_rows(
        instruction,
        action_mode_text=action_text,
        include_instruction=include_instruction,
        embodiment=embodiment,
        view_text=view_text,
        groups=groups,
    )


def render_multiview_prompt_rows(
    instruction: str,
    include_instruction: bool = True,
    joint_target_mode: str = JOINT_TARGET_ABSOLUTE,
    embodiment: str = ROBODOJO_EMBODIMENT,
    view_order: Sequence[str] = ROBODOJO_VIEW_ORDER,
    action_mode_text: Optional[str] = None,
) -> tuple[str, ...]:
    """Render the G = V + 1 token-group rows of the multiview layout: one per view in ``view_order``, then the robot row."""

    common = (
        prompt_field("Embodiment Type", embodiment),
        prompt_field("Action Mode", action_mode_text or ACTION_MODE_TEXT[validate_joint_target_mode(joint_target_mode)]),
    )
    instruction_field = (prompt_field("Instruction", instruction),) if include_instruction else ()
    view_rows = tuple(
        "\n".join((*common, prompt_field("Observation View", VIEW_PROMPT_TEXT[view]), *instruction_field))
        for view in view_order
    )
    return (*view_rows, "\n".join((*common, *instruction_field)))


def canvas_prompt_payload(instruction: str, joint_target_mode: str = JOINT_TARGET_ABSOLUTE) -> dict[str, tuple[str, ...]]:
    """Return ``{"conditional": rows, "unconditional": rows}`` as the canvas dataset ships them."""

    return {
        "conditional": render_canvas_prompt_rows(instruction, include_instruction=True, joint_target_mode=joint_target_mode),
        "unconditional": render_canvas_prompt_rows(instruction, include_instruction=False, joint_target_mode=joint_target_mode),
    }


def unconditional_rows_from_conditional(rows: Sequence[str]) -> tuple[str, ...]:
    """Drop the last (``Instruction:``) line of each rendered row."""

    return tuple("\n".join(row.split("\n")[:-1]) for row in rows)


__all__ = [
    "ACTION_MODE_TEXT",
    "COMPOSITE_VIEW_DESCRIPTOR",
    "COMPOSITE_VIEW_TEXT",
    "canvas_prompt_payload",
    "render_canvas_prompt_rows",
    "render_multiview_prompt_rows",
    "unconditional_rows_from_conditional",
]
