# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Flow-matching Euler sampler for the joint video + action policy stream.

This module is the policy-task port of Sana's ``sample_robot_world_model``
(``dev/rwm/diffusion/scheduler/robot_world_model_sampler.py``): two independent
``FlowMatchEulerDiscreteScheduler`` instances share one scalar timestep per step,
latent frame 0 is held clean and re-pinned after every step, masked action slots
are zeroed after every step, and the model velocity follows the ``noise - x0``
convention (hence ``scheduler.step(-prediction, ...)`` in the per-token branch).
The progress bar and the timing recorder of the source are dropped; every numeric
line is kept identical.
"""

from __future__ import annotations

import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

# Absolute Robot80 slots of the gripper closedness channels; clamped to [0, 1] after the loop.
LEFT_GRIPPER = 16
RIGHT_GRIPPER = 45

_TASKS = {"forward", "inverse", "policy"}


def task_timesteps(timestep, task):
    """Return the ``(video_timestep, action_timestep)`` pair for one base timestep.

    Args:
        timestep: Base scheduler timestep tensor, or None.
        task: One of "ti2v", "forward", "inverse", "policy".

    Returns:
        Video and action timesteps; the clean stream of the task gets zeros,
        "ti2v" yields ``(timestep, None)`` and a None timestep yields ``(None, None)``.
    """

    if task == "ti2v":
        return timestep, None
    if task not in _TASKS:
        raise ValueError(f"unsupported Robot80 task: {task!r}")
    if timestep is None:
        return None, None
    clean = torch.zeros_like(timestep)
    if task == "forward":
        return timestep, clean
    if task == "inverse":
        return clean, timestep
    return timestep, timestep


def _video_step(scheduler, prediction, timestep, sample, token_timesteps):
    sample_dtype = sample.dtype
    batch, channels, frames, height, width = sample.shape
    sample_tokens = sample.reshape(batch, channels, -1).transpose(1, 2)
    prediction_tokens = prediction.reshape(batch, channels, -1).transpose(1, 2)
    per_token_timesteps = token_timesteps[:, :, None, None].expand(
        batch, frames, height, width
    ).reshape(batch, -1)
    result = scheduler.step(
        -prediction_tokens,
        timestep,
        sample_tokens,
        per_token_timesteps=per_token_timesteps,
        return_dict=False,
    )[0]
    return result.transpose(1, 2).reshape_as(sample).to(sample_dtype)


def _action_step(scheduler, prediction, timestep, sample, token_timesteps):
    return scheduler.step(
        -prediction,
        timestep,
        sample,
        per_token_timesteps=token_timesteps,
        return_dict=False,
    )[0].to(sample.dtype)


def make_scheduler(steps: int, flow_shift: float, device) -> FlowMatchEulerDiscreteScheduler:
    """Return a ``FlowMatchEulerDiscreteScheduler(shift=flow_shift)`` with ``steps`` timesteps set on ``device``.

    All other scheduler kwargs stay at their diffusers defaults (``num_train_timesteps=1000``,
    exponential time shift, no dynamic shifting), matching the source sampler.
    """

    scheduler = FlowMatchEulerDiscreteScheduler(shift=float(flow_shift))
    scheduler.set_timesteps(steps, device=device)
    return scheduler


@torch.inference_mode()
def sample_policy(
    model,
    clean_video: torch.Tensor,
    video_noise: torch.Tensor | None,
    action_noise: torch.Tensor | None,
    clean_action: torch.Tensor,
    action_mask: torch.Tensor,
    caption_embeds: torch.Tensor,
    caption_mask: torch.Tensor,
    unconditional_caption_embeds: torch.Tensor | None,
    unconditional_caption_mask: torch.Tensor | None,
    cfg_scale: float,
    data_info: dict,
    steps: int,
    flow_shift: float,
    generator: torch.Generator | None = None,
    video_cfg_scale: float | None = None,
    action_cfg_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run joint Flow-Euler sampling of the video and action streams for the policy task.

    Args:
        model: Policy callable ``model(x, timestep, y, mask=..., data_info=...)`` returning a dict
            with video velocity "x" ``[1, C, F, H, W]`` and action velocity "action_pred" ``[1, K, 80]``.
        clean_video: Clean video latent [1, C, F, H, W]; only frame 0 is consumed and kept clean.
        video_noise: Initial video noise matching clean_video, or None to draw it from ``generator``.
        action_noise: Initial action noise matching clean_action, or None to draw it from ``generator``
            (drawn after the video noise).
        clean_action: Action tensor [1, K, 80]; only its shape, dtype and device are used.
        action_mask: Bool mask matching clean_action; masked-out slots are zeroed after every step.
        caption_embeds: Conditional caption embeddings (opaque, forwarded to the model).
        caption_mask: Conditional caption mask (opaque, forwarded to the model).
        unconditional_caption_embeds: Unconditional caption embeddings, required when either stream is guided.
        unconditional_caption_mask: Unconditional caption mask, required when either stream is guided.
        cfg_scale: Text classifier-free guidance scale shared by the video and action streams; values above 1
            enable guidance (two forwards per step).
        video_cfg_scale: Guidance scale of the video stream, or None to use ``cfg_scale``.
        action_cfg_scale: Guidance scale of the action stream, or None to use ``cfg_scale``. ``1.0`` together
            with a ``cfg_scale`` above 1 guides the video only: the action integrates the exact conditional
            velocity (no ``u + s * (c - u)`` rounding) while the video still follows the guided one. The two
            streams share one transformer, so the unconditional forward runs once per step whenever either
            scale is above 1.
        data_info: Extra conditioning dict shallow-copied into every model call; the sampler adds
            ``rwm_task``, ``action80``, ``action_mask80`` and ``action_timestep`` per step.
        steps: Number of Euler steps; must be positive.
        flow_shift: Flow-matching timestep shift; must be positive and finite.
        generator: Torch generator used only when video_noise / action_noise are None.

    Returns:
        Tuple (video, action): video in clean_video.dtype, action in clean_action.dtype with gripper
        slots clamped to [0, 1]; the action is in normalized space.

    Raises:
        ValueError: Batch size above 1, or shape/mask/steps/flow_shift/CFG argument violations.
        TypeError: Model output is not a dict containing the video velocity "x".
        FloatingPointError: Sampled video or masked action contains non-finite values.
    """

    task = "policy"
    if clean_video.ndim != 5 or clean_action.ndim != 3 or clean_action.shape[-1] != 80:
        raise ValueError("clean_video/action must be [B,C,F,H,W] and [B,A,80]")
    if clean_video.shape[0] != 1 or clean_action.shape[0] != 1:
        raise ValueError("Robot80 validation currently samples one example at a time")
    if action_mask.shape != clean_action.shape or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be a bool tensor matching clean_action")
    if steps <= 0 or not math.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError("steps and flow_shift must be positive")
    video_cfg_scale = float(cfg_scale if video_cfg_scale is None else video_cfg_scale)
    action_cfg_scale = float(cfg_scale if action_cfg_scale is None else action_cfg_scale)
    if not math.isfinite(video_cfg_scale) or not math.isfinite(action_cfg_scale):
        raise ValueError("video_cfg_scale / action_cfg_scale must be finite")
    use_cfg = video_cfg_scale > 1 or action_cfg_scale > 1
    if use_cfg and (
        unconditional_caption_embeds is None or unconditional_caption_mask is None
    ):
        raise ValueError("text CFG requires unconditional caption embeds and mask")

    if video_noise is None:
        video_noise = torch.randn(
            clean_video.shape,
            device=clean_video.device,
            dtype=clean_video.dtype,
            generator=generator,
        )
    elif video_noise.shape != clean_video.shape:
        raise ValueError("video_noise must match clean_video")
    else:
        video_noise = video_noise.to(clean_video)
    if action_noise is None:
        action_noise = torch.randn(
            clean_action.shape,
            device=clean_action.device,
            dtype=clean_action.dtype,
            generator=generator,
        )
    elif action_noise.shape != clean_action.shape:
        raise ValueError("action_noise must match clean_action")
    else:
        action_noise = action_noise.to(clean_action)

    video = video_noise.clone()
    video[:, :, :1] = clean_video[:, :, :1]
    action = action_noise.masked_fill(~action_mask, 0)
    condition_data = dict(data_info)

    video_scheduler = make_scheduler(steps, flow_shift, clean_video.device)
    action_scheduler = make_scheduler(steps, flow_shift, clean_video.device)

    def _denoise_step(timestep, video, action):
        video_timestep, action_timestep = task_timesteps(timestep, task)
        video_timesteps = video_timestep.expand(batch, frames).clone()
        video_timesteps[:, 0] = 0
        action_timesteps = action_timestep.expand(clean_action.shape[:2])

        model_data = dict(condition_data)
        model_data.update(
            {
                "rwm_task": task,
                "action80": action,
                "action_mask80": action_mask,
                "action_timestep": action_timesteps,
            }
        )
        conditional_output = model(
            video,
            video_timesteps[:, None],
            caption_embeds,
            mask=caption_mask,
            data_info=model_data,
        )
        if not isinstance(conditional_output, dict) or conditional_output.get("x") is None:
            raise TypeError("world model must return a dict containing video velocity 'x'")

        if use_cfg:
            unconditional_output = model(
                video,
                video_timesteps[:, None],
                unconditional_caption_embeds,
                mask=unconditional_caption_mask,
                data_info=model_data,
            )
            if not isinstance(unconditional_output, dict) or unconditional_output.get("x") is None:
                raise TypeError("world model must return an unconditional video velocity 'x'")
        else:
            unconditional_output = None

        video_prediction = conditional_output["x"]
        if unconditional_output is not None and video_cfg_scale > 1:
            video_prediction = unconditional_output["x"] + video_cfg_scale * (
                video_prediction - unconditional_output["x"]
            )
        if video_prediction.shape != video.shape:
            raise ValueError("world-model video velocity shape differs from the latent")
        video = _video_step(
            video_scheduler, video_prediction, timestep, video, video_timesteps
        )
        video[:, :, :1] = clean_video[:, :, :1]

        action_prediction = conditional_output["action_pred"]
        if unconditional_output is not None and action_cfg_scale > 1:
            unconditional_action_prediction = unconditional_output["action_pred"]
            action_prediction = unconditional_action_prediction + action_cfg_scale * (
                action_prediction - unconditional_action_prediction
            )
        if action_prediction.shape != action.shape:
            raise ValueError("world model must return an action velocity matching action80")
        action = _action_step(
            action_scheduler,
            action_prediction,
            timestep,
            action,
            action_timesteps,
        ).masked_fill(~action_mask, 0)
        return video, action

    batch, _, frames, _, _ = video.shape
    for timestep in video_scheduler.timesteps:
        video, action = _denoise_step(timestep, video, action)

    action[..., [LEFT_GRIPPER, RIGHT_GRIPPER]] = action[
        ..., [LEFT_GRIPPER, RIGHT_GRIPPER]
    ].clamp(0, 1)
    if not bool(torch.isfinite(video).all()) or not bool(
        torch.isfinite(action.masked_select(action_mask)).all()
    ):
        raise FloatingPointError("Robot80 validation sampling produced non-finite values")
    return video, action


__all__ = ["sample_policy", "task_timesteps", "make_scheduler", "LEFT_GRIPPER", "RIGHT_GRIPPER"]
