"""In-process inference session of the SANA MoT policy (RoboDojo ARX-X5).

Same shape as SANA_WAM's ``PolicyInferenceSession`` with the model swapped for the MoT mirror, in the video layouts
the policy is trained on (``video_layout``, resolved from the training yaml's ``data.extra.multiview`` and its
predecessors; a third one, ``sana_pixel_canvas``, tiles the cameras into one 320x480 2x2 pixel canvas):

* ``multiview`` (the default, the contract of Sana's own bidirectional deploy session): every camera frame goes through
  the training transform (``ToTensorVideo -> ResizeCrop 256x320 -> Normalize``), is VAE-encoded on its own as frame 0 of a
  zero-filled latent window, and the V windows are packed into one strip ``[1, C, F, 1, V*h*w]`` with ``view_count`` V,
  ``view_latent_shape`` and ``view_slot_ids`` (0, 2, 3); the text is G = V + 1 token-group rows.
* ``openwam_canvas``: the three frames are composited into ONE 384 x 320 canvas (native 12 x 10 latent grid,
  ``view_count`` 1) and the text is ONE shared prompt row.

Sampling (video-then-action noise from one generator, Flow-Euler under bf16 autocast, per-stream CFG) and the
post-processing (denormalize -> anchor + delta only for anchor_delta artifacts -> gripper clip) are the shared
``sana_wam_min`` routines. A strided-video checkpoint (``robot_sft.video_fps``) IS served: the stride
``s = (tier_num_frames - 1) / video_fps`` shortens the latent window the observation frame is placed in
(``1 + video_fps / 8`` frames instead of ``1 + (rows - 1) / 8``) and the batch carries
``data_info['video_frame_stride']``; the action rows stay dense. Sana's own deploy sessions still refuse such a
config -- they build the dense window of the tier, which would hand the policy a video clock it never saw.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from sana_wam_min.actions import model_action_to_absolute
from sana_wam_min.config import (
    sana_pixel_canvas_hw_from_train_config,
    ROBOT_BASE_EEF_LAYOUTS,
    ROBOT_BASE_EEF_ONLY,
    action_mode_from_train_config,
    eef_target_mode_from_train_config,
    robot_base_eef_layout_from_train_config,
    sana_pixel_pad_from_train_config,
    video_frame_stride_from_train_config,
    view_resize_from_train_config,
)
from sana_wam_min.openwam_canvas import OPENWAM_PROMPT_TEXTS
from sana_wam_min.sana_pixel_canvas import (
    SANA_PIXEL_PROMPT_TEXTS,
    SANA_PIXEL_VIEW_KEY,
    expected_sana_pixel_latent_hw,
    sana_pixel_canvas_from_frames,
)
from sana_wam_min.text import action_mode_text as render_action_mode_text
from sana_wam_min.frame_stride import strided_video_frames
from sana_wam_min.multiview import pack_multiview_latents
from sana_wam_min.pixels import frame_to_model_tensor, frames_to_vae_input, latent_frame_count, observation_window, target_size_hw, tier
from sana_wam_min.robodojo_io import ROBODOJO_MODEL_FPS_HZ, ROBODOJO_VIEW_ORDER, VIEW_SLOT_IDS
from sana_wam_min.robot80 import (
    ROBOT80_DIM,
    Normalization,
    load_normalization,
    normalize_state,
    normalized_gripper_bounds,
    select_eef_target_normalization,
    select_joint_target_normalization,
)
from sana_wam_min.sampler import sample_policy
from sana_wam_min.session import (
    resolve_action_flow_shift,
    NORMALIZATION_FILE_NAME,
    PredictResult,
    checkpoint_root,
    find_train_config,
    normalization_file_name,
    resolve_branch_cfg_scales,
    resolve_sampling_knobs,
    training_normalization_pin,
)
from sana_wam_min.text import encode_prompt_rows, load_text_encoder
from sana_wam_min.vae import VaeBundle, encode_video, load_vae

from .canvas import COMPOSITE_VIEW_KEY, OPENWAM_CAMERA_LAYOUT, canvas_from_frames, canvas_to_model_tensor
from .checkpoint import build_mot_model, load_mot_weights, read_mot_state_dict, resolve_checkpoint_file
from .config import (
    load_train_config,
    mot_config_from_train_config,
    resolve_mot_rope,
    resolve_mot_text_contract,
    strided_video_fps,
)
from .mot_model.checkpoint import detect_context_layout
from .prompt import render_canvas_prompt_rows, render_multiview_prompt_rows, unconditional_rows_from_conditional

# sha256 of the absolute-joint-target normalization artifact the local-cluster MoT RoboDojo yamls pin (the f25 multiview
# default sft_robodojo_mot_jointabs_f25_320px.yaml and the f25 canvas yaml). Checkpoints ship their own copy, which wins;
# the NSC canvas run's artifact is 3cd3ce1d... (same construction, slightly different quantiles -- not interchangeable).
TRAINING_NORMALIZATION_SHA256 = "802f8fe90cbe35688859d2ed2b058e04fd33a57261124422e7dc7e686db3d7ad"
# Fallback copy shipped next to the package (policy/SANA_MOT/normalization/).
PACKAGED_NORMALIZATION_PATH = Path(__file__).resolve().parent.parent / "normalization" / NORMALIZATION_FILE_NAME
PROMPT_CACHE_SIZE = 8
DEFAULT_JOINT_TARGET_MODE = "absolute"

if tuple(OPENWAM_CAMERA_LAYOUT) != tuple(ROBODOJO_VIEW_ORDER):
    raise RuntimeError("the OpenWAM canvas camera order must equal the RoboDojo view order")


def resolve_normalization_path(
    checkpoint_dir: str | Path, normalization_path: Optional[str | Path] = None, num_frames: Optional[int] = None
) -> Path:
    """Explicit path, else ``<ckpt>/normalization/<canonical name>``, else the packaged copy; never a directory scan.

    ``num_frames`` (the training window length) picks the canonical name -- the f33 lines ship ``..._f33_...json`` --
    and the f25 name is tried after it so every historical checkpoint resolves exactly as before.
    """

    if normalization_path is not None:
        return Path(normalization_path).expanduser().resolve()
    root = checkpoint_root(checkpoint_dir) / "normalization"
    for name in dict.fromkeys((normalization_file_name(num_frames), NORMALIZATION_FILE_NAME)):
        canonical = root / name
        if canonical.is_file():
            return canonical
    return PACKAGED_NORMALIZATION_PATH


def load_checked_normalization(
    checkpoint_dir: str | Path,
    train_cfg: dict,
    normalization_path: Optional[str | Path] = None,
    expected_sha256: Optional[str] = None,
) -> Normalization:
    """Resolve, sha-pin and load the normalization artifact, then cross-check it against the training yaml.

    The pin is enforced whenever ``expected_sha256`` is given; the packaged copy is pinned to the MoT line's
    artifact by default. An implicitly resolved artifact must also be the one the yaml pins.
    """

    explicit_pin = expected_sha256 is not None
    norm_path = resolve_normalization_path(
        checkpoint_dir, normalization_path, num_frames=(train_cfg.get("data") or {}).get("num_frames")
    )
    if not explicit_pin and norm_path == PACKAGED_NORMALIZATION_PATH:
        expected_sha256 = TRAINING_NORMALIZATION_SHA256
    normalization = load_normalization(norm_path, expected_sha256=expected_sha256)
    yaml_pin = training_normalization_pin(train_cfg)
    if normalization_path is None and not explicit_pin and yaml_pin is not None and yaml_pin != normalization.sha256:
        raise ValueError(
            f"normalization artifact {norm_path} (sha256 {normalization.sha256[:12]}...) is not the one the training yaml pins "
            f"(data.extra.robot_sft.normalization_sha256 {yaml_pin[:12]}...); put the run's artifact at "
            f"<checkpoint>/normalization/{NORMALIZATION_FILE_NAME} or set normalization_path / normalization_sha256 explicitly"
        )
    trained_mode = str(((train_cfg.get("data") or {}).get("extra") or {}).get("joint_target_mode", DEFAULT_JOINT_TARGET_MODE))
    # Same selection as Sana's dataset (select_joint_target_normalization, then select_eef_target_normalization for
    # every non-qwen action mode): the eefabs line pins the absolute-joint artifact 1fe3b7e7 whose EEF statistics are
    # anchor-delta, so its action EEF slots take the state statistics (the same absolute robot-base poses).
    try:
        normalization = select_joint_target_normalization(normalization, trained_mode)
        normalization = select_eef_target_normalization(normalization, eef_target_mode_from_train_config(train_cfg))
    except ValueError as error:
        raise ValueError(f"{error} ({norm_path})") from error
    num_frames = int(train_cfg["data"]["num_frames"])
    if normalization.num_frames is not None and normalization.num_frames != num_frames:
        raise ValueError(f"normalization num_frames {normalization.num_frames} != training {num_frames} ({norm_path})")
    return normalization


def video_frame_stride(train_cfg: dict) -> int:
    """The video frame stride a MoT checkpoint trained with: 1 without ``robot_sft.video_fps``, else
    ``(tier_num_frames - 1) / video_fps`` (Sana ``rwm/mot`` @ ``b3b9e0e9e``, design doc ``strided_video_design.md``).

    Sana's own deploy sessions still refuse a strided config; this session serves it instead, because the deployment
    difference is closed: only the observation frame is real, so a stride shortens the latent window the observation is
    placed in and publishes ``data_info['video_frame_stride']``, which the MoT port already reads to expect
    ``(F - 1) * 8 * s`` action rows and to take the independent action RoPE branch.
    """

    return video_frame_stride_from_train_config(train_cfg)


class MoTInferenceSession:
    """The loaded MoT policy and its conditioning encoders, driven one observation chunk at a time."""

    def __init__(
        self,
        model: torch.nn.Module,
        vae: VaeBundle,
        tokenizer: Any,
        text_encoder: Any,
        normalization: Normalization,
        train_config: dict,
        device: str | torch.device = "cuda",
        steps: int = 50,
        cfg_scale: float = 1.0,
        flow_shift: float = 3.5,
        checkpoint_path: Optional[str] = None,
        video_cfg_scale: Optional[float] = None,
        action_cfg_scale: Optional[float] = None,
        video_layout: Optional[str] = None,
        text_groups: Any = None,
        canvas_prompt: Optional[str] = None,
        robot_base_eef_layout: Optional[str] = None,
        context_layout: Optional[str] = None,
        action_flow_shift: Optional[float] = None,
        view_resize: Optional[str] = None,
    ) -> None:
        self.video_fps = strided_video_fps(train_config)
        self.video_frame_stride = video_frame_stride(train_config)
        self.video_layout = str(video_layout or getattr(model, "video_layout", None) or "multiview")
        if self.video_layout not in ("multiview", "openwam_canvas", "sana_pixel_canvas"):
            raise ValueError(f"unknown MoT video_layout {self.video_layout!r}")
        self.canvas = self.video_layout != "multiview"
        context_layout = context_layout or getattr(model, "context_layout", None) or "context_embedder"
        # text contract: the strip's G = V + 1 rows, or a canvas's rows with the era's descriptor
        self.text_groups, self.canvas_prompt, self.text_contract = resolve_mot_text_contract(
            train_config, context_layout, text_groups, canvas_prompt
        )
        if self.video_layout == "openwam_canvas":
            self.canvas_view_text = OPENWAM_PROMPT_TEXTS[self.canvas_prompt]
        elif self.video_layout == "sana_pixel_canvas":
            self.canvas_view_text = SANA_PIXEL_PROMPT_TEXTS[self.canvas_prompt]
        else:
            self.canvas_view_text = None
        # the sana_pixel canvas size the recipe's aspect_ratio_type names (320x480 or 320x512, zekai-merge 88a22ba0c)
        self.sana_pixel_canvas_hw = (
            sana_pixel_canvas_hw_from_train_config(train_config) if self.video_layout == "sana_pixel_canvas" else None
        )
        # the Action Mode sentence follows the checkpoint's action mode / target modes (the MoT eefabs line is EEF-only)
        self.action_mode = action_mode_from_train_config(train_config)
        self.eef_target_mode = eef_target_mode_from_train_config(train_config)
        layout = robot_base_eef_layout or robot_base_eef_layout_from_train_config(train_config, normalization)
        if layout not in ROBOT_BASE_EEF_LAYOUTS:
            raise ValueError(f"robot_base_eef_layout must be one of {ROBOT_BASE_EEF_LAYOUTS}, got {layout!r}")
        self.robot_base_eef_layout = layout if self.action_mode == "robot_base_eef" else None
        self.action_mode_text = render_action_mode_text(
            self.action_mode,
            normalization.joint_target_mode,
            self.eef_target_mode,
            eef_only=self.robot_base_eef_layout == ROBOT_BASE_EEF_ONLY,
        )
        self.model = model
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.normalization = normalization
        self.train_config = train_config
        self.device = torch.device(device)
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.video_cfg_scale, self.action_cfg_scale = resolve_branch_cfg_scales(self.cfg_scale, video_cfg_scale, action_cfg_scale)
        self.flow_shift = float(flow_shift)
        # the action stream's own schedule when the recipe declares one (rwm/mot b865ad732); None = shared with the video
        self.action_flow_shift = resolve_action_flow_shift(train_config, action_flow_shift)
        # how the checkpoint's SFT views reached their bucket (crop before rwm/mot 034e55dca, stretch after)
        self.view_resize, self.view_resize_source = view_resize_from_train_config(train_config, view_resize)
        if sana_pixel_pad_from_train_config(train_config) == "masked":
            raise NotImplementedError(
                "model.extra.sana_pixel_pad: masked (rwm/mot 2a0c69d1f) drops the black quadrant's latent cells from the "
                "token sequence; this adapter mirrors only the unmasked MoT -- port the mask before serving it"
            )
        self.checkpoint_path = checkpoint_path
        self.multi_fps = train_config["data"].get("multi_fps") or None
        self.fps = ROBODOJO_MODEL_FPS_HZ
        self.view_order = ROBODOJO_VIEW_ORDER
        self.view_slot_ids = VIEW_SLOT_IDS
        self.image_size = int(train_config["model"]["image_size"])
        self.joint_target_mode = normalization.joint_target_mode
        self.caption_max_length = int(train_config["text_encoder"]["model_max_length"])
        self._prompt_cache: OrderedDict = OrderedDict()

    @classmethod
    def from_paths(
        cls,
        checkpoint_dir: str,
        text_encoder_path: str,
        vae_path: str,
        normalization_path: Optional[str] = None,
        device: str | torch.device = "cuda",
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        flow_shift: Optional[float] = None,
        expected_normalization_sha256: Optional[str] = None,
        video_cfg_scale: Optional[float] = None,
        action_cfg_scale: Optional[float] = None,
        rope_mode: Optional[str] = None,
        text_groups: Any = None,
        canvas_prompt: Optional[str] = None,
        robot_base_eef_layout: Optional[str] = None,
        action_flow_shift: Optional[float] = None,
        view_resize: Optional[str] = None,
    ) -> "MoTInferenceSession":
        """Load every component from disk: train yaml -> normalization -> bf16 MoT model + weights -> VAE -> Gemma."""

        device = torch.device(device)
        config_path = find_train_config(checkpoint_dir)
        train_cfg = load_train_config(str(config_path))
        steps, cfg_scale, flow_shift = resolve_sampling_knobs(train_cfg, steps, cfg_scale, flow_shift)
        video_cfg_scale, action_cfg_scale = resolve_branch_cfg_scales(cfg_scale, video_cfg_scale, action_cfg_scale)
        normalization = load_checked_normalization(checkpoint_dir, train_cfg, normalization_path, expected_normalization_sha256)

        state, weights_path = read_mot_state_dict(str(checkpoint_dir))
        context_layout = detect_context_layout(state)
        rope, rope_label = resolve_mot_rope(train_cfg, context_layout, rope_mode)
        mot_config = mot_config_from_train_config(
            train_cfg, context_layout=context_layout, rope=rope, canvas_text_groups=text_groups
        )
        model = build_mot_model(mot_config, dtype=torch.bfloat16, device=device)
        load_report = load_mot_weights(model, state, device=device, source=weights_path)
        del state

        vae = load_vae(vae_path, device=device, dtype=torch.bfloat16)
        tokenizer, text_encoder = load_text_encoder(text_encoder_path, device=device)
        session = cls(
            model=model,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            normalization=normalization,
            train_config=train_cfg,
            device=device,
            steps=steps,
            cfg_scale=cfg_scale,
            flow_shift=flow_shift,
            checkpoint_path=resolve_checkpoint_file(str(checkpoint_dir)),
            video_cfg_scale=video_cfg_scale,
            action_cfg_scale=action_cfg_scale,
            video_layout=mot_config.video_layout,
            text_groups=text_groups,
            canvas_prompt=canvas_prompt,
            robot_base_eef_layout=robot_base_eef_layout,
            context_layout=context_layout,
            action_flow_shift=action_flow_shift,
            view_resize=view_resize,
        )
        session.load_report = load_report
        session.rope_contract = rope_label
        return session

    # -- conditioning --------------------------------------------------------

    @property
    def uses_cfg(self) -> bool:
        """True when either stream is guided, i.e. the unconditional row must be encoded and forwarded."""

        return self.video_cfg_scale > 1 or self.action_cfg_scale > 1

    def encode_rows(
        self,
        conditional_rows: Sequence[str],
        unconditional_rows: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode the G conditional rows (+ the unconditional rows in the same batch when either stream is guided)."""

        rows = [tuple(conditional_rows)]
        if self.uses_cfg:
            if unconditional_rows is None:
                unconditional_rows = unconditional_rows_from_conditional(conditional_rows)
            rows.append(tuple(unconditional_rows))
        y, mask = encode_prompt_rows(rows, self.tokenizer, self.text_encoder, self.device, max_length=self.caption_max_length)
        if self.uses_cfg:
            return y[:1], mask[:1], y[1:2], mask[1:2]
        return y, mask, None, None

    def encode_instruction(self, instruction: str):
        """Render the prompt rows of the video layout for ``instruction`` (G = V + 1, or the canvas's one shared row) and encode them (LRU-cached)."""

        cached = self._prompt_cache.get(instruction)
        if cached is not None:
            return cached
        if self.canvas:
            rows = render_canvas_prompt_rows(
                instruction,
                include_instruction=True,
                view_text=self.canvas_view_text,
                groups=self.text_groups,
                action_mode_text=self.action_mode_text,
            )
        else:
            rows = render_multiview_prompt_rows(
                instruction, include_instruction=True, view_order=self.view_order, action_mode_text=self.action_mode_text
            )
        result = self.encode_rows(rows, None)
        if len(self._prompt_cache) >= PROMPT_CACHE_SIZE:
            self._prompt_cache.popitem(last=False)
        self._prompt_cache[instruction] = result
        return result

    def encode_observation(self, frames_rgb: Sequence[np.ndarray], latent_frames: int) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
        """Encode frame 0 of the observation into the model's latent window; returns ``(window, view_shapes)``.

        multiview: each view through the training transform, encoded on its own, zero-filled window per view, V windows
        packed into the strip ``[1, C, F, 1, V*h*w]``. canvas: the three frames composited, encoded once, native grid.
        """

        if len(frames_rgb) != len(self.view_order):
            raise ValueError(f"expected {len(self.view_order)} views in order {self.view_order}, got {len(frames_rgb)}")
        if self.video_layout == "openwam_canvas":
            canvas = canvas_from_frames(frames_rgb, self.view_order)
            pixels = [canvas_to_model_tensor(canvas)]
        elif self.video_layout == "sana_pixel_canvas":
            pixels = [sana_pixel_canvas_from_frames(
                [np.asarray(frame) for frame in frames_rgb], self.view_slot_ids, self.sana_pixel_canvas_hw, view_resize=self.view_resize
            )]
        else:
            pixels = []
            for frame in frames_rgb:
                frame = np.asarray(frame)
                target = target_size_hw(self.image_size, frame_hw=(int(frame.shape[0]), int(frame.shape[1])))
                pixels.append(frame_to_model_tensor(frame, target, self.view_resize))
        windows = []
        for view_pixels in pixels:
            video = frames_to_vae_input(view_pixels.unsqueeze(0)).to(device=self.vae.device, dtype=self.vae.dtype)
            windows.append(observation_window(encode_video(self.vae, video), latent_frames))
        view_shapes = tuple((int(w.shape[-2]), int(w.shape[-1])) for w in windows)
        if len(set(view_shapes)) != 1:
            raise ValueError(f"the MoT video expert needs equal view tiles; got view latent shapes {view_shapes}")
        if self.video_layout == "sana_pixel_canvas":
            expected = expected_sana_pixel_latent_hw(self.vae.spatial_compression, self.sana_pixel_canvas_hw)
            if view_shapes[0] != expected:
                raise ValueError(f"sana_pixel canvas latent grid {view_shapes[0]} differs from the trained {expected[0]}x{expected[1]} grid")
        window, _ = pack_multiview_latents(windows)
        return window, view_shapes

    def build_data_info(
        self,
        view_shapes: Sequence[tuple[int, int]],
        initial_state80: torch.Tensor,
        initial_state_mask80: torch.Tensor,
        action80: torch.Tensor,
        action_mask80: torch.Tensor,
    ) -> dict:
        """Assemble the conditioning dict the MoT forward consumes (the sampler adds the per-step keys)."""

        num_views = len(view_shapes)
        expected = 1 if self.canvas else len(self.view_slot_ids)
        if num_views != expected:
            raise ValueError(f"{self.video_layout} layout expects {expected} latent view(s), got {num_views}")
        device = self.device
        info = {
            "rwm_task": "policy",
            "model_fps": torch.tensor([float(self.fps)], device=device),
            "num_views_per_sample": num_views,
            "sample_batch_size": 1,
            "view_count": torch.tensor([num_views], dtype=torch.int64, device=device),
            "view_latent_shape": torch.tensor([[list(shape) for shape in view_shapes]], dtype=torch.int64, device=device),
            "initial_state80": initial_state80.reshape(1, ROBOT80_DIM).to(device=device, dtype=torch.float32),
            "initial_state_condition_mask80": initial_state_mask80.reshape(1, ROBOT80_DIM).to(device=device, dtype=torch.bool),
            "action80": action80.to(device=device, dtype=torch.float32),
            "action_mask80": action_mask80.to(device=device, dtype=torch.bool),
            "camera_conditioning_enabled": False,
        }
        if self.canvas:
            info["view_keys"] = [SANA_PIXEL_VIEW_KEY if self.video_layout == "sana_pixel_canvas" else COMPOSITE_VIEW_KEY]
        else:
            info["view_slot_ids"] = torch.tensor(self.view_slot_ids, dtype=torch.int64, device=device)
        # Published only above stride 1: a dense batch must stay byte-for-byte the batch the historical lines sent.
        if self.video_frame_stride != 1:
            info["video_frame_stride"] = torch.tensor([int(self.video_frame_stride)], dtype=torch.int64, device=device)
        return info

    # -- sampling -----------------------------------------------------------

    def _sample(
        self,
        clean_video: torch.Tensor,
        clean_action: torch.Tensor,
        action_mask: torch.Tensor,
        text: tuple,
        data_info: dict,
        generator: Optional[torch.Generator],
        video_noise: Optional[torch.Tensor],
        action_noise: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        caption_embeds, caption_mask, uncond_embeds, uncond_mask = text
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            if video_noise is None:
                video_noise = torch.randn(clean_video.shape, device=self.device, dtype=clean_video.dtype, generator=generator)
            if action_noise is None:
                action_noise = torch.randn(clean_action.shape, device=self.device, dtype=clean_action.dtype, generator=generator)
            return sample_policy(
                self.model,
                clean_video,
                video_noise,
                action_noise,
                clean_action,
                action_mask,
                caption_embeds,
                caption_mask,
                uncond_embeds,
                uncond_mask,
                self.cfg_scale,
                data_info,
                self.steps,
                self.flow_shift,
                video_cfg_scale=self.video_cfg_scale,
                action_cfg_scale=self.action_cfg_scale,
                gripper_bounds=normalized_gripper_bounds(self.normalization),
                action_flow_shift=self.action_flow_shift,
            )

    @torch.inference_mode()
    def predict(
        self,
        frames_rgb: Sequence[np.ndarray],
        state80_raw: np.ndarray,
        state_mask80: np.ndarray,
        instruction: str,
        generator: Optional[torch.Generator] = None,
    ) -> PredictResult:
        """Run one chunk prediction from RGB uint8 HxWx3 frames (view order), raw Robot80 state and instruction."""

        start = time.perf_counter()
        frames, k_actions = tier(self.fps, None, self.multi_fps)
        # The action rows stay dense (k_actions = rows - 1); only the video is sampled every stride-th row.
        video_frames = strided_video_frames(frames, self.video_frame_stride, self.vae.temporal_compression)
        latent_frames = latent_frame_count(video_frames, self.vae.temporal_compression)

        text = self.encode_instruction(instruction)
        window, view_shapes = self.encode_observation(frames_rgb, latent_frames)

        mask80 = torch.as_tensor(np.asarray(state_mask80)).to(device=self.device, dtype=torch.bool)
        action_mask = mask80.reshape(1, 1, ROBOT80_DIM).expand(1, k_actions, ROBOT80_DIM).contiguous()
        clean_action = torch.zeros((1, k_actions, ROBOT80_DIM), device=self.device, dtype=torch.float32)
        initial_state80 = normalize_state(state80_raw, state_mask80, self.normalization)
        data_info = self.build_data_info(view_shapes, initial_state80, mask80, clean_action, action_mask)

        video, action = self._sample(window, clean_action, action_mask, text, data_info, generator, None, None)

        action80_model = action.reshape(k_actions, ROBOT80_DIM).detach().to(device="cpu", dtype=torch.float32)
        action_mask_cpu = action_mask[0].detach().cpu()
        action80_raw = model_action_to_absolute(
            action80_model, action_mask_cpu, self.normalization, np.asarray(state80_raw), np.asarray(state_mask80)
        )
        receipt = {
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "video_cfg_scale": self.video_cfg_scale,
            "action_cfg_scale": self.action_cfg_scale,
            "flow_shift": self.flow_shift,
            "action_flow_shift": self.action_flow_shift,
            "view_resize": self.view_resize,
            "seed": None if generator is None else int(generator.initial_seed()),
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "checkpoint": self.checkpoint_path,
            "normalization_sha256": self.normalization.sha256,
            "joint_target_mode": self.joint_target_mode,
            "video_layout": self.video_layout,
            "text_groups": self.text_groups,
            "canvas_prompt": self.canvas_prompt,
            "rope": getattr(self.model, "rope", None),
            "view_latent_shapes": view_shapes,
            "frames": frames,
            "video_frames": video_frames,
            "video_frame_stride": self.video_frame_stride,
            "k_actions": k_actions,
        }
        return PredictResult(
            action80_raw_absolute=action80_raw,
            action80_model=action80_model,
            action_mask=action_mask_cpu,
            video_latent=video,
            receipt=receipt,
        )

    @torch.inference_mode()
    def predict_from_latent(
        self,
        clean_video_window: torch.Tensor,
        view_latent_shapes: Sequence[tuple[int, int]],
        prompt_rows_cond: Sequence[str],
        prompt_rows_uncond: Optional[Sequence[str]],
        initial_state80_normalized: torch.Tensor,
        state_mask: torch.Tensor,
        clean_action_normalized: torch.Tensor,
        action_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        video_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay the sampler on a pre-encoded window (strip or canvas); noise comes from ``generator`` unless both tensors are given."""

        clean_video = clean_video_window.to(self.device)
        clean_action = clean_action_normalized.to(device=self.device, dtype=torch.float32)
        mask = action_mask.to(device=self.device, dtype=torch.bool)
        text = self.encode_rows(prompt_rows_cond, prompt_rows_uncond)
        data_info = self.build_data_info(
            tuple((int(h), int(w)) for h, w in view_latent_shapes),
            torch.as_tensor(initial_state80_normalized),
            torch.as_tensor(state_mask),
            clean_action,
            mask,
        )
        noise = (
            None if video_noise is None else video_noise.to(self.device),
            None if action_noise is None else action_noise.to(self.device),
        )
        video, action = self._sample(clean_video, clean_action, mask, text, data_info, generator, *noise)
        return action, video


__all__ = [
    "DEFAULT_JOINT_TARGET_MODE",
    "video_frame_stride",
    "MoTInferenceSession",
    "NORMALIZATION_FILE_NAME",
    "PACKAGED_NORMALIZATION_PATH",
    "PredictResult",
    "TRAINING_NORMALIZATION_SHA256",
    "find_train_config",
    "load_checked_normalization",
    "resolve_normalization_path",
]
