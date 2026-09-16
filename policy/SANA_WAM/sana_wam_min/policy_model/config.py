"""Resolved architecture contract for the unified policy inference mirror.

This is intentionally not a second copy of the general SANA configuration
system.  It describes only the architecture materialized by the unified
policy checkpoint and provides a narrow adapter for the kwargs used by the
live model builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ROBOT80_DIM = 80

# Class names that share one state_dict layout and differ only in kernels
# (layers/block.py SOFTMAX_ATTENTION_CLASSES / LINEAR_ATTENTION_CLASSES /
# FFN_CLASSES).  The mirror runs the fp64-rope base math for every alias.
LINEAR_ATTENTION_NAMES = ("GatedDeltaNet", "GatedDeltaNetFP32Rope")
SOFTMAX_ATTENTION_NAMES = (
    "GatedSoftmaxAttention",
    "GatedSoftmaxAttentionFP32Rope",
    "GatedSoftmaxAttentionFlashRope",
)
FFN_NAMES = ("SwiGLU", "SwiGLUFusedAct")
MULTIVIEW_SPATIAL_ROPE_LAYOUTS = ("local_reset", "semantic_2x2")


def _value(container: Any, name: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        value = container.get(name, default)
    else:
        value = getattr(container, name, default)
    return default if value is None else value


def _require(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"kernel policy only implements the resolved inference setting "
            f"{name}={expected!r}; got {actual!r}"
        )


def _require_one_of(name: str, actual: Any, accepted: Sequence[Any]) -> None:
    if actual not in accepted:
        raise ValueError(
            f"kernel policy only implements {name} in {tuple(accepted)!r}; "
            f"got {actual!r}"
        )


def evenly_spaced_softmax_layers(depth: int, ratio: float) -> tuple[int, ...]:
    count = max(1, int(depth * ratio))
    step = depth / count
    return tuple(int((index + 1) * step) - 1 for index in range(count))


@dataclass(frozen=True)
class PolicyConfig:
    """All and only shape/runtime choices used by policy inference.

    Defaults are the production 5B P1 unified policy.  Dimensions remain
    configurable so the same implementation can be exercised by small parity
    tests; feature branches which the production policy does not use are
    deliberately absent.
    """

    input_size: int = 15
    patch_size: tuple[int, int, int] = (1, 1, 1)
    in_channels: int = 128
    hidden_size: int = 2560
    depth: int = 32
    num_heads: int = 20
    mlp_ratio: float = 4.0
    caption_channels: int = 2304
    model_max_length: int = 300
    qk_norm: bool = True
    cross_norm: bool = True
    y_norm: bool = True
    # model_video_init_config does not forward the YAML text-encoder scale, so
    # the materialized policy constructor uses Sana's 1.0 default.
    y_norm_scale_factor: float = 1.0
    norm_eps: float = 1e-5
    linear_head_dim: int = 128
    softmax_head_dim: int = 256
    softmax_layer_indices: tuple[int, ...] = (
        3,
        7,
        11,
        15,
        19,
        23,
        27,
        31,
    )
    attn_res_block_size: int = 8
    timestep_norm_scale_factor: float = 1.0
    action_dim: int = ROBOT80_DIM
    state_dim: int = ROBOT80_DIM
    action_temporal_compression: int = 8
    multiview_spatial_rope_layout: str = "semantic_2x2"
    multiview_spatial_rope_tile_shape: tuple[int, int] = (15, 30)
    fp32_attention: bool = True
    out_channels: int = 128
    # One text group shared by every token (G = 1: one query span over the video tokens and the robot tail) --
    # the OpenWAM canvas policy (sana_qwennext_openwam_canvas_policy.py _prompt_group_spans). False keeps one
    # group per view plus one for the robot tail.
    shared_prompt: bool = False
    # The canvas policy's opt-in ``model.extra.state_as_cross_attention``: the state row is projected onto one
    # appended cross-attention key of the (single) text group instead of riding the self-attention robot tail as
    # a clean token, and the action rows get an independent full-head-dim 1D RoPE over local positions 0..A-1.
    state_as_cross_attention: bool = False

    def validate(self) -> "PolicyConfig":
        _require("patch_size", tuple(self.patch_size), (1, 1, 1))
        _require("out_channels", self.out_channels, self.in_channels)
        # The unified token clock is one action row per source frame at
        # temporal compression 8 (sana_qwennext_pretrain.py).
        _require(
            "action_temporal_compression", self.action_temporal_compression, 8
        )
        _require_one_of(
            "multiview_spatial_rope_layout",
            self.multiview_spatial_rope_layout,
            MULTIVIEW_SPATIAL_ROPE_LAYOUTS,
        )
        tile = tuple(self.multiview_spatial_rope_tile_shape)
        if len(tile) != 2 or min(tile) <= 0:
            raise ValueError(
                "multiview_spatial_rope_tile_shape must contain two positive "
                f"integers, got {tile}"
            )
        if self.depth <= 0 or self.attn_res_block_size <= 0:
            raise ValueError("depth and attn_res_block_size must be positive")
        if self.linear_head_dim <= 0 or self.softmax_head_dim <= 0:
            raise ValueError("attention head dimensions must be positive")
        if self.linear_head_dim % 2 or self.softmax_head_dim % 2:
            raise ValueError("RoPE requires even attention head dimensions")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size % self.linear_head_dim:
            raise ValueError("hidden_size must be divisible by linear_head_dim")
        if self.hidden_size % self.softmax_head_dim:
            raise ValueError("hidden_size must be divisible by softmax_head_dim")
        if not self.qk_norm or not self.cross_norm or not self.y_norm:
            raise ValueError("the resolved policy requires qk_norm, cross_norm, and y_norm")
        indices = tuple(self.softmax_layer_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("softmax_layer_indices must be non-empty and unique")
        if min(indices) < 0 or max(indices) >= self.depth:
            raise ValueError("softmax_layer_indices fall outside the trunk depth")
        if self.state_as_cross_attention and not self.shared_prompt:
            # the live model defines the opt-in only on the canvas policy, whose prompt is shared (G = 1)
            raise ValueError(
                "state_as_cross_attention is only defined for the shared-prompt (OpenWAM canvas) policy"
            )
        return self

    @classmethod
    def from_sana_kwargs(cls, **kwargs: Any) -> "PolicyConfig":
        """Resolve the live builder kwargs without importing the live code.

        Absent input_size / softmax_head_dim / softmax_ratio / vae_stride /
        rope layout and tile take the live constructor defaults; the trunk
        dims and the branch flags (pred_sigma, y_norm, cross_norm,
        use_time_conditioning) are pinned to the production policy and must
        be passed explicitly to match a differently configured live model.
        """

        config = kwargs.get("config")
        model_cfg = _value(config, "model")
        vae_cfg = _value(config, "vae")

        # Fail loudly if a caller points the mirror at another branch.
        expected_settings = {
            "use_pe": True,
            "pos_embed_type": "wan_rope",
            "use_attn_res": True,
            "use_time_conditioning": False,
            "use_dual_attn_res_routing": False,
            "cross_attn_image_embeds": False,
            "pred_sigma": False,
            "rope_fhw_dim": None,
        }
        for name, expected in expected_settings.items():
            actual = kwargs.get(name, _value(model_cfg, name, expected))
            _require(name, actual, expected)
        for name, accepted in (
            ("ffn_type", FFN_NAMES),
            ("linear_attn_type", LINEAR_ATTENTION_NAMES),
            ("softmax_attn_type", SOFTMAX_ATTENTION_NAMES),
        ):
            actual = kwargs.get(name, _value(model_cfg, name, accepted[0]))
            _require_one_of(name, actual, accepted)

        depth = int(kwargs.get("depth", cls.depth))
        ratio = float(kwargs.get("softmax_ratio", _value(model_cfg, "softmax_ratio", 0.25)))
        indices = kwargs.get(
            "softmax_layer_indices",
            _value(model_cfg, "softmax_layer_indices", None),
        )
        if indices is None:
            indices = evenly_spaced_softmax_layers(depth, ratio)
        else:
            indices = tuple(int(index) for index in indices)

        stride: Sequence[float] | None = _value(vae_cfg, "vae_stride", None)
        temporal_compression = int(stride[0]) if stride else 8
        in_channels = int(kwargs.get("in_channels", 128))
        pred_sigma = bool(kwargs.get("pred_sigma", False))
        out_channels = in_channels * 2 if pred_sigma else in_channels
        linear_head_dim = int(kwargs.get("linear_head_dim", cls.linear_head_dim))
        softmax_head_dim = kwargs.get("softmax_head_dim")
        if softmax_head_dim is None:
            softmax_head_dim = 2 * linear_head_dim

        model_extra = _value(model_cfg, "extra", None) or {}
        state_as_cross_attention = bool(
            kwargs.get(
                "state_as_cross_attention",
                _value(model_extra, "state_as_cross_attention", False),
            )
        )

        resolved = cls(
            input_size=int(kwargs.get("input_size", 32)),
            patch_size=tuple(kwargs.get("patch_size", cls.patch_size)),
            in_channels=in_channels,
            hidden_size=int(kwargs.get("hidden_size", cls.hidden_size)),
            depth=depth,
            num_heads=int(kwargs.get("num_heads", cls.num_heads)),
            mlp_ratio=float(kwargs.get("mlp_ratio", cls.mlp_ratio)),
            caption_channels=int(kwargs.get("caption_channels", cls.caption_channels)),
            model_max_length=int(kwargs.get("model_max_length", cls.model_max_length)),
            qk_norm=bool(kwargs.get("qk_norm", cls.qk_norm)),
            cross_norm=bool(kwargs.get("cross_norm", cls.cross_norm)),
            y_norm=bool(kwargs.get("y_norm", cls.y_norm)),
            y_norm_scale_factor=float(
                kwargs.get("y_norm_scale_factor", cls.y_norm_scale_factor)
            ),
            norm_eps=float(kwargs.get("norm_eps", cls.norm_eps)),
            linear_head_dim=linear_head_dim,
            softmax_head_dim=int(softmax_head_dim),
            softmax_layer_indices=tuple(indices),
            attn_res_block_size=int(
                kwargs.get("attn_res_block_size", cls.attn_res_block_size)
            ),
            timestep_norm_scale_factor=float(
                kwargs.get(
                    "timestep_norm_scale_factor", cls.timestep_norm_scale_factor
                )
            ),
            action_dim=ROBOT80_DIM,
            state_dim=ROBOT80_DIM,
            action_temporal_compression=temporal_compression,
            multiview_spatial_rope_layout=str(
                _value(model_cfg, "multiview_spatial_rope_layout", "local_reset")
            ),
            multiview_spatial_rope_tile_shape=tuple(
                int(value)
                for value in _value(
                    model_cfg, "multiview_spatial_rope_tile_shape", (15, 30)
                )
            ),
            fp32_attention=bool(
                kwargs.get(
                    "use_fp32_attention",
                    _value(model_cfg, "fp32_attention", cls.fp32_attention),
                )
            ),
            out_channels=out_channels,
            shared_prompt=bool(kwargs.get("shared_prompt", False)),
            state_as_cross_attention=state_as_cross_attention,
        )
        return resolved.validate()

    @property
    def block_attn_types(self) -> tuple[str, ...]:
        """Resolved hybrid-attention kind for every trunk block."""

        softmax = set(self.softmax_layer_indices)
        return tuple(
            "GatedSoftmaxAttention" if index in softmax else "GatedDeltaNet"
            for index in range(self.depth)
        )


__all__ = [
    "FFN_NAMES",
    "LINEAR_ATTENTION_NAMES",
    "MULTIVIEW_SPATIAL_ROPE_LAYOUTS",
    "PolicyConfig",
    "ROBOT80_DIM",
    "SOFTMAX_ATTENTION_NAMES",
    "evenly_spaced_softmax_layers",
]
