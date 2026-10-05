"""Live parity of the causal mirror against Sana's chunk-causal policy (``SANA_CAUSAL_REPO``, default ~/zekail/Sana).

The live class and the mirror get identical random weights (tiny dims, fp64, CPU, xformers off, ``flash_attn`` bound to
an fp64 reference varlen as Sana's own tests do). The decisive check is the teacher-forced deploy contract: the
two-stream TRAINING forward over [obs | clean 0..L-2 | noisy 0..L-1] -- including the softmax sliding window, whose
observation entry evicts -- must equal, chunk by chunk, what the streaming chain computes (observation prefill, the
cache-free chunk-0 window, then for every chunk a cached denoising window followed by the commit of the clean chunk).
Run on its own (the first file of a session owns the ``dev`` package).
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import _paths  # noqa: F401

from sana_wam_causal.contract import CausalContract
from sana_wam_causal.layers import LayerCache
from sana_wam_causal.model import CausalPolicyModel
from sana_wam_causal.session import CausalMemory
from sana_wam_min.policy_model.config import PolicyConfig

SANA_CAUSAL_REPO = os.path.expanduser(os.environ.get("SANA_CAUSAL_REPO", "~/zekail/Sana"))
FPS = 16.0
C_LAT = 4
VIEW = (3, 4)
TEXT_LEN, TEXT_CH = 7, 16
CLOSE = {"rtol": 1e-9, "atol": 1e-12}            # fp64 everywhere (fp32 attention off on both sides)
FP32_ATTENTION_CLOSE = {"rtol": 1e-5, "atol": 1e-8}  # fp32 attention on: CPU SDPA kernels pick layouts freely


def _reference_varlen(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0):
    """fp64 varlen attention over packed (T, H, D): one (query span, key span) pair at a time (Sana
    ``tests/test_causal_text_cp._reference_varlen``)."""
    assert dropout_p == 0.0
    scale = q.shape[-1] ** -0.5
    q_bounds = [int(b) for b in cu_seqlens_q.tolist()]
    k_bounds = [int(b) for b in cu_seqlens_k.tolist()]
    outs = []
    for (q_lo, q_hi), (k_lo, k_hi) in zip(zip(q_bounds, q_bounds[1:]), zip(k_bounds, k_bounds[1:])):
        qs, ks, vs = q[q_lo:q_hi], k[k_lo:k_hi], v[k_lo:k_hi]
        att = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        outs.append(torch.einsum("hqk,khd->qhd", att.softmax(-1), vs))
    return torch.cat(outs, dim=0)


@pytest.fixture(autouse=True)
def _fake_flash_attn(monkeypatch):
    # import the live tree first: diffusers probes flash_attn.__spec__ while Sana's builder imports it, and the
    # causal cross-attention imports flash_attn_varlen_func at call time, which is when the reference must be bound
    _live_modules()
    module = types.ModuleType("flash_attn")
    module.flash_attn_varlen_func = _reference_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", module)


class _Cfg(SimpleNamespace):
    def __getattr__(self, name):
        return None


def _live_modules():
    if not os.path.isdir(os.path.join(SANA_CAUSAL_REPO, "dev", "rwm")):
        pytest.skip(f"no Sana checkout at {SANA_CAUSAL_REPO} (set SANA_CAUSAL_REPO)")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    if SANA_CAUSAL_REPO not in sys.path:
        sys.path.insert(0, SANA_CAUSAL_REPO)
    causal = pytest.importorskip("dev.rwm.diffusion.model.nets.sana_qwennext_policy_causal")
    if not os.path.abspath(causal.__file__).startswith(os.path.abspath(SANA_CAUSAL_REPO)):
        pytest.skip(f"dev is already imported from another checkout ({causal.__file__}); run this file on its own")
    crwm = pytest.importorskip("dev.rwm.diffusion.model.utils.causal_rwm_train")
    return causal, crwm


class Twins:
    """A tiny live causal policy and the mirror with the same weights (fp64), plus one random episode."""

    def __init__(self, *, stride=4, chunk=32, window=None, separate=True, chunks=4, depth=4, seed=11, fp32_attention=False):
        causal, self.crwm = _live_modules()
        Live = causal.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal
        self.stride, self.chunk, self.window, self.separate = stride, chunk, window, separate
        self.f_chunk = chunk // (8 * stride)
        self.width = VIEW[0] * VIEW[1]
        torch.manual_seed(7)
        kwargs = dict(
            depth=depth, hidden_size=64, patch_size=(1, 1, 1), num_heads=2, in_channels=C_LAT, pred_sigma=False,
            caption_channels=TEXT_CH, model_max_length=TEXT_LEN, linear_head_dim=32, softmax_head_dim=32,
            softmax_ratio=0.5, attn_res_block_size=2, use_time_conditioning=False, y_norm=True, cross_norm=True,
        )
        options = {"actions_per_chunk": {str(int(FPS)): chunk}, "obs_in_first_chunk": True}
        if window is not None:
            options["sliding_window_chunks"] = window
        config = _Cfg(
            model=_Cfg(extra={"chunk_causal_policy": options, "rope": "aligned"}),
            scheduler=_Cfg(action_flow_shift=1.0 if separate else None),
            data=_Cfg(extra={"multiview": "sana_pixel"}),
            vae=_Cfg(vae_stride=[8, 32, 32]),
        )
        self.live = Live(config=config, **kwargs)
        assert self.live.obs_in_first_chunk and self.live.sliding_window_chunks == window
        with torch.no_grad():
            for parameter in self.live.parameters():
                parameter.uniform_(-0.05, 0.05)
        self.live = self.live.double().eval()
        contract = CausalContract(
            actions_per_chunk=chunk, video_frame_stride=stride, latent_frames_per_chunk=self.f_chunk,
            sliding_window_chunks=window, obs_in_first_chunk=True, model_fps=FPS, tier_num_frames=chunk + 1,
            multiview="sana_pixel", visual_layout="sana_pixel_canvas", rope="aligned", dt_bias_init=-5.0,
            a_log_init=0.0, reset_legacy_beta=False,
        )
        self.mirror = CausalPolicyModel(PolicyConfig.from_sana_kwargs(config=config, **kwargs), contract)
        state = {k: v for k, v in self.live.state_dict().items() if k not in ("pos_embed", "y_embedder.y_embedding", "plucker_embed.weight")}
        result = self.mirror.load_state_dict(state, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        self.mirror = self.mirror.double().eval()
        # Both sides run softmax attention in fp32 when fp32_attention is set (the recipe's value); CPU SDPA then
        # rounds by memory layout, so the exact-logic checks run the attention in fp64 on both sides.
        for model in (self.live, self.mirror):
            for block in model.blocks:
                block.attn.fp32_attention = bool(fp32_attention)

        g = torch.Generator().manual_seed(seed)

        def randn(*shape):
            return torch.randn(*shape, generator=g, dtype=torch.float64)

        self.chunks = chunks
        self.obs = randn(1, C_LAT, 1, 1, self.width)
        self.clean = [randn(1, C_LAT, self.f_chunk, 1, self.width) for _ in range(chunks)]
        self.noisy = [randn(1, C_LAT, self.f_chunk, 1, self.width) for _ in range(chunks)]
        self.a_clean = [randn(1, chunk, 80) for _ in range(chunks)]
        self.a_noisy = [randn(1, chunk, 80) for _ in range(chunks)]
        self.states = [randn(1, 80) for _ in range(chunks)]
        self.t_video = [float(900 - 97 * c) for c in range(chunks)]
        self.t_action = [float(950 - 61 * c) for c in range(chunks)] if separate else list(self.t_video)
        self.y = randn(1, 1, 1, TEXT_LEN, TEXT_CH)
        self.mask = torch.ones(1, 1, TEXT_LEN, dtype=torch.int16)
        self.mask[..., 5:] = 0

    def grid(self, strip: torch.Tensor) -> torch.Tensor:
        return strip.reshape(*strip.shape[:3], *VIEW)

    def info(self) -> dict:
        return {
            "model_fps": FPS,
            "view_count": torch.tensor([1]),
            "view_latent_shape": torch.tensor([[list(VIEW)]]),
            "video_frame_stride": torch.tensor([self.stride]),
            "view_slot_ids": torch.tensor([0]),
        }

    def two_stream(self, length: int):
        """Sana's two-stream training forward: every noisy chunk's (video [1, C, f, H, W], action [1, C, 80])."""

        crwm, f_c, chunk = self.crwm, self.f_chunk, self.chunk
        x = torch.cat([self.obs] + self.clean[: length - 1] + self.noisy[:length], dim=2)
        t_video = torch.zeros(1, x.shape[2], dtype=torch.float64)
        noisy_start = 1 + (length - 1) * f_c
        for c in range(length):
            t_video[:, noisy_start + c * f_c : noisy_start + (c + 1) * f_c] = self.t_video[c]
        rows = torch.cat(self.a_clean[: length - 1] + self.a_noisy[:length], dim=1)
        row_t = torch.cat(
            [torch.zeros(1, (length - 1) * chunk, dtype=torch.float64)]
            + [torch.full((1, chunk), self.t_action[c], dtype=torch.float64) for c in range(length)],
            dim=1,
        )
        grid = 2 * length - 1
        info = self.info()
        info.update(
            rwm_task="policy",
            action80=crwm.insert_state_slots(rows, grid, chunk),
            action_mask80=crwm.insert_state_slots(torch.ones(1, grid * chunk, 80, dtype=torch.bool), grid, chunk),
            action_timestep=crwm.insert_state_slots(row_t, grid, chunk),
            initial_state80=self.states[0],
            initial_state_condition_mask80=torch.ones(1, 80, dtype=torch.bool),
            two_stream_num_chunks=length,
        )
        if grid > 1:
            info["chunk_states80"] = torch.stack(self.states[1 : length - 1] + self.states[:length], dim=1)
            info["chunk_states_mask80"] = torch.ones(1, grid - 1, 80, dtype=torch.bool)
        with torch.no_grad():
            out = self.live(x, t_video, self.y, mask=self.mask, data_info=info)
        positions = crwm.action_row_positions(grid, chunk)
        return [
            (
                self.grid(out["x"][:, :, noisy_start + k * f_c : noisy_start + (k + 1) * f_c]),
                out["action_pred"].index_select(1, positions[(length - 1 + k) * chunk : (length + k) * chunk]),
            )
            for k in range(length)
        ]

    def robot(self, k: int, *, noisy: bool) -> dict:
        return {
            "state80": self.states[k],
            "state_mask80": torch.ones(1, 80, dtype=torch.bool),
            "action80": self.a_noisy[k] if noisy else self.a_clean[k],
            "action_mask80": torch.ones(1, self.chunk, 80, dtype=torch.bool),
            "action_timesteps": torch.full((1, self.chunk), self.t_action[k] if noisy else 0.0, dtype=torch.float64),
        }

    def streaming(self, length: int, rule: str = "training"):
        """The deploy chain on the mirror: prefill, then per chunk the denoising window and the clean commit."""

        mirror, f_c, chunk = self.mirror, self.f_chunk, self.chunk
        memory = CausalMemory(mirror, self.window, rule)
        caches = memory.layer_caches(commit=True)
        mirror.forward_window(
            self.grid(self.obs), torch.zeros(1, 1, dtype=torch.float64), self.y, self.mask,
            model_fps=FPS, frame_offset=0, step_offset=0, robot=None, caches=caches,
        )
        memory.absorb(caches)
        outputs = []
        for k in range(length):
            t_video = torch.full((1, f_c), self.t_video[k], dtype=torch.float64)
            if k == 0:
                out = mirror.forward_window(
                    self.grid(torch.cat((self.obs, self.noisy[0]), dim=2)),
                    torch.cat((torch.zeros(1, 1, dtype=torch.float64), t_video), dim=1),
                    self.y, self.mask, model_fps=FPS, frame_offset=0, step_offset=0,
                    robot=self.robot(0, noisy=True), caches=None,
                )
                outputs.append((out["x"][:, :, 1:], out["action_pred"]))
            else:
                out = mirror.forward_window(
                    self.grid(self.noisy[k]), t_video, self.y, self.mask, model_fps=FPS,
                    frame_offset=k * f_c + 1, step_offset=k * chunk,
                    robot=self.robot(k, noisy=True), caches=memory.layer_caches(commit=False),
                )
                outputs.append((out["x"], out["action_pred"]))
            if k < length - 1:
                caches = memory.layer_caches(commit=True)
                mirror.forward_window(
                    self.grid(self.clean[k]), torch.zeros(1, f_c, dtype=torch.float64), self.y, self.mask,
                    model_fps=FPS, frame_offset=k * f_c + 1, step_offset=k * chunk,
                    robot=self.robot(k, noisy=False), caches=caches,
                )
                memory.absorb(caches)
        return outputs


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


@pytest.mark.parametrize("separate", [True, False], ids=["split-shift", "shared-shift"])
@pytest.mark.parametrize("stride,chunk", [(4, 32), (1, 16)], ids=["s4c32", "s1c16"])
def test_the_chunk0_window_is_the_live_one_chunk_forward(stride, chunk, separate):
    twins = Twins(stride=stride, chunk=chunk, separate=separate, chunks=1)
    (x_live, a_live), = twins.two_stream(1)
    (x_ours, a_ours), = twins.streaming(1)
    torch.testing.assert_close(x_ours, x_live, **CLOSE)
    torch.testing.assert_close(a_ours, a_live, **CLOSE)


@pytest.mark.parametrize(
    "window,length",
    [(None, 4), (2, 6), (3, 6), (1, 4)],
    ids=["no-window-L4", "N2-L6", "N3-L6", "N1-L4"],
)
@pytest.mark.parametrize("stride,chunk", [(4, 32), (1, 16)], ids=["s4c32", "s1c16"])
def test_deploy_chain_equals_two_stream_training_forward(stride, chunk, window, length):
    """Every noisy chunk of the training forward == the streaming chain's chunk (incl. the observation eviction)."""

    twins = Twins(stride=stride, chunk=chunk, window=window, chunks=length)
    training = twins.two_stream(length)
    streaming = twins.streaming(length)
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming)):
        torch.testing.assert_close(x_ours, x_live, **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")


def test_the_observation_entry_matters_until_it_slides_out():
    """With N = 2 the observation is read by chunks 0..1 and gone from chunk 2 on, in training and in deploy."""

    twins = Twins(window=2, chunks=4)
    reference = twins.streaming(4)
    twins.obs = twins.obs + 0.5
    moved = twins.streaming(4)
    assert _max_abs(moved[1][0], reference[1][0]) > 1e-6
    # chunk 3 reads entries {chunk 1, chunk 2} through the softmax layers; the GDN state still carries the
    # observation (it never evicts), so the output moves, but the softmax read set no longer contains it
    memory = CausalMemory(twins.mirror, 2, "training")
    memory.entries = 4
    assert memory.read_entry_ids() == [2, 3]
    keep = CausalMemory(twins.mirror, 3, "keep_obs")
    keep.entries = 5
    assert keep.read_entry_ids() == [0, 3, 4]


def test_keep_obs_rule_matches_the_training_rule_while_the_observation_fits():
    twins = Twins(window=3, chunks=3)
    a = twins.streaming(3, "training")
    b = twins.streaming(3, "keep_obs")
    for (xa, aa), (xb, ab) in zip(a, b):
        assert torch.equal(xa, xb) and torch.equal(aa, ab)


def test_layer_cache_gdn_commit_is_the_live_scan():
    """One GDN shell over three committed segments == Sana's gdn_scan over the same three chunks."""

    _live_modules()
    from dev.rwm.diffusion.model.layers.chunk_causal_gdn import CachedChunkCausalPolicyGDNAttention

    from sana_wam_causal.layers import CausalGDNAttention

    torch.manual_seed(3)
    live = CachedChunkCausalPolicyGDNAttention(
        64, 64, heads=2, dim=32, eps=1e-8, use_bias=False, qk_norm=True, norm_eps=1e-5, chunk_size=1,
        chunk_split_strategy="first_frame", conv_kernel_size=0, k_conv_only=True, output_gate_act="sigmoid",
        use_o_norm=True, key_scale_mode="dim_spatial",
    ).double().eval()
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.uniform_(-0.3, 0.3)
    ours = CausalGDNAttention(64, 64, heads=2).double().eval()
    ours.load_state_dict(live.state_dict(), strict=True)
    sizes = [5, 7, 6]
    x = torch.randn(1, sum(sizes), 64, dtype=torch.float64)
    bounds = [0, 5, 12, 18]
    with torch.no_grad():
        expected = live(x, HW=(x.shape[1], 1, 1), chunk_index=bounds)
        cache = LayerCache(commit=True)
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            cache = LayerCache(commit=True, gdn_state=cache.gdn_next if lo else None)
            pieces.append(ours(x[:, lo:hi], cache=cache))
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=1e-12, atol=1e-15)


def test_fp32_attention_chain_stays_within_float32_round_off():
    twins = Twins(window=2, chunks=5, fp32_attention=True)
    for (x_live, a_live), (x_ours, a_ours) in zip(twins.two_stream(5), twins.streaming(5)):
        torch.testing.assert_close(x_ours, x_live, **FP32_ATTENTION_CLOSE)
        torch.testing.assert_close(a_ours, a_live, **FP32_ATTENTION_CLOSE)


def test_the_wrong_window_rule_is_caught():
    """Negative control: keeping the observation entry past the window (Sana's deploy KV manager) does NOT reproduce
    the training forward once the observation has slid out -- the parity test above can tell the rules apart."""

    twins = Twins(window=2, chunks=5)
    training = twins.two_stream(5)
    keep_obs = twins.streaming(5, "keep_obs")
    for k in (0, 1):
        torch.testing.assert_close(keep_obs[k][0], training[k][0], **CLOSE)
    gaps = [_max_abs(keep_obs[k][0], training[k][0]) for k in (2, 3, 4)]
    assert min(gaps) > 1e-6, gaps


@pytest.mark.parametrize("video_cfg,action_cfg", [(1.0, 1.0), (3.0, 1.0), (2.5, 1.7)], ids=["nocfg", "video-cfg", "both-cfg"])
def test_session_matches_the_live_deploy_session(video_cfg, action_cfg):
    """generate / commit cycles of the mirror's session == Sana's CausalPolicyChunkSession (all entries read, i.e.
    within the window), sampling with per-stream CFG on separate video / action flow-shift schedules."""

    from dev.rwm.deploy.causal.session import CausalPolicyChunkSession

    from sana_wam_causal.session import Caption, CausalPolicySession

    twins = Twins(chunks=4)
    twins.live.sliding_window_chunks = None   # the live session's chunk-0 window runs through forward (no spans rule)
    steps, chunks = 3, 4
    y_null = torch.randn(1, 1, 1, TEXT_LEN, TEXT_CH, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    template = {k: v for k, v in twins.info().items()}
    live = CausalPolicyChunkSession(
        twins.live, actions_per_chunk=twins.chunk, caption_embeds=twins.y, caption_mask=twins.mask, steps=steps,
        flow_shift=5.0, data_info=template, unconditional_caption_embeds=y_null, unconditional_caption_mask=twins.mask,
        cfg_scale=video_cfg, action_flow_shift=1.0, action_cfg_scale=action_cfg, kv_max_length=8,
    )
    ours = CausalPolicySession(
        twins.mirror, caption=Caption(twins.y, twins.mask), unconditional_caption=Caption(y_null, twins.mask),
        steps=steps, flow_shift=5.0, action_flow_shift=1.0, video_cfg_scale=video_cfg, action_cfg_scale=action_cfg,
        model_fps=FPS, window=None, autocast_dtype=None,
    )
    live.commit_observation(twins.obs)
    ours.commit_observation(twins.grid(twins.obs))
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    mask[..., 60:] = False
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    g = torch.Generator().manual_seed(17)
    for k in range(chunks):
        video_noise = torch.randn(1, C_LAT, twins.f_chunk, 1, twins.width, generator=g, dtype=torch.float64)
        action_noise = torch.randn(1, twins.chunk, 80, generator=g, dtype=torch.float64)
        a_live, v_live = live.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
            video_noise=video_noise, action_noise=action_noise,
        )
        a_ours, v_ours = ours.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
            video_noise=twins.grid(video_noise), action_noise=action_noise,
        )
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")
        torch.testing.assert_close(v_ours, twins.grid(v_live), **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        live.commit_chunk(twins.clean[k], twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
        ours.commit_chunk(twins.grid(twins.clean[k]), twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
    assert ours.chunk_idx == live.chunk_idx == chunks
