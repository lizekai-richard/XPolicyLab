"""Live parity of the causal mirror against Sana's chunk-causal policy (``SANA_CAUSAL_REPO``, default ~/zekail/Sana, at
or after 3a6b2bf07: the per-chunk observation protocol).

The live class and the mirror get identical random weights (tiny dims, fp64, CPU, xformers off, ``flash_attn`` bound to
an fp64 reference varlen as Sana's own tests do). The decisive check is the teacher-forced deploy contract: the
two-stream TRAINING forward over [clean targets 0..L-2 | noisy (obs_c | targets_c) 0..L-1] -- the softmax sliding window
and both memory contents included -- must equal, chunk by chunk, what the streaming chain computes (for every chunk a
denoising window [obs_c | targets_c] against the memory, then the commit of the clean chunk). The mirror's session is
also checked against Sana's deploy ``CausalPolicyChunkSession`` across the window boundary, on both of its position
clocks. Run on its own (the first file of a session owns the ``dev`` package).
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import _paths  # noqa: F401

from sana_wam_causal.contract import MEMORY_VIDEO_AND_ACTION, MEMORY_VIDEO_ONLY, CausalContract
from sana_wam_causal.model import CausalPolicyModel
from sana_wam_causal.session import CausalMemory
from sana_wam_min.policy_model.config import PolicyConfig

SANA_CAUSAL_REPO = os.path.expanduser(os.environ.get("SANA_CAUSAL_REPO", "~/zekail/Sana"))
FPS = 16.0
C_LAT = 4
VIEW = (3, 4)
TEXT_LEN, TEXT_CH = 7, 16
MEMORIES = [MEMORY_VIDEO_AND_ACTION, MEMORY_VIDEO_ONLY]
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
    if not hasattr(crwm, "memory_row_chunks"):
        pytest.skip(f"{SANA_CAUSAL_REPO} predates the per-chunk observation protocol (61aad2d58 / 388d8aaf5)")
    return causal, crwm


class Twins:
    """A tiny live causal policy and the mirror with the same weights (fp64), plus one random episode whose every chunk
    has its own observation latent."""

    def __init__(self, *, stride=4, chunk=32, window=None, separate=True, chunks=4, depth=4, seed=11,
                 fp32_attention=False, memory=MEMORY_VIDEO_AND_ACTION):
        causal, self.crwm = _live_modules()
        Live = causal.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal
        self.stride, self.chunk, self.window, self.separate, self.memory = stride, chunk, window, separate, memory
        self.f_chunk = chunk // (8 * stride)
        self.width = VIEW[0] * VIEW[1]
        torch.manual_seed(7)
        kwargs = dict(
            depth=depth, hidden_size=64, patch_size=(1, 1, 1), num_heads=2, in_channels=C_LAT, pred_sigma=False,
            caption_channels=TEXT_CH, model_max_length=TEXT_LEN, linear_head_dim=32, softmax_head_dim=32,
            softmax_ratio=0.5, attn_res_block_size=2, use_time_conditioning=False, y_norm=True, cross_norm=True,
        )
        options = {"actions_per_chunk": {str(int(FPS)): chunk}, "memory_content": memory}
        if window is not None:
            options["sliding_window_chunks"] = window
        config = _Cfg(
            model=_Cfg(extra={"chunk_causal_policy": options, "rope": "aligned"}),
            scheduler=_Cfg(action_flow_shift=1.0 if separate else None),
            data=_Cfg(extra={"multiview": "sana_pixel"}),
            vae=_Cfg(vae_stride=[8, 32, 32]),
        )
        self.live = Live(config=config, **kwargs)
        assert self.live.memory_content == memory and self.live.sliding_window_chunks == window
        with torch.no_grad():
            for parameter in self.live.parameters():
                parameter.uniform_(-0.05, 0.05)
        self.live = self.live.double().eval()
        contract = CausalContract(
            actions_per_chunk=chunk, video_frame_stride=stride, latent_frames_per_chunk=self.f_chunk,
            sliding_window_chunks=window, memory_content=memory, model_fps=FPS, tier_num_frames=chunk + 1,
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
        self.obs = [randn(1, C_LAT, 1, 1, self.width) for _ in range(chunks)]
        self.clean = [randn(1, C_LAT, self.f_chunk, 1, self.width) for _ in range(chunks)]
        self.noisy = [randn(1, C_LAT, self.f_chunk, 1, self.width) for _ in range(chunks)]
        self.a_clean = [randn(1, chunk, 80) for _ in range(chunks)]
        self.a_noisy = [randn(1, chunk, 80) for _ in range(chunks)]
        self.states = [randn(1, 80) for _ in range(chunks)]
        self.t_video = [float(900 - 97 * c) for c in range(chunks)]
        self.t_action = [float(950 - 61 * c) for c in range(chunks)] if separate else list(self.t_video)
        self.y = randn(1, 1, 1, TEXT_LEN, TEXT_CH)
        self.y_null = randn(1, 1, 1, TEXT_LEN, TEXT_CH)
        self.mask = torch.ones(1, 1, TEXT_LEN, dtype=torch.int16)
        self.mask[..., 5:] = 0

    @property
    def video_and_action(self) -> bool:
        return self.memory == MEMORY_VIDEO_AND_ACTION

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
        """Sana's two-stream training forward: every noisy chunk's (video targets [1, C, f, H, W], action [1, C, 80])."""

        crwm, f_c, chunk = self.crwm, self.f_chunk, self.chunk
        units = [torch.cat([self.obs[c], self.noisy[c]], dim=2) for c in range(length)]
        x = torch.cat(self.clean[: length - 1] + units, dim=2)
        t_video = torch.zeros(1, x.shape[2], dtype=torch.float64)
        noisy_start = (length - 1) * f_c
        for c in range(length):
            lo = noisy_start + c * (1 + f_c) + 1
            t_video[:, lo : lo + f_c] = self.t_video[c]
        memory = crwm.memory_row_chunks(length, self.memory)
        rows = torch.cat(self.a_clean[:memory] + self.a_noisy[:length], dim=1)
        row_t = torch.cat(
            [torch.zeros(1, memory * chunk, dtype=torch.float64)]
            + [torch.full((1, chunk), self.t_action[c], dtype=torch.float64) for c in range(length)],
            dim=1,
        )
        anchors = self.states[:memory] + self.states[:length]
        grid = len(anchors)
        info = self.info()
        info.update(
            rwm_task="policy",
            action80=crwm.insert_state_slots(rows, grid, chunk),
            action_mask80=crwm.insert_state_slots(torch.ones(1, grid * chunk, 80, dtype=torch.bool), grid, chunk),
            action_timestep=crwm.insert_state_slots(row_t, grid, chunk),
            initial_state80=anchors[0],
            initial_state_condition_mask80=torch.ones(1, 80, dtype=torch.bool),
            two_stream_num_chunks=length,
        )
        if grid > 1:
            info["chunk_states80"] = torch.stack(anchors[1:], dim=1)
            info["chunk_states_mask80"] = torch.ones(1, grid - 1, 80, dtype=torch.bool)
        with torch.no_grad():
            out = self.live(x, t_video, self.y, mask=self.mask, data_info=info)
        positions = crwm.action_row_positions(grid, chunk)
        chunks = []
        for k in range(length):
            lo = noisy_start + k * (1 + f_c) + 1
            rows_k = positions[(memory + k) * chunk : (memory + k + 1) * chunk]
            chunks.append((self.grid(out["x"][:, :, lo : lo + f_c]), out["action_pred"].index_select(1, rows_k)))
        return chunks

    def robot(self, k: int, *, noisy: bool) -> dict:
        return {
            "state80": self.states[k],
            "state_mask80": torch.ones(1, 80, dtype=torch.bool),
            "action80": self.a_noisy[k] if noisy else self.a_clean[k],
            "action_mask80": torch.ones(1, self.chunk, 80, dtype=torch.bool),
            "action_timesteps": torch.full((1, self.chunk), self.t_action[k] if noisy else 0.0, dtype=torch.float64),
        }

    def streaming(self, length: int, window="model"):
        """The deploy chain on the mirror: per chunk the denoising window [obs_k | targets_k] reading the memory, then
        the clean commit of chunk k. ``window`` overrides the memory's read window (a negative control)."""

        mirror, f_c, chunk = self.mirror, self.f_chunk, self.chunk
        memory = CausalMemory(mirror, self.window if window == "model" else window)

        def rope_of(slot):
            return mirror.softmax_entry_rope(slot, FPS, VIEW[0], VIEW[1], torch.device("cpu"))

        def offsets(k, first_frame):   # chunk k's window on both clocks (softmax: numbered from 0 at the read's origin)
            origin = memory.read_plan()[1]
            return {
                "frame_offset": first_frame, "step_offset": k * chunk,
                "softmax_frame_offset": first_frame - origin * f_c, "softmax_step_offset": (k - origin) * chunk,
            }

        outputs = []
        for k in range(length):
            caches = None if memory.committed == 0 else memory.layer_caches(commit=False, rope_of=rope_of)
            t_window = torch.cat(
                (torch.zeros(1, 1, dtype=torch.float64), torch.full((1, f_c), self.t_video[k], dtype=torch.float64)), dim=1
            )
            out = mirror.forward_window(
                self.grid(torch.cat((self.obs[k], self.noisy[k]), dim=2)), t_window, self.y, self.mask,
                model_fps=FPS, robot=self.robot(k, noisy=True), caches=caches, **offsets(k, k * f_c),
            )
            outputs.append((out["x"][:, :, 1:], out["action_pred"]))
            if k < length - 1:
                where = offsets(k, k * f_c + 1)
                caches = memory.layer_caches(commit=True, rope_of=rope_of)
                mirror.forward_window(
                    self.grid(self.clean[k]), torch.zeros(1, f_c, dtype=torch.float64), self.y, self.mask,
                    model_fps=FPS, robot=self.robot(k, noisy=False) if self.video_and_action else None, caches=caches,
                    **where,
                )
                memory.absorb(caches)
        return outputs


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize("separate", [True, False], ids=["split-shift", "shared-shift"])
@pytest.mark.parametrize("stride,chunk", [(4, 32), (1, 16)], ids=["s4c32", "s1c16"])
def test_the_first_chunk_is_the_live_one_chunk_forward(stride, chunk, separate, memory):
    twins = Twins(stride=stride, chunk=chunk, separate=separate, chunks=1, memory=memory)
    (x_live, a_live), = twins.two_stream(1)
    (x_ours, a_ours), = twins.streaming(1)
    torch.testing.assert_close(x_ours, x_live, **CLOSE)
    torch.testing.assert_close(a_ours, a_live, **CLOSE)


@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize(
    "window,length",
    [(None, 4), (1, 4), (2, 6), (3, 6)],
    ids=["no-window-L4", "W1-L4", "W2-L6", "W3-L6"],
)
@pytest.mark.parametrize("stride,chunk", [(4, 32), (1, 16)], ids=["s4c32", "s1c16"])
def test_deploy_chain_equals_two_stream_training_forward(stride, chunk, window, length, memory):
    """Every noisy chunk of the training forward == the streaming chain's chunk, past the softmax window included."""

    twins = Twins(stride=stride, chunk=chunk, window=window, chunks=length, memory=memory)
    training = twins.two_stream(length)
    streaming = twins.streaming(length)
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming)):
        torch.testing.assert_close(x_ours, x_live, **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")


@pytest.mark.parametrize("memory", MEMORIES)
def test_an_observation_conditions_its_own_chunk_only(memory):
    """obs_k enters chunk k's window and never the memory: moving it moves chunk k alone, in training and in deploy."""

    twins = Twins(window=2, chunks=4, memory=memory)
    reference = twins.streaming(4)
    twins.obs[1] = twins.obs[1] + 0.5
    moved = twins.streaming(4)
    training = twins.two_stream(4)
    assert _max_abs(moved[1][0], reference[1][0]) > 1e-6 and _max_abs(moved[1][1], reference[1][1]) > 1e-6
    for k in (0, 2, 3):
        assert torch.equal(moved[k][0], reference[k][0]) and torch.equal(moved[k][1], reference[k][1])
    for (x_live, a_live), (x_ours, a_ours) in zip(training, moved):
        torch.testing.assert_close(x_ours, x_live, **CLOSE)
        torch.testing.assert_close(a_ours, a_live, **CLOSE)


def test_a_video_only_memory_holds_no_action_rows():
    """The live training forward reads the clean chunks' actions under video_and_action and ignores them under
    video_only (the chain never commits them there), so the parity above covers what each memory holds."""

    for memory, reads_rows in ((MEMORY_VIDEO_AND_ACTION, True), (MEMORY_VIDEO_ONLY, False)):
        twins = Twins(window=3, chunks=3, memory=memory)
        reference = twins.two_stream(3)
        twins.a_clean[0] = twins.a_clean[0] + 0.5
        moved = twins.two_stream(3)
        gap = max(_max_abs(moved[k][i], reference[k][i]) for k in (1, 2) for i in (0, 1))
        assert (gap > 1e-6) == reads_rows, (memory, gap)


def test_layer_cache_gdn_commit_is_the_live_scan():
    """One GDN shell over three committed segments == Sana's gdn_scan over the same three chunks."""

    _live_modules()
    from dev.rwm.diffusion.model.layers.chunk_causal_gdn import CachedChunkCausalPolicyGDNAttention

    from sana_wam_causal.layers import CausalGDNAttention, LayerCache

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


@pytest.mark.parametrize("memory", MEMORIES)
def test_fp32_attention_chain_stays_within_float32_round_off(memory):
    twins = Twins(window=2, chunks=5, fp32_attention=True, memory=memory)
    for (x_live, a_live), (x_ours, a_ours) in zip(twins.two_stream(5), twins.streaming(5)):
        torch.testing.assert_close(x_ours, x_live, **FP32_ATTENTION_CLOSE)
        torch.testing.assert_close(a_ours, a_live, **FP32_ATTENTION_CLOSE)


@pytest.mark.parametrize("wrong", [1, 3], ids=["one-chunk-short", "one-chunk-long"])
def test_a_wrong_read_window_is_caught(wrong):
    """Negative control: reading one committed chunk fewer or more than training does matches the training forward only
    while the history fits both windows -- the parity test above can tell read windows apart."""

    twins = Twins(window=2, chunks=5)
    training = twins.two_stream(5)
    off = twins.streaming(5, window=wrong)
    fits = min(2, wrong)
    for k in range(fits + 1):
        torch.testing.assert_close(off[k][0], training[k][0], **CLOSE)
    gaps = [_max_abs(off[k][0], training[k][0]) for k in range(fits + 1, 5)]
    assert min(gaps) > 1e-6, gaps


def _live_session(twins, steps, video_cfg, action_cfg, position_mode):
    from dev.rwm.deploy.causal.session import CausalPolicyChunkSession

    return CausalPolicyChunkSession(
        twins.live, actions_per_chunk=twins.chunk, caption_embeds=twins.y, caption_mask=twins.mask, steps=steps,
        flow_shift=5.0, data_info=dict(twins.info()), unconditional_caption_embeds=twins.y_null,
        unconditional_caption_mask=twins.mask, cfg_scale=video_cfg, action_flow_shift=1.0, action_cfg_scale=action_cfg,
        kv_max_length=twins.chunks + 2, kv_position_mode=position_mode,
    )


def _our_session(twins, steps, video_cfg, action_cfg):
    from sana_wam_causal.session import Caption, CausalPolicySession

    return CausalPolicySession(
        twins.mirror, caption=Caption(twins.y, twins.mask), unconditional_caption=Caption(twins.y_null, twins.mask),
        steps=steps, flow_shift=5.0, action_flow_shift=1.0, video_cfg_scale=video_cfg, action_cfg_scale=action_cfg,
        model_fps=FPS, window=twins.window, autocast_dtype=None,
    )


@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize("position_mode", ["rank", "absolute"])
@pytest.mark.parametrize("video_cfg,action_cfg", [(1.0, 1.0), (3.0, 1.0), (2.5, 1.7)], ids=["nocfg", "video-cfg", "both-cfg"])
def test_session_matches_the_live_deploy_session(video_cfg, action_cfg, position_mode, memory):
    """generate / commit cycles of the mirror's session == Sana's CausalPolicyChunkSession (its default read: W - 1 older
    chunks + the most recent one), sampling with per-stream CFG on separate video / action flow-shift schedules, past
    the window boundary (W = 2, 5 chunks). The mirror numbers every softmax read from 0 at its oldest chunk (Sana's
    ``rank`` clock); Sana's ``absolute`` clock differs by one common shift, which the aligned RoPE does not see."""

    steps, chunks = 3, 5
    twins = Twins(window=2, chunks=chunks, memory=memory)
    live = _live_session(twins, steps, video_cfg, action_cfg, position_mode)
    ours = _our_session(twins, steps, video_cfg, action_cfg)
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    mask[..., 60:] = False
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    g = torch.Generator().manual_seed(17)
    origins = []
    for k in range(chunks):
        origins.append(ours.softmax_origin())
        video_noise = torch.randn(1, C_LAT, twins.f_chunk, 1, twins.width, generator=g, dtype=torch.float64)
        action_noise = torch.randn(1, twins.chunk, 80, generator=g, dtype=torch.float64)
        a_live, v_live = live.generate_chunk(
            twins.obs[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
            video_noise=video_noise, action_noise=action_noise,
        )
        a_ours, v_ours = ours.generate_chunk(
            twins.grid(twins.obs[k]), anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
            video_noise=twins.grid(video_noise), action_noise=action_noise,
        )
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")
        torch.testing.assert_close(v_ours, twins.grid(v_live), **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        live.commit_chunk(twins.clean[k], twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
        ours.commit_chunk(twins.grid(twins.clean[k]), twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
    assert ours.chunk_idx == live.chunk_idx == chunks
    assert origins == [max(0, k - 2) for k in range(chunks)]


def test_the_softmax_clock_starts_at_zero_and_the_gdn_clock_is_physical():
    """Every rotary table the mirror builds during a 6-chunk chain with W = 3: the softmax layers' positions stay inside
    a training window (the read's chunks in slots 0..W-1, the forward's own chunk at most in slot W), the GDN layers'
    windows run on the physical clock of the episode."""

    window, length = 3, 6
    twins = Twins(window=window, chunks=length)
    mirror, f_c, chunk = twins.mirror, twins.f_chunk, twins.chunk
    calls = []
    original = mirror.window_rope

    def recording(rope, fps, frames, height, width, robot_rows, frame_offset, step_offset, device):
        calls.append((rope is mirror.rope_softmax, frames, robot_rows, frame_offset, step_offset))
        return original(rope, fps, frames, height, width, robot_rows, frame_offset, step_offset, device)

    mirror.window_rope = recording
    twins.streaming(length)
    soft = [c for c in calls if c[0]]
    gdn = [c for c in calls if not c[0]]
    assert max(c[3] + c[1] - 1 for c in soft) == (window + 1) * f_c     # the last target frame of slot W
    assert min(c[3] for c in soft) == 0                                  # chunk 0's observation
    assert max(c[4] for c in soft if c[2]) == window * chunk             # the state row of slot W
    assert max(c[3] for c in gdn) == (length - 1) * f_c                  # GDN: the last chunk's observation, physical
    assert max(c[4] for c in gdn) == (length - 1) * chunk
