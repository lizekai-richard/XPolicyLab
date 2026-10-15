"""Live parity of the causal mirror against Sana's chunk-causal policy on the episode protocol (``SANA_CAUSAL_REPO``,
default ~/zekail/Sana_episode_0081aa702: Sana at or after 4bd1b674c, the deploy side of the episode canvas).

The live class and the mirror get identical random weights (tiny dims, fp64, CPU, xformers off, ``flash_attn`` bound to
an fp64 reference varlen as Sana's own tests do). The decisive check is the teacher-forced deploy contract: the
two-stream TRAINING forward over [sink | clean targets 0..L-2 | noisy chunks 0..L-1] -- the first-observation sink, the
softmax sliding window with the sink taken next to an evicting reader's window, both memory contents, both
``chunk_observation`` settings, the one-view canvas and the three-view strip -- must equal, chunk by chunk from chunk 1
on, what the streaming chain computes: the sink commit, the commit of the frozen chunk 0, then for every chunk its
denoising window against the memory and the commit of its clean chunk. On a first-frame window
(``aligned_phase_window: first_frame``, Sana 04c8acbee) it must equal from chunk 0 on the chain that opens with chunk 0:
reading the sink without chunk observations (Sana's ``two_stream_first_reads_sink``), nothing with them. The mirror's session is also checked against
Sana's deploy ``CausalPolicyChunkSession`` past the window boundary, on both of its position clocks. Run on its own
(the first file of a session owns the ``dev`` package).
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
from sana_wam_causal.session import Caption, CausalMemory, CausalPolicySession
from sana_wam_min.policy_model.config import PolicyConfig

SANA_CAUSAL_REPO = os.path.expanduser(os.environ.get("SANA_CAUSAL_REPO", "~/zekail/Sana_episode_0081aa702"))
FPS = 16.0
C_LAT = 4
VIEW = (3, 4)
SLOTS = {1: (0,), 3: (0, 2, 3)}
TEXT_LEN, TEXT_CH = 7, 16
MEMORIES = [MEMORY_VIDEO_AND_ACTION, MEMORY_VIDEO_ONLY]
OBSERVATIONS = [True, False]
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
        if qs.shape[0] == 0:
            continue
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
    live_class = causal.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal
    if not hasattr(crwm, "two_stream_sink_segments") or not hasattr(live_class, "forward_sink"):
        pytest.skip(f"{SANA_CAUSAL_REPO} predates the episode protocol's deploy side (d37817788 / 4bd1b674c)")
    return causal, crwm


class Twins:
    """A tiny live causal policy and the mirror with the same weights (fp64), plus one random episode: the sink (chunk
    0's observation), every chunk's own observation latent, clean targets and noisy targets."""

    def __init__(self, *, views=1, stride=4, chunk=32, window=None, separate=True, chunks=4, depth=4, seed=11,
                 fp32_attention=False, memory=MEMORY_VIDEO_AND_ACTION, observation=True, first_frame=False):
        causal, self.crwm = _live_modules()
        Live = causal.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal
        self.views, self.stride, self.chunk, self.window = views, stride, chunk, window
        self.separate, self.memory, self.observation = separate, memory, observation
        self.f_chunk = chunk // (8 * stride)
        self.width = views * VIEW[0] * VIEW[1]
        self.slots = SLOTS[views]
        self.groups = 1 if views == 1 else views + 1
        torch.manual_seed(7)
        kwargs = dict(
            depth=depth, hidden_size=64, patch_size=(1, 1, 1), num_heads=2, in_channels=C_LAT, pred_sigma=False,
            caption_channels=TEXT_CH, model_max_length=TEXT_LEN, linear_head_dim=32, softmax_head_dim=32,
            softmax_ratio=0.5, attn_res_block_size=2, use_time_conditioning=False, y_norm=True, cross_norm=True,
        )
        options = {"actions_per_chunk": {str(int(FPS)): chunk}, "memory_content": memory, "chunk_observation": observation}
        if window is not None:
            options["sliding_window_chunks"] = window
        multiview = "sana_pixel" if views == 1 else "sana_latent"
        config = _Cfg(
            model=_Cfg(
                extra={"chunk_causal_policy": options, "rope": "aligned"},
                multiview_spatial_rope_tile_shape=(8, 16),
            ),
            scheduler=_Cfg(action_flow_shift=1.0 if separate else None),
            data=_Cfg(extra={"multiview": multiview}),
            vae=_Cfg(vae_stride=[8, 32, 32]),
        )
        self.live = Live(config=config, **kwargs)
        assert self.live.memory_content == memory and self.live.sliding_window_chunks == window
        assert self.live.chunk_observation == observation
        assert tuple(self.live.multiview_spatial_rope_tile_shape) == (8, 16)
        with torch.no_grad():
            for parameter in self.live.parameters():
                parameter.uniform_(-0.05, 0.05)
        self.live = self.live.double().eval()
        contract = CausalContract(
            actions_per_chunk=chunk, video_frame_stride=stride, latent_frames_per_chunk=self.f_chunk,
            sliding_window_chunks=window, memory_content=memory, chunk_observation=observation, model_fps=FPS,
            tier_num_frames=chunk + 1, multiview=multiview,
            visual_layout="sana_pixel_canvas" if views == 1 else "three_view_strip", rope="aligned",
            dt_bias_init=-5.0, a_log_init=0.0, reset_legacy_beta=False,
            aligned_phase_window="first_frame" if first_frame else "padded",
        )
        policy_config = PolicyConfig.from_sana_kwargs(config=config, **kwargs)
        assert tuple(policy_config.multiview_spatial_rope_tile_shape) == (8, 16)
        self.mirror = CausalPolicyModel(policy_config, contract)
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
        self.obs = [randn(1, C_LAT, 1, 1, self.width) for _ in range(chunks)]     # obs[0] is the sink
        self.clean = [randn(1, C_LAT, self.f_chunk, 1, self.width) for _ in range(chunks)]
        self.noisy = [randn(1, C_LAT, self.f_chunk, 1, self.width) for _ in range(chunks)]
        self.a_clean = [randn(1, chunk, 80) for _ in range(chunks)]
        self.a_noisy = [randn(1, chunk, 80) for _ in range(chunks)]
        self.states = [randn(1, 80) for _ in range(chunks)]
        self.t_video = [float(900 - 97 * c) for c in range(chunks)]
        self.t_action = [float(950 - 61 * c) for c in range(chunks)] if separate else list(self.t_video)
        self.y = randn(1, self.groups, 1, TEXT_LEN, TEXT_CH)
        self.y_null = randn(1, self.groups, 1, TEXT_LEN, TEXT_CH)
        self.mask = torch.ones(1, self.groups, TEXT_LEN, dtype=torch.int16)
        self.mask[..., 5:] = 0

    @property
    def video_and_action(self) -> bool:
        return self.memory == MEMORY_VIDEO_AND_ACTION

    @property
    def view_shapes(self):
        return None if self.views == 1 else (VIEW,) * self.views

    def ours(self, strip: torch.Tensor) -> torch.Tensor:
        """The mirror's layout of a strip latent: the one-view grid, or the strip itself for several views."""

        return strip.reshape(*strip.shape[:3], *VIEW) if self.views == 1 else strip

    def info(self) -> dict:
        return {
            "model_fps": FPS,
            "view_count": torch.tensor([self.views]),
            "view_latent_shape": torch.tensor([[list(VIEW)] * self.views]),
            "video_frame_stride": torch.tensor([self.stride]),
            "view_slot_ids": torch.tensor(list(self.slots)),
        }

    def two_stream(self, length: int):
        """Sana's two-stream training forward over [sink | clean 0..L-2 | noisy 0..L-1]: every noisy chunk's (video
        targets in the mirror's layout, action [1, C, 80])."""

        crwm, f_c, chunk = self.crwm, self.f_chunk, self.chunk
        obs = int(self.observation)
        f_noisy = obs + f_c
        units = [torch.cat(([self.obs[c]] if obs else []) + [self.noisy[c]], dim=2) for c in range(length)]
        x = torch.cat([self.obs[0]] + self.clean[: length - 1] + units, dim=2)
        t_video = torch.zeros(1, x.shape[2], dtype=torch.float64)
        noisy_start = 1 + (length - 1) * f_c
        for c in range(length):
            lo = noisy_start + c * f_noisy + obs
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
            lo = noisy_start + k * f_noisy + obs
            rows_k = positions[(memory + k) * chunk : (memory + k + 1) * chunk]
            chunks.append((self.ours(out["x"][:, :, lo : lo + f_c]), out["action_pred"].index_select(1, rows_k)))
        return chunks

    def robot(self, k: int, *, noisy: bool) -> dict:
        return {
            "state80": self.states[k],
            "state_mask80": torch.ones(1, 80, dtype=torch.bool),
            "action80": self.a_noisy[k] if noisy else self.a_clean[k],
            "action_mask80": torch.ones(1, self.chunk, 80, dtype=torch.bool),
            "action_timesteps": torch.full((1, self.chunk), self.t_action[k] if noisy else 0.0, dtype=torch.float64),
        }

    def session(self, window="model", steps=1, video_cfg=1.0, action_cfg=1.0) -> CausalPolicySession:
        session = CausalPolicySession(
            self.mirror, caption=Caption(self.y, self.mask), unconditional_caption=Caption(self.y_null, self.mask),
            steps=steps, flow_shift=5.0, action_flow_shift=1.0, video_cfg_scale=video_cfg, action_cfg_scale=action_cfg,
            model_fps=FPS, window=self.window if window == "model" else window, view_shapes=self.view_shapes,
            view_slot_ids=self.slots, autocast_dtype=None,
        )
        return session

    def commit(self, session: CausalPolicySession, k: int) -> None:
        mask = torch.ones(1, self.chunk, 80, dtype=torch.bool)
        session.commit_chunk(
            self.ours(self.clean[k]), self.a_clean[k], anchor_state=self.states[k],
            anchor_state_mask=torch.ones(1, 80, dtype=torch.bool), action_mask=mask,
        )

    def denoise_forward(self, session: CausalPolicySession, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward of chunk k's denoising window against the session memory (what every solver step of
        ``generate_chunk`` runs), on the given noisy latents and timesteps."""

        f_c, obs = self.f_chunk, int(self.observation)
        latent = torch.cat(([self.ours(self.obs[k])] if obs else []) + [self.ours(self.noisy[k])], dim=2)
        t_window = torch.cat(
            (torch.zeros(1, obs, dtype=torch.float64), torch.full((1, f_c), self.t_video[k], dtype=torch.float64)), dim=1
        )
        memory = session.memories[0]
        caches = session._caches(memory, commit=False)
        out = session._forward(
            latent, t_window, session.captions[0], memory, frame_offset=k * f_c + (0 if obs else 1),
            step_offset=k * self.chunk, robot=self.robot(k, noisy=True), caches=caches,
        )
        return out["x"][:, :, obs:], out["action_pred"]

    def streaming(self, length: int, window="model"):
        """The deploy chain on the mirror: the sink, the frozen chunk 0, then per chunk k >= 1 its denoising window
        reading the memory and the commit of clean chunk k. ``window`` overrides the memory's read window (a negative
        control). Returns chunks 1..L-1."""

        session = self.session(window=window)
        session.commit_sink(self.ours(self.obs[0]))
        self.commit(session, 0)
        outputs = []
        for k in range(1, length):
            outputs.append(self.denoise_forward(session, k))
            if k < length - 1:
                self.commit(session, k)
        return outputs

    def streaming_first_frame(self, length: int, sink_at_chunk_0=None):
        """The deploy chain of a first-frame window on the mirror: chunk 0's denoising window at the first frame --
        after the sink without chunk observations (it reads it), before it with them (it reads nothing) -- then per
        chunk the commit of the previous clean chunk and its own window. ``sink_at_chunk_0`` overrides whether chunk 0
        reads the sink (a negative control). Returns chunks 0..L-1."""

        session = self.session()
        assert session.first_chunk == 0
        reads_sink = (not self.observation) if sink_at_chunk_0 is None else bool(sink_at_chunk_0)
        if reads_sink:
            session.commit_sink(self.ours(self.obs[0]))
        outputs = [self.denoise_forward(session, 0)]
        if not reads_sink:
            session.commit_sink(self.ours(self.obs[0]))
        for k in range(1, length):
            self.commit(session, k - 1)
            outputs.append(self.denoise_forward(session, k))
        return outputs


def _first_reads_sink_in_training() -> bool:
    """Whether the live tree's two-stream forward lets noisy chunk 0 read the sink without chunk observations (04c8acbee)."""

    import inspect

    _live_modules()
    from dev.rwm.diffusion.model.layers import chunk_causal_gdn

    return "first_reads_sink" in inspect.signature(chunk_causal_gdn.two_stream_read_version).parameters


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


CASES = [
    # views, stride, chunk, window, length
    (1, 4, 32, None, 4),
    (1, 4, 32, 1, 5),
    (1, 4, 32, 2, 6),
    (1, 1, 16, 3, 6),
    (3, 4, 32, None, 4),
    (3, 4, 32, 2, 6),
]
CASE_IDS = ["V1-s4c32-no-window-L4", "V1-s4c32-W1-L5", "V1-s4c32-W2-L6", "V1-s1c16-W3-L6", "V3-s4c32-no-window-L4", "V3-s4c32-W2-L6"]


@pytest.mark.parametrize("observation", OBSERVATIONS, ids=["obs", "noobs"])
@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize("views,stride,chunk,window,length", CASES, ids=CASE_IDS)
def test_deploy_chain_equals_two_stream_training_forward(views, stride, chunk, window, length, memory, observation):
    """Every noisy chunk k >= 1 of the training forward == the streaming chain's chunk k, past the softmax window
    (the evicting readers take the sink next to their window) included."""

    twins = Twins(views=views, stride=stride, chunk=chunk, window=window, chunks=length, memory=memory,
                  observation=observation)
    training = twins.two_stream(length)[1:]
    streaming = twins.streaming(length)
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming), start=1):
        torch.testing.assert_close(x_ours, x_live, **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")


@pytest.mark.parametrize("observation", OBSERVATIONS, ids=["obs", "noobs"])
@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize("views,stride,chunk,window,length", CASES, ids=CASE_IDS)
def test_first_frame_deploy_chain_equals_two_stream_training_forward(views, stride, chunk, window, length, memory,
                                                                     observation):
    """aligned_phase_window first_frame: EVERY noisy chunk of the training forward, chunk 0 included, == the deploy
    chain that opens with chunk 0 at the first frame (past the softmax window too)."""

    if not observation and not _first_reads_sink_in_training():
        pytest.skip(f"{SANA_CAUSAL_REPO} predates noisy chunk 0 reading the sink without chunk observations (04c8acbee)")
    twins = Twins(views=views, stride=stride, chunk=chunk, window=window, chunks=length, memory=memory,
                  observation=observation, first_frame=True)
    training = twins.two_stream(length)
    streaming = twins.streaming_first_frame(length)
    assert len(training) == len(streaming) == length
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming)):
        torch.testing.assert_close(x_ours, x_live, **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")


@pytest.mark.parametrize("views", [1, 3])
def test_a_first_frame_chunk_0_without_the_sink_is_caught(views):
    """Negative control: without chunk observations, a chunk 0 that does not read the sink (noisy chunk 0's rule before
    04c8acbee) misses the training forward at chunk 0 -- the parity above can tell whether chunk 0 reads the sink."""

    if not _first_reads_sink_in_training():
        pytest.skip(f"{SANA_CAUSAL_REPO} predates noisy chunk 0 reading the sink without chunk observations (04c8acbee)")
    twins = Twins(views=views, window=2, chunks=4, memory=MEMORY_VIDEO_ONLY, observation=False, first_frame=True)
    training = twins.two_stream(4)
    off = twins.streaming_first_frame(4, sink_at_chunk_0=False)
    for index in (0, 1):   # video, action
        with pytest.raises(AssertionError):
            torch.testing.assert_close(off[0][index], training[0][index], **CLOSE)
        assert _max_abs(off[0][index], training[0][index]) > 1e-8


@pytest.mark.parametrize("views", [1, 3])
def test_a_pinned_sink_is_caught_on_the_evicting_chunks(views, monkeypatch):
    """Negative control: a sink kept at its own frame 0 (no StreamingLLM move) matches training while nothing slid
    out of the window and misses the fp64 tolerance on every evicting reader -- the parity above can tell the sink's
    position (the tiny random model attends weakly, so the gap is ~1e-7 here; Sana measured 5e-2 on real weights)."""

    window, length = 2, 6
    twins = Twins(views=views, window=window, chunks=length)
    training = twins.two_stream(length)[1:]
    original = CausalPolicySession._sink_rope

    def pinned(self):
        rope = original(self)
        if rope is None:
            return None
        origin = self.softmax_origin()
        return self.model.softmax_entry_rope(
            frames=1, frame_offset=-origin * self.f_chunk, robot_rows=0, step_offset=0, model_fps=self.model_fps,
            view_shapes=self._views(), view_slot_ids=self.view_slot_ids, device=rope.device,
        )

    monkeypatch.setattr(CausalPolicySession, "_sink_rope", pinned)
    off = twins.streaming(length)
    for k in range(1, window + 1):   # reads clean 0..k-1, nothing evicted
        torch.testing.assert_close(off[k - 1][0], training[k - 1][0], **CLOSE)
    for k in range(window + 1, length):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(off[k - 1][0], training[k - 1][0], **CLOSE)
    gaps = [_max_abs(off[k - 1][0], training[k - 1][0]) for k in range(window + 1, length)]
    assert min(gaps) > 1e-8, gaps


@pytest.mark.parametrize("memory", MEMORIES)
def test_an_observation_conditions_its_own_chunk_only(memory):
    """obs_k (k >= 1) enters chunk k's window and never the memory: moving it moves chunk k alone, in training and in
    deploy; the sink (obs_0) conditions every chunk."""

    twins = Twins(window=2, chunks=4, memory=memory)
    reference = twins.streaming(4)
    twins.obs[2] = twins.obs[2] + 0.5
    moved = twins.streaming(4)
    training = twins.two_stream(4)[1:]
    assert _max_abs(moved[1][0], reference[1][0]) > 1e-6 and _max_abs(moved[1][1], reference[1][1]) > 1e-6
    for index in (0, 2):
        assert torch.equal(moved[index][0], reference[index][0]) and torch.equal(moved[index][1], reference[index][1])
    for (x_live, a_live), (x_ours, a_ours) in zip(training, moved):
        torch.testing.assert_close(x_ours, x_live, **CLOSE)
        torch.testing.assert_close(a_ours, a_live, **CLOSE)
    twins.obs[0] = twins.obs[0] + 0.5
    sunk = twins.streaming(4)
    assert all(_max_abs(sunk[i][0], moved[i][0]) > 1e-6 for i in range(3))


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


@pytest.mark.parametrize("observation", OBSERVATIONS, ids=["obs", "noobs"])
@pytest.mark.parametrize("views", [1, 3])
def test_fp32_attention_chain_stays_within_float32_round_off(views, observation):
    twins = Twins(views=views, window=2, chunks=5, fp32_attention=True, memory=MEMORY_VIDEO_ONLY, observation=observation)
    for (x_live, a_live), (x_ours, a_ours) in zip(twins.two_stream(5)[1:], twins.streaming(5)):
        torch.testing.assert_close(x_ours, x_live, **FP32_ATTENTION_CLOSE)
        torch.testing.assert_close(a_ours, a_live, **FP32_ATTENTION_CLOSE)


@pytest.mark.parametrize("wrong", [1, 3], ids=["one-chunk-short", "one-chunk-long"])
def test_a_wrong_read_window_is_caught(wrong):
    """Negative control: reading one committed chunk fewer or more than training does matches the training forward only
    while the history fits both windows -- the parity test above can tell read windows apart."""

    twins = Twins(window=2, chunks=6)
    training = twins.two_stream(6)[1:]
    off = twins.streaming(6, window=wrong)
    fits = min(2, wrong)
    for k in range(1, fits + 1):
        torch.testing.assert_close(off[k - 1][0], training[k - 1][0], **CLOSE)
    gaps = [_max_abs(off[k - 1][0], training[k - 1][0]) for k in range(fits + 1, 6)]
    assert min(gaps) > 1e-6, gaps


def _live_session(twins, steps, video_cfg, action_cfg, position_mode):
    from dev.rwm.deploy.causal.session import CausalPolicyChunkSession

    return CausalPolicyChunkSession(
        twins.live, actions_per_chunk=twins.chunk, caption_embeds=twins.y, caption_mask=twins.mask, steps=steps,
        flow_shift=5.0, data_info=dict(twins.info()), unconditional_caption_embeds=twins.y_null,
        unconditional_caption_mask=twins.mask, cfg_scale=video_cfg, action_flow_shift=1.0, action_cfg_scale=action_cfg,
        kv_max_length=twins.chunks + 2, kv_position_mode=position_mode,
    )


SESSION_CASES = [
    # views, memory, observation, video_cfg, action_cfg, position mode
    (1, MEMORY_VIDEO_AND_ACTION, True, 1.0, 1.0, "rank"),
    (1, MEMORY_VIDEO_AND_ACTION, True, 3.0, 1.0, "absolute"),
    (1, MEMORY_VIDEO_ONLY, True, 2.5, 1.7, "rank"),
    (1, MEMORY_VIDEO_ONLY, False, 3.0, 1.0, "absolute"),
    (3, MEMORY_VIDEO_ONLY, True, 3.0, 1.0, "absolute"),
    (3, MEMORY_VIDEO_ONLY, True, 2.5, 1.7, "rank"),
    (3, MEMORY_VIDEO_ONLY, False, 3.0, 1.0, "rank"),
    (3, MEMORY_VIDEO_AND_ACTION, False, 2.5, 1.7, "absolute"),
]


@pytest.mark.parametrize("views,memory,observation,video_cfg,action_cfg,position_mode", SESSION_CASES)
def test_session_matches_the_live_deploy_session(views, memory, observation, video_cfg, action_cfg, position_mode):
    """The episode of the mirror's session == Sana's CausalPolicyChunkSession (its default read: W - 1 older chunks +
    the most recent one, besides the sink): the sink and frozen-chunk commits, then generate / commit cycles from chunk
    1 with per-stream CFG on separate video / action flow-shift schedules, past the window boundary (W = 2, 6 chunks).
    The mirror numbers every softmax read from 0 at its oldest chunk with the sink at frame 0 (Sana's ``rank`` clock);
    Sana's ``absolute`` clock differs by one common shift, which the aligned RoPE does not see."""

    steps, chunks = 3, 6
    twins = Twins(views=views, window=2, chunks=chunks, memory=memory, observation=observation)
    live = _live_session(twins, steps, video_cfg, action_cfg, position_mode)
    ours = twins.session(steps=steps, video_cfg=video_cfg, action_cfg=action_cfg)
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    mask[..., 60:] = False
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    live.commit_sink(twins.obs[0])
    ours.commit_sink(twins.ours(twins.obs[0]))
    for side, layout in ((live, lambda t: t), (ours, twins.ours)):
        side.commit_chunk(layout(twins.clean[0]), twins.a_clean[0], anchor_state=twins.states[0],
                          anchor_state_mask=state_mask, action_mask=mask)
    g = torch.Generator().manual_seed(17)
    origins = []
    for k in range(1, chunks):
        origins.append(ours.softmax_origin())
        video_noise = torch.randn(1, C_LAT, twins.f_chunk, 1, twins.width, generator=g, dtype=torch.float64)
        action_noise = torch.randn(1, twins.chunk, 80, generator=g, dtype=torch.float64)
        a_live, v_live = live.generate_chunk(
            twins.obs[k] if observation else None, anchor_state=twins.states[k], anchor_state_mask=state_mask,
            action_mask=mask, video_noise=video_noise, action_noise=action_noise,
        )
        a_ours, v_ours = ours.generate_chunk(
            twins.ours(twins.obs[k]) if observation else None, anchor_state=twins.states[k], anchor_state_mask=state_mask,
            action_mask=mask, video_noise=twins.ours(video_noise), action_noise=action_noise,
        )
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")
        torch.testing.assert_close(v_ours, twins.ours(v_live), **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        live.commit_chunk(twins.clean[k], twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
        ours.commit_chunk(twins.ours(twins.clean[k]), twins.a_clean[k], anchor_state=twins.states[k],
                          anchor_state_mask=state_mask, action_mask=mask)
    assert ours.chunk_idx == live.chunk_idx == chunks
    assert origins == [max(0, k - 2) for k in range(1, chunks)]


def _live_session_opens_on_the_first_frame() -> bool:
    """Whether the live tree's deploy side starts on the first-frame window (9fca24879: ``CausalViewStream`` lost its
    frozen chunk, hence ``frames_per_chunk``)."""

    import inspect

    _live_modules()
    from dev.rwm.deploy.causal import view_stream

    return "frames_per_chunk" not in inspect.signature(view_stream.CausalViewStream.__init__).parameters


FIRST_FRAME_SESSION_CASES = [
    # views, memory, observation, video_cfg, action_cfg, position mode
    (1, MEMORY_VIDEO_ONLY, False, 3.0, 1.0, "rank"),
    (3, MEMORY_VIDEO_ONLY, False, 2.5, 1.7, "absolute"),
    (3, MEMORY_VIDEO_ONLY, True, 3.0, 1.0, "rank"),
    (1, MEMORY_VIDEO_AND_ACTION, True, 2.5, 1.7, "absolute"),
    (3, MEMORY_VIDEO_AND_ACTION, False, 3.0, 1.0, "rank"),
]


@pytest.mark.parametrize("views,memory,observation,video_cfg,action_cfg,position_mode", FIRST_FRAME_SESSION_CASES)
def test_a_first_frame_session_matches_the_live_deploy_session(views, memory, observation, video_cfg, action_cfg,
                                                              position_mode):
    """A first-frame episode of the mirror's session == Sana's CausalPolicyChunkSession, which starts there since
    9fca24879: chunk 0 generated at the first frame (after the sink without chunk observations, before it with them),
    then generate / commit cycles with per-stream CFG past the window boundary (W = 2, 6 chunks)."""

    if not _live_session_opens_on_the_first_frame():
        pytest.skip(f"{SANA_CAUSAL_REPO}'s deploy session predates the first-frame start (9fca24879)")
    steps, chunks = 3, 6
    twins = Twins(views=views, window=2, chunks=chunks, memory=memory, observation=observation, first_frame=True)
    live = _live_session(twins, steps, video_cfg, action_cfg, position_mode)
    ours = twins.session(steps=steps, video_cfg=video_cfg, action_cfg=action_cfg)
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    mask[..., 60:] = False
    state_mask = torch.ones(1, 80, dtype=torch.bool)

    def commit_sink():
        live.commit_sink(twins.obs[0])
        ours.commit_sink(twins.ours(twins.obs[0]))

    if not observation:
        commit_sink()
    g = torch.Generator().manual_seed(19)
    for k in range(chunks):
        video_noise = torch.randn(1, C_LAT, twins.f_chunk, 1, twins.width, generator=g, dtype=torch.float64)
        action_noise = torch.randn(1, twins.chunk, 80, generator=g, dtype=torch.float64)
        a_live, v_live = live.generate_chunk(
            twins.obs[k] if observation else None, anchor_state=twins.states[k], anchor_state_mask=state_mask,
            action_mask=mask, video_noise=video_noise, action_noise=action_noise,
        )
        a_ours, v_ours = ours.generate_chunk(
            twins.ours(twins.obs[k]) if observation else None, anchor_state=twins.states[k], anchor_state_mask=state_mask,
            action_mask=mask, video_noise=twins.ours(video_noise), action_noise=action_noise,
        )
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")
        torch.testing.assert_close(v_ours, twins.ours(v_live), **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        if k == 0 and observation:
            commit_sink()
        live.commit_chunk(twins.clean[k], twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
        ours.commit_chunk(twins.ours(twins.clean[k]), twins.a_clean[k], anchor_state=twins.states[k],
                          anchor_state_mask=state_mask, action_mask=mask)
    assert ours.chunk_idx == live.chunk_idx == chunks


def test_the_softmax_clock_stays_in_a_training_window_and_the_gdn_clock_is_physical():
    """Every rotary table the mirror builds during a 7-chunk episode with W = 3: the softmax layers' positions stay
    inside a training window (the sink at frame 0, the read's chunks in slots 0..W-1, the forward's own chunk at most in
    slot W), the GDN layers' windows run on the physical clock of the episode (the sink at frame 0)."""

    window, length = 3, 7
    twins = Twins(window=window, chunks=length)
    mirror, f_c, chunk = twins.mirror, twins.f_chunk, twins.chunk
    calls = []
    original = mirror.window_rope

    def recording(rope, fps, frames, view_shapes, view_slot_ids, robot_rows, frame_offset, step_offset, device):
        calls.append((rope is mirror.rope_softmax, frames, robot_rows, frame_offset, step_offset))
        return original(rope, fps, frames, view_shapes, view_slot_ids, robot_rows, frame_offset, step_offset, device)

    mirror.window_rope = recording
    twins.streaming(length)
    soft = [c for c in calls if c[0]]
    gdn = [c for c in calls if not c[0]]
    assert max(c[3] + c[1] - 1 for c in soft) == (window + 1) * f_c     # the last target frame of slot W
    assert min(c[3] for c in soft) == 0                                  # the sink (and chunk 0's slot)
    assert max(c[4] for c in soft if c[2]) == window * chunk             # the state row of slot W
    assert max(c[3] for c in gdn) == (length - 1) * f_c                  # GDN: the last chunk's observation, physical
    assert max(c[4] for c in gdn) == (length - 1) * chunk
    assert (False, 1, 0, 0, 0) in calls                                  # the sink's own GDN window at frame 0


def test_the_session_refuses_a_broken_opening():
    twins = Twins(chunks=3)
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    session = twins.session()
    with pytest.raises(RuntimeError, match="commit_sink first"):
        twins.commit(session, 0)
    with pytest.raises(RuntimeError, match="first emission is chunk 1"):
        session.generate_chunk(twins.ours(twins.obs[0]), anchor_state=twins.states[0], anchor_state_mask=state_mask,
                               action_mask=mask)
    session.commit_sink(twins.ours(twins.obs[0]))
    with pytest.raises(RuntimeError, match="once, before the first chunk"):
        session.commit_sink(twins.ours(twins.obs[0]))
    with pytest.raises(RuntimeError, match="first emission is chunk 1"):
        session.generate_chunk(twins.ours(twins.obs[0]), anchor_state=twins.states[0], anchor_state_mask=state_mask,
                               action_mask=mask)
    twins.commit(session, 0)
    with pytest.raises(ValueError, match=r"must be \[1, C, 1"):
        session.generate_chunk(None, anchor_state=twins.states[1], anchor_state_mask=state_mask, action_mask=mask)
    noobs = Twins(chunks=3, observation=False)
    session = noobs.session()
    session.commit_sink(noobs.ours(noobs.obs[0]))
    noobs.commit(session, 0)
    with pytest.raises(ValueError, match="observation_latent=None"):
        session.generate_chunk(noobs.ours(noobs.obs[1]), anchor_state=noobs.states[1], anchor_state_mask=state_mask,
                               action_mask=mask)
    session.reset_memory()
    with pytest.raises(RuntimeError, match="first emission is chunk 1"):
        session.generate_chunk(None, anchor_state=noobs.states[1], anchor_state_mask=state_mask, action_mask=mask)
