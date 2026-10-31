"""Live parity of the causal mirror against Sana's chunk-causal policy on the episode protocol in its single form
(``SANA_CAUSAL_REPO``: Sana rwm/zekai-merge at or after 4bd6eca83, checked at 7de7711c2, b297028fe, c4d1d246d,
ff004392a, 9a1b3e33d and 7ed0d28bf; no chunk observations, the sink on every canvas). The plain-residual tests
(``model.use_attn_res: false``) need a tree at or after 53234167c and skip on an older one. A tree from 0c88c11ab on
reads the softmax sink at chunk max(0, c - 7) on the physical clock only: the sink-horizon tests run on it and every
test of the window-start sink skips, and the other way round on an older tree, so the file runs once per rule (e.g.
9a1b3e33d, then 7ed0d28bf). The numerator-only GDN readout (``model.memory_gdn_output_normalizer:
false``) needs a tree at or after c414e9b32; those tests skip on an older one. The observation tests need a tree at or
after fd5261ff6 (the GDN branch of the observation group, sana-rwm's two-stream obs-block) and skip on an older one.

The live class and the mirror get identical random weights (tiny dims, fp64, CPU, xformers off, ``flash_attn`` bound to
an fp64 reference varlen as Sana's own tests do). The decisive check is the teacher-forced deploy contract: the
two-stream TRAINING forward over [sink | clean targets 0..L-2 | noisy targets 0..L-1] -- the sink (the first frame),
every noisy chunk reading its observation from the memory, the softmax sliding window with the sink taken next to an
evicting reader's window, both memory contents, the one-view canvas and the three-view strip -- must equal, chunk by
chunk from chunk 1 on, what the streaming chain computes: the sink commit, the commit of the frozen chunk 0, then for
every chunk its denoising window against the memory and the commit of its clean chunk. On a first-frame window
(``aligned_phase_window: first_frame``, Sana 04c8acbee) it must equal from chunk 0 on the chain that opens with chunk 0
after the sink, which it reads as its observation. The mirror's session is also checked against Sana's deploy
``CausalPolicyChunkSession`` past the window boundary, on both of its position clocks. Run on its own (the first file of
a session owns the ``dev`` package).
"""

from __future__ import annotations

import inspect
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

SANA_CAUSAL_REPO = os.path.expanduser(os.environ.get("SANA_CAUSAL_REPO", "~/zekail/Sana_zekai_merge_9a1b3e33d"))
FPS = 16.0
C_LAT = 4
VIEW = (3, 4)
SLOTS = {1: (0,), 3: (0, 2, 3)}
TEXT_LEN, TEXT_CH = 7, 16
MEMORIES = [MEMORY_VIDEO_AND_ACTION, MEMORY_VIDEO_ONLY]
CLOSE = {"rtol": 1e-9, "atol": 1e-12}            # fp64 everywhere (fp32 attention off on both sides)
FP32_ATTENTION_CLOSE = {"rtol": 1e-5, "atol": 1e-8}  # fp32 attention on: CPU SDPA kernels pick layouts freely
# causal_observation: the denoise read splits the noisy segment's sums (its decay pool, its K^T diag(beta) V) between the
# prefill and the window, which moves fp64 round-off; Sana's own observation-session parity uses the same bound
# (tests/test_causal_observation_session.py EXACT)
OBSERVED_CLOSE = {"rtol": 1e-7, "atol": 1e-9}


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
    if not hasattr(crwm, "refuse_chunk_observation") or not hasattr(live_class, "forward_sink"):
        pytest.skip(f"{SANA_CAUSAL_REPO} predates the single form without chunk observations (4bd6eca83)")
    return causal, crwm


def _require_live_gdn_normalizer():
    """Skip unless the live tree has the numerator-only GDN switch (Sana c414e9b32: ``_readout(..., normalizer)``)."""

    _live_modules()
    from dev.rwm.diffusion.model.layers import chunk_causal_gdn

    if "normalizer" not in inspect.signature(chunk_causal_gdn._readout).parameters:
        pytest.skip(f"{SANA_CAUSAL_REPO} predates model.memory_gdn_output_normalizer (c414e9b32)")


def _require_live_causal_observation(state: bool = False):
    """Skip unless the live tree serves causal_observation (Sana 764ee2113) and, for ``state``, causal_state
    (afc9874b6) in its cached session, with the GDN branch of the observation group (fd5261ff6), the form this adapter
    serves."""

    _, crwm = _live_modules()
    if not hasattr(crwm, "resolve_causal_observation") or (state and not hasattr(crwm, "resolve_causal_state")):
        pytest.skip(f"{SANA_CAUSAL_REPO} predates causal_observation / causal_state (764ee2113 / afc9874b6)")
    from dev.rwm.diffusion.model.layers import chunk_causal_gdn

    if not hasattr(chunk_causal_gdn, "gdn_observation_branch"):
        pytest.skip(f"{SANA_CAUSAL_REPO} predates the GDN branch of the observation group (fd5261ff6)")


def _require_live_plain_residual():
    """Skip unless the live tree runs the causal block stack without AttnRes (Sana 53234167c, ``use_attn_res`` false)."""

    causal, _ = _live_modules()
    if not hasattr(causal, "_plain_layer_step"):
        pytest.skip(f"{SANA_CAUSAL_REPO} predates use_attn_res false on the causal policy (53234167c)")


def _require_live_physical_rope():
    """Skip unless the live tree has physical_rope (Sana 6e91eaf2a) on its training canvas."""

    _, crwm = _live_modules()
    if not hasattr(crwm, "resolve_physical_rope"):
        pytest.skip(f"{SANA_CAUSAL_REPO} predates physical_rope (6e91eaf2a)")


def _live_sink_horizon() -> bool:
    """Whether the live tree reads the softmax sink at chunk max(0, c - 7) (Sana 0c88c11ab: ``sink_read_chunks``)."""

    _, crwm = _live_modules()
    return hasattr(crwm, "sink_read_chunks")


def _require_live_sink_rule(horizon: bool):
    """Skip unless the live tree reads the softmax sink by the rule asked: at chunk max(0, c - 7) on the physical clock
    (Sana 0c88c11ab on) or at the reader's window start (before it)."""

    if _live_sink_horizon() != bool(horizon):
        rule = "the sink horizon (0c88c11ab on)" if horizon else "the window-start sink (before 0c88c11ab)"
        pytest.skip(f"{SANA_CAUSAL_REPO} does not train {rule}; run this file on such a tree too")


class Twins:
    """A tiny live causal policy and the mirror with the same weights (fp64), plus one random episode: the sink (the
    first frame, chunk 0's observation), clean targets and noisy targets."""

    def __init__(self, *, views=1, stride=4, chunk=32, window=None, separate=True, chunks=4, depth=4, seed=11,
                 fp32_attention=False, memory=MEMORY_VIDEO_AND_ACTION, first_frame=False, normalizer=True,
                 live_normalizer=None, observation=False, state=False, physical=False, live_physical=None,
                 attn_res=True, horizon=False, live_horizon=None):
        causal, self.crwm = _live_modules()
        # the softmax sink rule (Sana 0c88c11ab: at chunk max(0, c - 7), the physical clock only) of the mirror and of
        # the live tree (live_horizon differs only in the negative control)
        self.horizon = bool(horizon)
        live_horizon = self.horizon if live_horizon is None else bool(live_horizon)
        _require_live_sink_rule(live_horizon)
        # the mirror's GDN readout (contract.gdn_output_normalizer) and the live one (model.memory_gdn_output_normalizer;
        # live_normalizer differs only in the negative control)
        live_normalizer = normalizer if live_normalizer is None else live_normalizer
        if not live_normalizer:
            _require_live_gdn_normalizer()
        Live = causal.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal
        self.views, self.stride, self.chunk, self.window = views, stride, chunk, window
        self.separate, self.memory = separate, memory
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
        options = {"actions_per_chunk": {str(int(FPS)): chunk}, "memory_content": memory}
        self.observation, self.state = bool(observation), bool(state)
        if observation:
            _require_live_causal_observation(state)
            options.update(causal_observation=True, causal_state=bool(state))
        # physical_rope (Sana 6e91eaf2a) on the mirror and on the live policy (live_physical differs only in the negative
        # control)
        self.physical = bool(physical) or self.horizon
        live_physical = self.physical if live_physical is None else bool(live_physical)
        if not live_horizon:   # a horizon tree runs the physical clock without the key
            if self.physical or live_physical:
                _require_live_physical_rope()
            if live_physical:
                options["physical_rope"] = True
        if window is not None:
            options["sliding_window_chunks"] = window
        # model.use_attn_res false (Sana 53234167c): both block stacks a plain residual
        self.attn_res = bool(attn_res)
        if not self.attn_res:
            _require_live_plain_residual()
        multiview = "sana_pixel" if views == 1 else "sana_latent"
        model_options = {} if live_normalizer else {"memory_gdn_output_normalizer": False}
        config = _Cfg(
            model=_Cfg(
                extra={"chunk_causal_policy": options, "rope": "aligned"},
                multiview_spatial_rope_tile_shape=(8, 16),
                **model_options,
            ),
            scheduler=_Cfg(action_flow_shift=1.0 if separate else None),
            data=_Cfg(extra={"multiview": multiview}),
            vae=_Cfg(vae_stride=[8, 32, 32]),
        )
        self.live = Live(config=config, use_attn_res=self.attn_res, **kwargs)
        assert (self.live.attn_res is not None) is self.attn_res
        assert self.live.memory_content == memory and self.live.sliding_window_chunks == window
        assert tuple(self.live.multiview_spatial_rope_tile_shape) == (8, 16)
        assert getattr(self.live, "memory_gdn_output_normalizer", True) is live_normalizer
        with torch.no_grad():
            for parameter in self.live.parameters():
                parameter.uniform_(-0.05, 0.05)
        self.live = self.live.double().eval()
        contract = CausalContract(
            actions_per_chunk=chunk, video_frame_stride=stride, latent_frames_per_chunk=self.f_chunk,
            sliding_window_chunks=window, memory_content=memory, model_fps=FPS,
            tier_num_frames=chunk + 1, multiview=multiview,
            visual_layout="sana_pixel_canvas" if views == 1 else "three_view_strip", rope="aligned",
            dt_bias_init=-5.0, a_log_init=0.0, reset_legacy_beta=False,
            aligned_phase_window="first_frame" if first_frame else "padded",
            gdn_output_normalizer=normalizer, causal_observation=self.observation, causal_state=self.state,
            physical_rope=self.physical, use_attn_res=self.attn_res, sink_horizon=7 if self.horizon else None,
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
        self.sink = randn(1, C_LAT, 1, 1, self.width)
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
        # causal_observation: every chunk's own observation latent (its start frame encoded alone), drawn after the
        # rest so the episode above is the same with and without the option
        self.observations = [randn(1, C_LAT, 1, 1, self.width) for _ in range(chunks)]

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
        # under causal_observation noisy chunk c is [o_c | targets], o_c clean (Sana noisy_chunk_frames)
        lead = int(self.observation)
        noisy = [torch.cat([self.observations[c], self.noisy[c]], dim=2) if lead else self.noisy[c] for c in range(length)]
        x = torch.cat([self.sink] + self.clean[: length - 1] + noisy, dim=2)
        t_video = torch.zeros(1, x.shape[2], dtype=torch.float64)
        noisy_start = 1 + (length - 1) * f_c
        for c in range(length):
            lo = noisy_start + c * (lead + f_c) + lead
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
            lo = noisy_start + k * (lead + f_c) + lead
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

        f_c = self.f_chunk
        assert session.chunk_idx == k
        t_window = torch.full((1, f_c), self.t_video[k], dtype=torch.float64)
        memory = session.memories[0]
        if session.observation:
            caches = session.prefill_observation(
                session.captions[0], memory, self.ours(self.observations[k]), state=self.states[k],
                state_mask=torch.ones(1, 80, dtype=torch.bool),
            )
        else:
            caches = session._caches(memory, commit=False)
        frame_offset, step_offset = session.window_offsets()
        robot, form = session.denoise_robot(self.robot(k, noisy=True))
        out = session._forward(
            self.ours(self.noisy[k]), t_window, session.captions[0], memory, frame_offset=frame_offset,
            step_offset=step_offset, robot=robot, caches=caches, robot_form=form,
        )
        return out["x"], out["action_pred"]

    def streaming(self, length: int, window="model"):
        """The deploy chain on the mirror: the sink, the frozen chunk 0, then per chunk k >= 1 its denoising window
        reading the memory and the commit of clean chunk k. ``window`` overrides the memory's read window (a negative
        control). Returns chunks 1..L-1."""

        session = self.session(window=window)
        session.commit_sink(self.ours(self.sink))
        self.commit(session, 0)
        outputs = []
        for k in range(1, length):
            outputs.append(self.denoise_forward(session, k))
            if k < length - 1:
                self.commit(session, k)
        return outputs

    def streaming_first_frame(self, length: int, sink_at_chunk_0=True):
        """The deploy chain of a first-frame window on the mirror: chunk 0's denoising window at the first frame after
        the sink, which it reads as its observation, then per chunk the commit of the previous clean chunk and its own
        window. ``sink_at_chunk_0`` False generates chunk 0 before the sink (a negative control; the session itself
        refuses it, so the forward is driven directly). Returns chunks 0..L-1."""

        session = self.session()
        assert session.first_chunk == 0
        if sink_at_chunk_0:
            session.commit_sink(self.ours(self.sink))
        outputs = [self.denoise_forward(session, 0)]
        if not sink_at_chunk_0:
            session.commit_sink(self.ours(self.sink))
        for k in range(1, length):
            self.commit(session, k - 1)
            outputs.append(self.denoise_forward(session, k))
        return outputs


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


@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize("views,stride,chunk,window,length", CASES, ids=CASE_IDS)
def test_deploy_chain_equals_two_stream_training_forward(views, stride, chunk, window, length, memory):
    """Every noisy chunk k >= 1 of the training forward == the streaming chain's chunk k, past the softmax window
    (the evicting readers take the sink next to their window) included."""

    twins = Twins(views=views, stride=stride, chunk=chunk, window=window, chunks=length, memory=memory)
    training = twins.two_stream(length)[1:]
    streaming = twins.streaming(length)
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming), start=1):
        torch.testing.assert_close(x_ours, x_live, **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")


@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize("views,stride,chunk,window,length", CASES, ids=CASE_IDS)
def test_first_frame_deploy_chain_equals_two_stream_training_forward(views, stride, chunk, window, length, memory):
    """aligned_phase_window first_frame: EVERY noisy chunk of the training forward, chunk 0 included, == the deploy
    chain that opens with chunk 0 at the first frame (past the softmax window too)."""

    twins = Twins(views=views, stride=stride, chunk=chunk, window=window, chunks=length, memory=memory, first_frame=True)
    training = twins.two_stream(length)
    streaming = twins.streaming_first_frame(length)
    assert len(training) == len(streaming) == length
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming)):
        torch.testing.assert_close(x_ours, x_live, **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")


NUMERATOR_CASES = [CASES[0], CASES[2], CASES[5]]
NUMERATOR_IDS = [CASE_IDS[0], CASE_IDS[2], CASE_IDS[5]]


@pytest.mark.parametrize("first_frame", [False, True], ids=["padded", "first-frame"])
@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize("views,stride,chunk,window,length", NUMERATOR_CASES, ids=NUMERATOR_IDS)
def test_numerator_only_deploy_chain_equals_two_stream_training_forward(views, stride, chunk, window, length, memory,
                                                                       first_frame):
    """model.memory_gdn_output_normalizer false (Sana c414e9b32): every GDN layer reads its numerator alone on both
    sides, and the deploy chain still equals the two-stream training forward chunk by chunk."""

    twins = Twins(views=views, stride=stride, chunk=chunk, window=window, chunks=length, memory=memory,
                  first_frame=first_frame, normalizer=False)
    assert all(block.attn.output_normalizer is False for block in twins.mirror.blocks if block.attn_type == "GatedDeltaNet")
    training = twins.two_stream(length) if first_frame else twins.two_stream(length)[1:]
    streaming = twins.streaming_first_frame(length) if first_frame else twins.streaming(length)
    assert len(training) == len(streaming)
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming)):
        torch.testing.assert_close(x_ours, x_live, **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")


@pytest.mark.parametrize("views", [1, 3])
def test_a_mirror_that_divides_is_caught_on_a_numerator_only_policy(views):
    """Negative control: the mirror dividing by the denominator while the live policy reads the numerator alone misses
    the training forward -- the parity above can tell the two readouts apart."""

    twins = Twins(views=views, window=2, chunks=4, memory=MEMORY_VIDEO_ONLY, first_frame=True, live_normalizer=False)
    training = twins.two_stream(4)
    streaming = twins.streaming_first_frame(4)
    gap = max(_max_abs(streaming[k][i], training[k][i]) for k in range(4) for i in (0, 1))
    assert gap > 1e-6, gap


OBSERVATION_CASES = [
    # views, stride, chunk, window, length, memory, first_frame, normalizer
    (1, 4, 32, 1, 5, MEMORY_VIDEO_ONLY, True, False),      # the N1-obs recipe on the canvas
    (3, 4, 32, 1, 5, MEMORY_VIDEO_ONLY, True, False),      # ... on the strip (the released N1-obs checkpoint)
    (3, 4, 32, 2, 6, MEMORY_VIDEO_ONLY, True, True),       # N2-obs before the numerator-only switch
    (1, 4, 32, 2, 6, MEMORY_VIDEO_AND_ACTION, False, True),
    (3, 4, 32, None, 4, MEMORY_VIDEO_AND_ACTION, True, True),
    (1, 1, 16, 3, 6, MEMORY_VIDEO_ONLY, False, False),     # f_c = 2: the rows move C / f_c = 8 steps
]
OBSERVATION_IDS = ["V1-W1-vo-ff-num", "V3-W1-vo-ff-num", "V3-W2-vo-ff", "V1-W2-va-pad", "V3-noW-va-ff", "V1-s1c16-W3-vo-pad-num"]


@pytest.mark.parametrize("state", [False, True], ids=["obs", "obs+state"])
@pytest.mark.parametrize("views,stride,chunk,window,length,memory,first_frame,normalizer", OBSERVATION_CASES,
                         ids=OBSERVATION_IDS)
def test_an_observed_deploy_chain_equals_two_stream_training_forward(views, stride, chunk, window, length, memory,
                                                                    first_frame, normalizer, state):
    """causal_observation (Sana 764ee2113) and causal_state (afc9874b6): every noisy chunk of the training canvas
    [sink | clean | (o_c, targets_c) ...] == the deploy chain's chunk (its observation prefill, then its denoising
    window reading it besides the memory), past the softmax window, on both front-ends, both memory contents, both phase
    windows, with and without the GDN denominator."""

    twins = Twins(views=views, stride=stride, chunk=chunk, window=window, chunks=length, memory=memory,
                  first_frame=first_frame, normalizer=normalizer, observation=True, state=state)
    assert twins.live.causal_observation and twins.live.causal_state is state
    training = twins.two_stream(length) if first_frame else twins.two_stream(length)[1:]
    streaming = twins.streaming_first_frame(length) if first_frame else twins.streaming(length)
    assert len(training) == len(streaming)
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming)):
        torch.testing.assert_close(x_ours, x_live, **OBSERVED_CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **OBSERVED_CLOSE, msg=lambda m: f"chunk {k} action: {m}")


def _observed_window(twins, session, k, *, drop_gdn=False, drop_softmax=False, state_in_window=False, shift=0):
    """Chunk k's denoising window after its observation prefill, with one part of the protocol broken (negative
    controls): the GDN or softmax observation terms dropped, the state row in the window, the window ``shift`` frames
    off its slot."""

    memory = session.memories[0]
    caches = session.prefill_observation(
        session.captions[0], memory, twins.ours(twins.observations[k]), state=twins.states[k],
        state_mask=torch.ones(1, 80, dtype=torch.bool),
    )
    for cache in caches:
        if drop_gdn:
            cache.gdn_observation = None
        if drop_softmax and cache.softmax_context is not None:
            n_obs = twins.width + int(twins.state)
            cache.softmax_context = tuple(t[:, :-n_obs] for t in cache.softmax_context)
    frame_offset, step_offset = session.window_offsets()
    robot = twins.robot(k, noisy=True)
    robot, form = (robot, "chunk") if state_in_window else session.denoise_robot(robot)
    if state_in_window:
        session.model.causal_state = False   # the model would refuse a full chunk window under causal_state
    try:
        out = session._forward(
            twins.ours(twins.noisy[k]), torch.full((1, twins.f_chunk), twins.t_video[k], dtype=torch.float64),
            session.captions[0], memory, frame_offset=frame_offset + shift, step_offset=step_offset, robot=robot,
            caches=caches, robot_form=form,
        )
    finally:
        session.model.causal_state = twins.state
    return out["x"], out["action_pred"]


@pytest.mark.parametrize(
    "broken", ["drop_gdn", "drop_softmax", "state_in_window", "shift"],
    ids=["no-gdn-observation", "no-softmax-observation", "state-in-the-window", "targets-at-the-observation-frame"],
)
def test_a_broken_observation_protocol_is_caught(broken):
    """Negative controls: the parity above misses as soon as the denoise read drops the observation's GDN terms or its
    softmax K/V, carries the state row itself under causal_state, or puts its targets on the observation's frame."""

    twins = Twins(views=3, window=1, chunks=3, memory=MEMORY_VIDEO_ONLY, first_frame=True, observation=True,
                  state=broken == "state_in_window" or broken == "drop_softmax")
    training = twins.two_stream(3)
    session = twins.session()
    session.commit_sink(twins.ours(twins.sink))
    kwargs = {"shift": -1} if broken == "shift" else {broken: True}
    moved = _observed_window(twins, session, 0, **kwargs)
    gap = max(_max_abs(moved[i], training[0][i]) for i in (0, 1))
    assert gap > 1e-6, (broken, gap)


@pytest.mark.parametrize("views", [1, 3])
def test_a_first_frame_chunk_0_without_the_sink_is_caught(views):
    """Negative control: a chunk 0 that does not read the sink (noisy chunk 0's rule with chunk observations, before
    4bd6eca83) misses the training forward at chunk 0 -- the parity above can tell whether chunk 0 reads the sink."""

    twins = Twins(views=views, window=2, chunks=4, memory=MEMORY_VIDEO_ONLY, first_frame=True)
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
def test_a_noisy_chunk_conditions_itself_only_and_the_sink_every_chunk(memory):
    """A chunk's noisy targets never enter the memory (its commit holds the executed chunk's clean targets): moving
    chunk 2's moves chunk 2 alone, in training and in deploy; the sink, the first frame every chunk reads, conditions
    every chunk; and chunk c's observation is in the memory: moving clean chunk c - 1 moves chunk c."""

    twins = Twins(window=2, chunks=4, memory=memory)
    reference = twins.streaming(4)
    twins.noisy[2] = twins.noisy[2] + 0.5
    moved = twins.streaming(4)
    training = twins.two_stream(4)[1:]
    assert _max_abs(moved[1][0], reference[1][0]) > 1e-6 and _max_abs(moved[1][1], reference[1][1]) > 1e-6
    for index in (0, 2):
        assert torch.equal(moved[index][0], reference[index][0]) and torch.equal(moved[index][1], reference[index][1])
    for (x_live, a_live), (x_ours, a_ours) in zip(training, moved):
        torch.testing.assert_close(x_ours, x_live, **CLOSE)
        torch.testing.assert_close(a_ours, a_live, **CLOSE)
    twins.sink = twins.sink + 0.5
    sunk = twins.streaming(4)
    assert all(_max_abs(sunk[i][0], moved[i][0]) > 1e-6 for i in range(3))
    twins.clean[1] = twins.clean[1] + 0.5
    observed = twins.streaming(4)
    assert torch.equal(observed[0][0], sunk[0][0]) and _max_abs(observed[1][0], sunk[1][0]) > 1e-6


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


def test_a_numerator_only_gdn_commit_is_the_live_scan():
    """The same three committed segments with ``output_normalizer`` false on both shells == Sana's gdn_scan reading the
    numerator alone; the carried (S, z) is the default one's (a commit writes z either way)."""

    _require_live_gdn_normalizer()
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
    live.output_normalizer = False
    ours = CausalGDNAttention(64, 64, heads=2, output_normalizer=False).double().eval()
    ours.load_state_dict(live.state_dict(), strict=True)
    divides = CausalGDNAttention(64, 64, heads=2).double().eval()
    divides.load_state_dict(live.state_dict(), strict=True)
    sizes = [5, 7, 6]
    x = torch.randn(1, sum(sizes), 64, dtype=torch.float64)
    bounds = [0, 5, 12, 18]
    with torch.no_grad():
        expected = live(x, HW=(x.shape[1], 1, 1), chunk_index=bounds)
        caches = {"ours": LayerCache(commit=True), "divides": LayerCache(commit=True)}
        pieces = {"ours": [], "divides": []}
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            for name, layer in (("ours", ours), ("divides", divides)):
                caches[name] = LayerCache(commit=True, gdn_state=caches[name].gdn_next if lo else None)
                pieces[name].append(layer(x[:, lo:hi], cache=caches[name]))
    torch.testing.assert_close(torch.cat(pieces["ours"], dim=1), expected, rtol=1e-12, atol=1e-15)
    assert _max_abs(torch.cat(pieces["divides"], dim=1), expected) > 1e-6
    for carried, reference in zip(caches["ours"].gdn_next, caches["divides"].gdn_next):
        torch.testing.assert_close(carried, reference, rtol=0, atol=0)


@pytest.mark.parametrize("views", [1, 3])
def test_fp32_attention_chain_stays_within_float32_round_off(views):
    twins = Twins(views=views, window=2, chunks=5, fp32_attention=True, memory=MEMORY_VIDEO_ONLY)
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
    # views, memory, video_cfg, action_cfg, position mode
    (1, MEMORY_VIDEO_AND_ACTION, 1.0, 1.0, "rank"),
    (1, MEMORY_VIDEO_AND_ACTION, 3.0, 1.0, "absolute"),
    (1, MEMORY_VIDEO_ONLY, 2.5, 1.7, "rank"),
    (1, MEMORY_VIDEO_ONLY, 3.0, 1.0, "absolute"),
    (3, MEMORY_VIDEO_ONLY, 3.0, 1.0, "absolute"),
    (3, MEMORY_VIDEO_ONLY, 2.5, 1.7, "rank"),
    (3, MEMORY_VIDEO_AND_ACTION, 2.5, 1.7, "absolute"),
]


@pytest.mark.parametrize("views,memory,video_cfg,action_cfg,position_mode", SESSION_CASES)
def test_session_matches_the_live_deploy_session(views, memory, video_cfg, action_cfg, position_mode):
    """The episode of the mirror's session == Sana's CausalPolicyChunkSession (its default read: W - 1 older chunks +
    the most recent one, besides the sink): the sink and frozen-chunk commits, then generate / commit cycles from chunk
    1 with per-stream CFG on separate video / action flow-shift schedules, past the window boundary (W = 2, 6 chunks).
    The mirror numbers every softmax read from 0 at its oldest chunk with the sink at frame 0 (Sana's ``rank`` clock);
    Sana's ``absolute`` clock differs by one common shift, which the aligned RoPE does not see."""

    steps, chunks = 3, 6
    twins = Twins(views=views, window=2, chunks=chunks, memory=memory)
    live = _live_session(twins, steps, video_cfg, action_cfg, position_mode)
    ours = twins.session(steps=steps, video_cfg=video_cfg, action_cfg=action_cfg)
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    mask[..., 60:] = False
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    live.commit_sink(twins.sink)
    ours.commit_sink(twins.ours(twins.sink))
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
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask, video_noise=video_noise,
            action_noise=action_noise,
        )
        a_ours, v_ours = ours.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
            video_noise=twins.ours(video_noise), action_noise=action_noise,
        )
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")
        torch.testing.assert_close(v_ours, twins.ours(v_live), **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        live.commit_chunk(twins.clean[k], twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
        ours.commit_chunk(twins.ours(twins.clean[k]), twins.a_clean[k], anchor_state=twins.states[k],
                          anchor_state_mask=state_mask, action_mask=mask)
    assert ours.chunk_idx == live.chunk_idx == chunks
    assert origins == [max(0, k - 2) for k in range(1, chunks)]


FIRST_FRAME_SESSION_CASES = [
    # views, memory, video_cfg, action_cfg, position mode, window
    (1, MEMORY_VIDEO_ONLY, 3.0, 1.0, "rank", 2),
    (3, MEMORY_VIDEO_ONLY, 2.5, 1.7, "absolute", 2),
    (1, MEMORY_VIDEO_ONLY, 3.0, 1.0, "absolute", 1),       # the N1 recipes: the sink and the most recent chunk
    (3, MEMORY_VIDEO_ONLY, 2.5, 1.7, "rank", 1),
    (1, MEMORY_VIDEO_AND_ACTION, 2.5, 1.7, "absolute", 2),
    (3, MEMORY_VIDEO_AND_ACTION, 3.0, 1.0, "rank", 4),     # the N4 recipe's window
]


@pytest.mark.parametrize("views,memory,video_cfg,action_cfg,position_mode,window", FIRST_FRAME_SESSION_CASES)
def test_a_first_frame_session_matches_the_live_deploy_session(views, memory, video_cfg, action_cfg, position_mode,
                                                              window):
    """A first-frame episode of the mirror's session == Sana's CausalPolicyChunkSession (which starts there since
    9fca24879): the sink, chunk 0 generated at the first frame reading it, then generate / commit cycles with per-stream
    CFG past the window boundary (W = 1, 2, 4; 6 chunks)."""

    steps, chunks = 3, 6
    twins = Twins(views=views, window=window, chunks=chunks, memory=memory, first_frame=True)
    live = _live_session(twins, steps, video_cfg, action_cfg, position_mode)
    ours = twins.session(steps=steps, video_cfg=video_cfg, action_cfg=action_cfg)
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    mask[..., 60:] = False
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    live.commit_sink(twins.sink)
    ours.commit_sink(twins.ours(twins.sink))
    g = torch.Generator().manual_seed(19)
    for k in range(chunks):
        video_noise = torch.randn(1, C_LAT, twins.f_chunk, 1, twins.width, generator=g, dtype=torch.float64)
        action_noise = torch.randn(1, twins.chunk, 80, generator=g, dtype=torch.float64)
        a_live, v_live = live.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask, video_noise=video_noise,
            action_noise=action_noise,
        )
        a_ours, v_ours = ours.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
            video_noise=twins.ours(video_noise), action_noise=action_noise,
        )
        torch.testing.assert_close(a_ours, a_live, **CLOSE, msg=lambda m: f"chunk {k} action: {m}")
        torch.testing.assert_close(v_ours, twins.ours(v_live), **CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        live.commit_chunk(twins.clean[k], twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
        ours.commit_chunk(twins.ours(twins.clean[k]), twins.a_clean[k], anchor_state=twins.states[k],
                          anchor_state_mask=state_mask, action_mask=mask)
    assert ours.chunk_idx == live.chunk_idx == chunks


OBSERVED_SESSION_CASES = [
    # views, memory, video_cfg, action_cfg, position mode, window, state, normalizer
    (3, MEMORY_VIDEO_ONLY, 6.0, 1.0, "rank", 1, True, False),       # the released N1-obs checkpoint's contract
    (1, MEMORY_VIDEO_ONLY, 3.0, 1.0, "absolute", 1, True, False),
    (3, MEMORY_VIDEO_ONLY, 2.5, 1.7, "absolute", 2, False, True),
    (1, MEMORY_VIDEO_AND_ACTION, 2.5, 1.7, "rank", 2, True, True),
    (3, MEMORY_VIDEO_AND_ACTION, 3.0, 1.0, "rank", 4, False, False),
]


@pytest.mark.parametrize("views,memory,video_cfg,action_cfg,position_mode,window,state,normalizer", OBSERVED_SESSION_CASES)
def test_an_observed_session_matches_the_live_deploy_session(views, memory, video_cfg, action_cfg, position_mode,
                                                            window, state, normalizer):
    """A first-frame episode of a causal_observation policy (causal_state on or off): the mirror's session == Sana's
    CausalPolicyChunkSession (deploy adc4867ea / 8b599e49d), both generating every chunk from its own observation
    latent (prefilled once per text stream, read by every step) with per-stream CFG past the window boundary, then the
    same commits."""

    steps, chunks = 3, 6
    twins = Twins(views=views, window=window, chunks=chunks, memory=memory, first_frame=True, observation=True,
                  state=state, normalizer=normalizer)
    live = _live_session(twins, steps, video_cfg, action_cfg, position_mode)
    ours = twins.session(steps=steps, video_cfg=video_cfg, action_cfg=action_cfg)
    assert ours.observation and ours.causal_state is state
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    mask[..., 60:] = False
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    live.commit_sink(twins.sink)
    ours.commit_sink(twins.ours(twins.sink))
    g = torch.Generator().manual_seed(23)
    for k in range(chunks):
        video_noise = torch.randn(1, C_LAT, twins.f_chunk, 1, twins.width, generator=g, dtype=torch.float64)
        action_noise = torch.randn(1, twins.chunk, 80, generator=g, dtype=torch.float64)
        observation = twins.sink if k == 0 else twins.observations[k]   # chunk 0's observation is the sink's latent
        a_live, v_live = live.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask, video_noise=video_noise,
            action_noise=action_noise, observation=observation,
        )
        a_ours, v_ours = ours.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
            video_noise=twins.ours(video_noise), action_noise=action_noise, observation=twins.ours(observation),
        )
        torch.testing.assert_close(a_ours, a_live, **OBSERVED_CLOSE, msg=lambda m: f"chunk {k} action: {m}")
        torch.testing.assert_close(v_ours, twins.ours(v_live), **OBSERVED_CLOSE, msg=lambda m: f"chunk {k} video: {m}")
        live.commit_chunk(twins.clean[k], twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
        ours.commit_chunk(twins.ours(twins.clean[k]), twins.a_clean[k], anchor_state=twins.states[k],
                          anchor_state_mask=state_mask, action_mask=mask)
    assert ours.chunk_idx == live.chunk_idx == chunks


def test_an_observed_policy_requires_its_observation_and_others_refuse_one():
    """generate_chunk of a causal_observation policy needs the chunk's observation latent; a policy without the option
    refuses one (Sana's session does both)."""

    observed = Twins(chunks=2, memory=MEMORY_VIDEO_ONLY, first_frame=True, observation=True)
    plain = Twins(chunks=2, memory=MEMORY_VIDEO_ONLY, first_frame=True)
    mask = torch.ones(1, observed.chunk, 80, dtype=torch.bool)
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    for twins, observation, match in (
        (observed, None, "generates every chunk from its own observation latent"),
        (plain, plain.ours(plain.sink), "reads no observation latent"),
    ):
        session = twins.session()
        session.commit_sink(twins.ours(twins.sink))
        with pytest.raises(ValueError, match=match):
            session.generate_chunk(anchor_state=twins.states[0], anchor_state_mask=state_mask, action_mask=mask,
                                   observation=observation)
    session = observed.session()
    session.commit_sink(observed.ours(observed.sink))
    with pytest.raises(ValueError, match="observation must be"):
        session.generate_chunk(anchor_state=observed.states[0], anchor_state_mask=state_mask, action_mask=mask,
                               observation=observed.ours(observed.noisy[0]).repeat(1, 1, 2, 1, 1))


@pytest.mark.parametrize("state", [False, True], ids=["obs", "obs+state"])
def test_an_observed_generation_writes_nothing_to_the_memory(state):
    """The observation latent and the state row are read by their chunk's generation only: a whole generate_chunk (the
    prefill and every denoising step, both text streams) leaves every stream's memory -- the GDN (S, z) and the softmax
    sink and committed entries -- the same tensors with the same values, and the entry counts unchanged."""

    twins = Twins(views=3, window=2, chunks=4, memory=MEMORY_VIDEO_ONLY, first_frame=True, observation=True, state=state)
    session = twins.session(steps=2, video_cfg=3.0)
    session.commit_sink(twins.ours(twins.sink))
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    for k in range(3):
        before = [
            ({i: tuple(t.clone() for t in m.gdn[i]) for i in m.gdn}, {i: m.sink[i] for i in m.sink},
             {i: [tuple(t.clone() for t in e) for e in m.chunks[i]] for i in m.chunks}, m.committed, m.has_sink)
            for m in session.memories
        ]
        objects = [({i: m.gdn[i] for i in m.gdn}, {i: list(m.chunks[i]) for i in m.chunks}) for m in session.memories]
        session.generate_chunk(anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
                               observation=twins.ours(twins.observations[k]))
        for m, (gdn, sink, chunks, committed, has_sink), (gdn_obj, chunk_obj) in zip(session.memories, before, objects):
            assert m.committed == committed and m.has_sink == has_sink and session.chunk_idx == k
            for i in m.gdn:
                assert m.gdn[i] is gdn_obj[i]
                for now, then in zip(m.gdn[i], gdn[i]):
                    torch.testing.assert_close(now, then, rtol=0, atol=0)
            for i in m.sink:
                assert m.sink[i] is sink[i]
            for i in m.chunks:
                assert len(m.chunks[i]) == len(chunks[i]) and all(a is b for a, b in zip(m.chunks[i], chunk_obj[i]))
                for now, then in zip(m.chunks[i], chunks[i]):
                    for x, y in zip(now, then):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
        twins.commit(session, k)


def test_the_commits_after_observed_chunks_are_the_policys_without_the_option():
    """The commit is the policy's without causal_observation: after every chunk's prefill and denoise reads, the
    memory each commit leaves (GDN (S, z), softmax entries) equals that of the same policy that never observed."""

    observed = Twins(views=3, window=2, chunks=4, memory=MEMORY_VIDEO_ONLY, first_frame=True, observation=True,
                     state=True)
    plain = Twins(views=3, window=2, chunks=4, memory=MEMORY_VIDEO_ONLY, first_frame=True)
    plain.mirror.load_state_dict(observed.mirror.state_dict())
    sessions = []
    for twins in (observed, plain):
        session = twins.session()
        session.commit_sink(twins.ours(twins.sink))
        for k in range(3):
            if twins is observed:
                twins.denoise_forward(session, k)
            twins.commit(session, k)
        sessions.append(session)
    a, b = (s.memories[0] for s in sessions)
    for index in a.gdn:
        for x, y in zip(a.gdn[index], b.gdn[index]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    for index in a.chunks:
        for (ka, va), (kb, vb) in zip(a.chunks[index], b.chunks[index]):
            torch.testing.assert_close(ka, kb, rtol=0, atol=0)
            torch.testing.assert_close(va, vb, rtol=0, atol=0)


def test_the_softmax_clock_stays_in_a_training_window_and_the_gdn_clock_is_physical():
    """Every rotary table the mirror builds during a 7-chunk episode with W = 3: the softmax layers' positions stay
    inside a training window (the sink at frame 0, the read's chunks in slots 0..W-1, the forward's own chunk at most in
    slot W), the GDN layers' windows run on the physical clock of the episode (the sink at frame 0)."""

    window, length = 3, 7
    twins = Twins(window=window, chunks=length)
    mirror, f_c, chunk = twins.mirror, twins.f_chunk, twins.chunk
    calls = []
    original = mirror.window_rope

    def recording(rope, fps, frames, view_shapes, view_slot_ids, robot_rows, frame_offset, step_offset, device,
                  robot_step_start=0):
        calls.append((rope is mirror.rope_softmax, frames, robot_rows, frame_offset, step_offset))
        return original(rope, fps, frames, view_shapes, view_slot_ids, robot_rows, frame_offset, step_offset, device,
                        robot_step_start)

    mirror.window_rope = recording
    twins.streaming(length)
    soft = [c for c in calls if c[0]]
    gdn = [c for c in calls if not c[0]]
    assert max(c[3] + c[1] - 1 for c in soft) == (window + 1) * f_c     # the last target frame of slot W
    assert min(c[3] for c in soft) == 0                                  # the sink (and chunk 0's slot)
    assert max(c[4] for c in soft if c[2]) == window * chunk             # the state row of slot W
    assert max(c[3] for c in gdn) == (length - 1) * f_c + 1              # GDN: the last chunk's targets, physical
    assert max(c[4] for c in gdn) == (length - 1) * chunk
    assert (False, 1, 0, 0, 0) in calls                                  # the sink's own GDN window at frame 0


def test_the_session_refuses_a_broken_opening():
    twins = Twins(chunks=3)
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    kwargs = dict(anchor_state=twins.states[0], anchor_state_mask=state_mask, action_mask=mask)
    session = twins.session()
    with pytest.raises(RuntimeError, match="commit_sink first"):
        twins.commit(session, 0)
    with pytest.raises(RuntimeError, match="commit_sink first"):
        session.generate_chunk(**kwargs)
    session.commit_sink(twins.ours(twins.sink))
    with pytest.raises(RuntimeError, match="once, before the first chunk"):
        session.commit_sink(twins.ours(twins.sink))
    with pytest.raises(RuntimeError, match="first emission is chunk 1"):
        session.generate_chunk(**kwargs)
    twins.commit(session, 0)
    session.generate_chunk(**kwargs)
    session.reset_memory()
    with pytest.raises(RuntimeError, match="commit_sink first"):
        session.generate_chunk(**kwargs)


@pytest.mark.parametrize("state", [False, True], ids=["obs", "obs+state"])
@pytest.mark.parametrize("window", [1, 2])
def test_every_forward_of_an_observed_episode_reads_exactly_its_causal_set(state, window):
    """The causal read sets of an observed episode, layer by layer (the softmax kernel runs without a mask, so its keys
    ARE the read set): chunk c's observation prefill reads the sink, the window's committed chunks and its own group
    (the observation latent of every view, plus the state row under causal_state) -- no target or action key exists in
    it; every denoising step reads the sink, the window, the prefill's observation K/V (exactly the tensors the prefill
    left) and its own targets and rows; the commit reads the sink, the window and its own targets, no observation. GDN:
    the prefill and the denoise reads write nothing to the memory, the prefill leaves its branch and every denoise read
    takes that branch (the very tensors), the commit neither reads a branch nor skips its write."""

    from sana_wam_causal.layers import CausalGDNAttention, CausalSoftmaxAttention

    chunks = 4
    twins = Twins(views=3, window=window, chunks=chunks, memory=MEMORY_VIDEO_ONLY, first_frame=True, observation=True,
                  state=state)
    session = twins.session(steps=2, video_cfg=3.0)   # two text streams, two steps
    width, chunk = twins.width, twins.chunk
    n_obs = width + int(state)
    n_denoise = width + chunk + (0 if state else 1)
    soft, gdn, branches, phase = [], [], [], {"name": None}
    run_attention = CausalSoftmaxAttention._run_attention
    gdn_forward = CausalGDNAttention.forward

    def record_soft(self, q, k, v):
        soft.append((phase["name"], q.shape[1], k, v))
        return run_attention(self, q, k, v)

    def record_gdn(self, x, rotary_emb=None, cache=None):
        observed = cache.gdn_observation   # what the forward READS (a prefill leaves its branch here afterwards)
        out = gdn_forward(self, x, rotary_emb=rotary_emb, cache=cache)
        gdn.append((phase["name"], x.shape[1], bool(cache.observation_prefill), None if observed is None else observed,
                    cache.commit, cache.gdn_next is not None))
        if cache.observation_prefill:
            branch = cache.gdn_observation
            assert branch is not None and branch.state.shape[-2:] == (self.dim, self.dim) and branch.state_z.shape[-1] == self.dim
            branches.append(branch)
        return out

    prefill = session.prefill_observation
    left = {}

    def recording_prefill(caption, memory, observation, **kwargs):
        caches = prefill(caption, memory, observation, **kwargs)
        left.setdefault(session.chunk_idx, []).append(caches)
        return caches

    CausalSoftmaxAttention._run_attention, CausalGDNAttention.forward = record_soft, record_gdn
    session.prefill_observation = recording_prefill
    mask = torch.ones(1, chunk, 80, dtype=torch.bool)
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    try:
        phase["name"] = "sink"
        session.commit_sink(twins.ours(twins.sink))
        for c in range(chunks):
            n_window = min(c, window) * width
            soft.clear(), gdn.clear(), branches.clear()
            phase["name"] = "generate"
            session.generate_chunk(anchor_state=twins.states[c], anchor_state_mask=state_mask, action_mask=mask,
                                   observation=twins.ours(twins.observations[c]))
            prefills = [entry for entry in soft if entry[1] == n_obs]
            denoise = [entry for entry in soft if entry[1] == n_denoise]
            assert len(prefills) + len(denoise) == len(soft)
            layers = len(session.memories[0].sink)
            assert len(prefills) == 2 * layers and len(denoise) == 2 * 2 * layers   # 2 streams; 2 steps x 2 streams
            for _, q, k, _v in prefills:
                assert k.shape[1] == width + n_window + n_obs          # sink | window | its own group
            for i, (_, q, k, _v) in enumerate(denoise):
                assert k.shape[1] == width + n_window + n_obs + n_denoise   # sink | window | observation | itself
            # the observation keys every denoising step reads are the prefill's own (per stream, per softmax layer)
            for stream, caches in enumerate(left[c]):
                observation_k = [cache.softmax_context[0][:, -n_obs:] for cache in caches if cache.softmax_context is not None]
                steps = [entry for entry in denoise][stream * layers : (stream + 1) * layers]
                for k_obs, (_, _, k, _v) in zip(observation_k, steps):
                    torch.testing.assert_close(k[:, width + n_window : width + n_window + n_obs], k_obs.to(k.dtype),
                                               rtol=0, atol=0)
            gdn_prefill = [entry for entry in gdn if entry[2]]
            gdn_denoise = [entry for entry in gdn if not entry[2]]
            assert all(t == n_obs and read is None and not commit and not wrote for _, t, _, read, commit, wrote in gdn_prefill)
            assert all(t == n_denoise and not commit and not wrote for _, t, _, read, commit, wrote in gdn_denoise)
            # every denoise read takes the branch its stream's prefill left in that layer (the prefills run stream by
            # stream; every denoising step runs both streams, layer by layer)
            n_gdn = len(gdn_prefill) // 2
            for i, (_, _, _, read, _, _) in enumerate(gdn_denoise):
                stream = (i // n_gdn) % 2
                assert read is branches[stream * n_gdn + i % n_gdn]
            soft.clear(), gdn.clear()
            phase["name"] = "commit"
            twins.commit(session, c)
            assert all(q == width and k.shape[1] == width + n_window + width for _, q, k, _v in soft)
            assert all(read is None and commit and wrote and not pre for _, t, pre, read, commit, wrote in gdn)
    finally:
        CausalSoftmaxAttention._run_attention, CausalGDNAttention.forward = run_attention, gdn_forward


@pytest.mark.parametrize("state", [False, True], ids=["obs", "obs+state"])
def test_an_observation_and_its_state_are_read_by_their_own_chunk_only(state):
    """Who reads whom, on the deploy chain's outputs: moving chunk 1's observation latent (or its anchor state) moves
    chunk 1's generation and no other chunk's -- nothing of it reaches the memory; moving chunk 1's noisy targets moves
    chunk 1 only and leaves the observation's prefill branch as it was (the observation never reads its targets);
    moving the clean chunk 1 committed to the memory moves the chunks after it and none before."""

    chunks = 4
    twins = Twins(views=3, window=2, chunks=chunks, memory=MEMORY_VIDEO_ONLY, first_frame=True, observation=True,
                  state=state)
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    g = torch.Generator().manual_seed(29)
    noise = [(torch.randn(1, C_LAT, twins.f_chunk, 1, twins.width, generator=g, dtype=torch.float64),
              torch.randn(1, twins.chunk, 80, generator=g, dtype=torch.float64)) for _ in range(chunks)]

    def episode():
        session = twins.session(steps=2, video_cfg=3.0)
        session.commit_sink(twins.ours(twins.sink))
        terms, outs = [], []
        prefill = session.prefill_observation

        def recording(caption, memory, observation, **kwargs):
            caches = prefill(caption, memory, observation, **kwargs)
            terms.append([(c.gdn_observation.state.clone() if c.gdn_observation is not None else c.softmax_context[0][:, -1:].clone())
                          for c in caches])
            return caches

        session.prefill_observation = recording
        for c in range(chunks):
            outs.append(session.generate_chunk(
                anchor_state=twins.states[c], anchor_state_mask=state_mask, action_mask=mask,
                video_noise=twins.ours(noise[c][0]), action_noise=noise[c][1], observation=twins.ours(twins.observations[c]),
            ))
            twins.commit(session, c)
        return outs, terms

    base, base_terms = episode()

    def moved_chunks(edit):
        kind, index, value = edit
        original = getattr(twins, kind)[index] if kind != "noise" else noise[index]
        if kind == "noise":
            noise[index] = (noise[index][0] + value, noise[index][1])
        else:
            getattr(twins, kind)[index] = original + value
        try:
            outs, terms = episode()
        finally:
            if kind == "noise":
                noise[index] = original
            else:
                getattr(twins, kind)[index] = original
        moved = {c for c in range(chunks)
                 if any(float((a - b).abs().max()) > 1e-9 for a, b in zip(outs[c], base[c]))}
        same = {c for c in range(chunks)
                if all(torch.equal(a, b) for a, b in zip(outs[c], base[c]))}
        assert moved | same == set(range(chunks)) and not moved & same
        return moved, terms

    moved, _ = moved_chunks(("observations", 1, 1.0))
    assert moved == {1}
    moved, _ = moved_chunks(("states", 1, 0.5))
    assert moved == {1}
    moved, terms = moved_chunks(("noise", 1, 1.0))
    assert moved == {1}
    for stream in (0, 1):   # chunk 1's prefill (streams 2 and 3 of the episode's prefills) never saw its targets
        for a, b in zip(terms[2 + stream], base_terms[2 + stream]):
            assert torch.equal(a, b)
    moved, _ = moved_chunks(("clean", 1, 1.0))
    assert moved == {2, 3}



PHYSICAL_CASES = [
    # views, stride, chunk, window, length, memory, first_frame, observation, state, normalizer
    (3, 4, 32, 1, 5, MEMORY_VIDEO_ONLY, True, True, True, False),      # the released physical-RoPE checkpoint's contract
    (1, 4, 32, 1, 5, MEMORY_VIDEO_ONLY, True, True, True, False),
    (3, 4, 32, 2, 6, MEMORY_VIDEO_ONLY, True, True, False, True),
    (3, 4, 32, 2, 6, MEMORY_VIDEO_ONLY, True, False, False, True),     # physical_rope without observation latents
    (1, 4, 32, 2, 6, MEMORY_VIDEO_AND_ACTION, False, True, True, True),
    (1, 4, 32, None, 4, MEMORY_VIDEO_AND_ACTION, False, False, False, True),
    (3, 4, 32, 3, 6, MEMORY_VIDEO_AND_ACTION, True, True, True, True),
    (1, 1, 16, 2, 5, MEMORY_VIDEO_ONLY, True, True, True, False),      # f_c = 2: targets at c * f_c + i + 0.5625
    (1, 1, 16, 3, 6, MEMORY_VIDEO_AND_ACTION, False, False, False, True),
]
PHYSICAL_IDS = ["V3-W1-obs+state-ff-num", "V1-W1-obs+state-ff-num", "V3-W2-obs-ff", "V3-W2-noobs-ff", "V1-W2-va-obs+state-pad",
                "V1-noW-va-noobs-pad", "V3-W3-va-obs+state-ff", "V1-s1c16-W2-obs+state-ff-num", "V1-s1c16-W3-va-noobs-pad"]


@pytest.mark.parametrize("views,stride,chunk,window,length,memory,first_frame,observation,state,normalizer", PHYSICAL_CASES,
                         ids=PHYSICAL_IDS)
def test_a_physical_rope_deploy_chain_equals_two_stream_training_forward(views, stride, chunk, window, length, memory,
                                                                        first_frame, observation, state, normalizer):
    """physical_rope (Sana 6e91eaf2a): every noisy chunk of the training canvas on the physical clock (target latents at
    their frames' middle, the observation at its chunk's start, the rows at their steps, an evicting reader's sink at its
    window's start) == the deploy chain's chunk, its reads numbered from their origin; with and without observation
    latents and their state rows, both memory contents and phase windows, windows 1 / 2 / 3 / none, f_c 1 and 2."""

    twins = Twins(views=views, stride=stride, chunk=chunk, window=window, chunks=length, memory=memory,
                  first_frame=first_frame, normalizer=normalizer, observation=observation, state=state, physical=True)
    assert twins.live.physical_rope and twins.mirror.physical_rope
    training = twins.two_stream(length) if first_frame else twins.two_stream(length)[1:]
    streaming = twins.streaming_first_frame(length) if first_frame else twins.streaming(length)
    close = OBSERVED_CLOSE if observation else CLOSE
    assert len(training) == len(streaming)
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming)):
        torch.testing.assert_close(x_ours, x_live, **close, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **close, msg=lambda m: f"chunk {k} action: {m}")


@pytest.mark.parametrize("observation", [False, True], ids=["noobs", "obs+state"])
@pytest.mark.parametrize("mirror_physical", [False, True], ids=["mirror-slots", "mirror-physical"])
def test_the_wrong_clock_is_caught(observation, mirror_physical):
    """Negative controls: the mirror on the latent-slot clock against a physical-RoPE policy, and the mirror on the
    physical clock against a latent-slot policy, both miss the training forward -- the parity above can tell the clocks
    apart."""

    twins = Twins(views=3, window=2, chunks=4, memory=MEMORY_VIDEO_ONLY, first_frame=True, observation=observation,
                  state=observation, physical=mirror_physical, live_physical=not mirror_physical)
    training = twins.two_stream(4)
    streaming = twins.streaming_first_frame(4)
    gap = max(_max_abs(streaming[k][i], training[k][i]) for k in range(4) for i in (0, 1))
    assert gap > 1e-6, gap


def test_the_physical_clock_places_every_token_at_its_instant():
    """The rotary positions every forward of a physical-RoPE observed episode uses, on the GDN (physical) clock and the
    softmax (read) clock: the sink at 0; chunk c's observation prefill at frame c * f_c with its state row at step c * C;
    its denoising window's targets at c * f_c + 0.5625 and its action rows at c * C + 1 ..; its commit at c * f_c +
    0.5625; the softmax reads of an N = 1 window at sink / memory / observation / targets = 0 / 0.5625 / 1 / 1.5625."""

    twins = Twins(views=3, window=1, chunks=4, memory=MEMORY_VIDEO_ONLY, first_frame=True, observation=True, state=True,
                  physical=True)
    mirror, chunk = twins.mirror, twins.chunk
    calls = []
    original = mirror.window_rope

    def recording(rope, fps, frames, view_shapes, view_slot_ids, robot_rows, frame_offset, step_offset, device,
                  robot_step_start=0):
        calls.append(("soft" if rope is mirror.rope_softmax else "gdn", frames, robot_rows, float(frame_offset),
                      int(step_offset) + int(robot_step_start)))
        return original(rope, fps, frames, view_shapes, view_slot_ids, robot_rows, frame_offset, step_offset, device,
                        robot_step_start)

    mirror.window_rope = recording
    session = twins.session()
    session.commit_sink(twins.ours(twins.sink))
    mask = torch.ones(1, chunk, 80, dtype=torch.bool)
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    for c in range(4):
        calls.clear()
        session.generate_chunk(anchor_state=twins.states[c], anchor_state_mask=state_mask, action_mask=mask,
                               observation=twins.ours(twins.observations[c]))
        gdn = [call for call in calls if call[0] == "gdn"]
        # the window tables (robot rows: the prefill's state row, the denoise window's actions); the entry tables (the
        # sink, the video_only committed chunks) carry no rows and are built once, at their first read
        soft = [call for call in calls if call[0] == "soft" and call[2] > 0]
        assert gdn[0] == ("gdn", 1, 1, float(c), c * chunk)                 # the prefill: its observation and state row
        assert gdn[1] == ("gdn", 1, chunk, c + 0.5625, c * chunk + 1)        # a denoising step: targets and actions
        origin = max(0, c - 1)
        assert soft[0] == ("soft", 1, 1, float(c - origin), (c - origin) * chunk)
        assert soft[1] == ("soft", 1, chunk, c - origin + 0.5625, (c - origin) * chunk + 1)
        if c >= 1:   # N = 1: sink 0, memory 0.5625, observation 1, targets 1.5625 on the read's clock
            assert (soft[0][3], soft[1][3]) == (1.0, 1.5625)
        calls.clear()
        twins.commit(session, c)
        assert [call[3] for call in calls if call[0] == "gdn"] == [c + 0.5625]
    tables = {key[1]: value for key, value in session._rope_tables.items() if key[0] == "chunk"}
    reference = mirror.softmax_entry_rope(frames=1, frame_offset=0.5625, robot_rows=0, step_offset=0, model_fps=FPS,
                                          view_shapes=twins.view_shapes, view_slot_ids=twins.slots, device="cpu")
    torch.testing.assert_close(tables[0], reference, rtol=0, atol=0)   # the memory chunk of every N = 1 read at 0.5625


PLAIN_CASES = [
    # views, stride, chunk, window, length, memory, first_frame, observation, state, physical, normalizer
    (3, 4, 32, 1, 5, MEMORY_VIDEO_ONLY, True, False, False, True, False),     # the RoboDojo no-obs N1 recipe (020582851)
    (1, 4, 32, 1, 5, MEMORY_VIDEO_ONLY, True, True, True, True, False),       # obs+state on the canvas, physical, numerator
    (3, 4, 32, 2, 6, MEMORY_VIDEO_ONLY, True, True, True, True, False),
    (3, 4, 32, 2, 6, MEMORY_VIDEO_ONLY, True, True, False, False, True),
    (1, 4, 32, 2, 6, MEMORY_VIDEO_AND_ACTION, False, False, False, False, True),
    (3, 4, 32, None, 4, MEMORY_VIDEO_AND_ACTION, False, False, False, True, True),
    (1, 1, 16, 3, 6, MEMORY_VIDEO_ONLY, True, True, True, True, False),       # f_c = 2
]
PLAIN_IDS = ["V3-W1-noobs-phys-num", "V1-W1-obs+state-phys-num", "V3-W2-obs+state-phys-num", "V3-W2-obs-slots",
             "V1-W2-va-pad-slots", "V3-noW-va-pad-phys", "V1-s1c16-W3-obs+state-phys-num"]


@pytest.mark.parametrize("views,stride,chunk,window,length,memory,first_frame,observation,state,physical,normalizer",
                         PLAIN_CASES, ids=PLAIN_IDS)
def test_a_plain_residual_deploy_chain_equals_two_stream_training_forward(views, stride, chunk, window, length, memory,
                                                                         first_frame, observation, state, physical,
                                                                         normalizer):
    """model.use_attn_res false (Sana 53234167c): the block stack is a plain residual on both sides (each layer adds its
    attention sublayer's delta, then its MLP delta; no AttnRes module, no final aggregation), and every noisy chunk of
    the training forward == the deploy chain's chunk, with and without observation latents and their state rows, on both
    clocks, both memory contents and phase windows, with and without the GDN denominator."""

    twins = Twins(views=views, stride=stride, chunk=chunk, window=window, chunks=length, memory=memory,
                  first_frame=first_frame, normalizer=normalizer, observation=observation, state=state, physical=physical,
                  attn_res=False)
    assert twins.live.attn_res is None and twins.mirror.attn_res is None
    assert not any(key.startswith("attn_res.") for key in twins.mirror.state_dict())
    training = twins.two_stream(length) if first_frame else twins.two_stream(length)[1:]
    streaming = twins.streaming_first_frame(length) if first_frame else twins.streaming(length)
    close = OBSERVED_CLOSE if observation else CLOSE
    assert len(training) == len(streaming)
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming)):
        torch.testing.assert_close(x_ours, x_live, **close, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **close, msg=lambda m: f"chunk {k} action: {m}")


PLAIN_SESSION_CASES = [
    # views, memory, video_cfg, action_cfg, position mode, window, observation, state, normalizer
    (3, MEMORY_VIDEO_ONLY, 6.0, 1.0, "rank", 1, False, False, False),
    (1, MEMORY_VIDEO_ONLY, 6.0, 1.0, "absolute", 1, True, True, False),
    (3, MEMORY_VIDEO_AND_ACTION, 2.5, 1.7, "rank", 2, True, False, True),
]


@pytest.mark.parametrize("views,memory,video_cfg,action_cfg,position_mode,window,observation,state,normalizer",
                         PLAIN_SESSION_CASES)
def test_a_plain_residual_session_matches_the_live_deploy_session(views, memory, video_cfg, action_cfg, position_mode,
                                                                 window, observation, state, normalizer):
    """A first-frame episode of a policy without AttnRes: the mirror's session == Sana's CausalPolicyChunkSession (its
    unfused path runs the plain stack), per-stream CFG past the window boundary, with and without observation latents."""

    steps, chunks = 3, 5
    twins = Twins(views=views, window=window, chunks=chunks, memory=memory, first_frame=True, observation=observation,
                  state=state, normalizer=normalizer, attn_res=False)
    live = _live_session(twins, steps, video_cfg, action_cfg, position_mode)
    ours = twins.session(steps=steps, video_cfg=video_cfg, action_cfg=action_cfg)
    close = OBSERVED_CLOSE if observation else CLOSE
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    mask[..., 60:] = False
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    live.commit_sink(twins.sink)
    ours.commit_sink(twins.ours(twins.sink))
    g = torch.Generator().manual_seed(29)
    for k in range(chunks):
        video_noise = torch.randn(1, C_LAT, twins.f_chunk, 1, twins.width, generator=g, dtype=torch.float64)
        action_noise = torch.randn(1, twins.chunk, 80, generator=g, dtype=torch.float64)
        extra_live, extra_ours = {}, {}
        if observation:
            seen = twins.sink if k == 0 else twins.observations[k]   # chunk 0's observation is the sink's latent
            extra_live, extra_ours = {"observation": seen}, {"observation": twins.ours(seen)}
        a_live, v_live = live.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask, video_noise=video_noise,
            action_noise=action_noise, **extra_live,
        )
        a_ours, v_ours = ours.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
            video_noise=twins.ours(video_noise), action_noise=action_noise, **extra_ours,
        )
        torch.testing.assert_close(a_ours, a_live, **close, msg=lambda m: f"chunk {k} action: {m}")
        torch.testing.assert_close(v_ours, twins.ours(v_live), **close, msg=lambda m: f"chunk {k} video: {m}")
        live.commit_chunk(twins.clean[k], twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
        ours.commit_chunk(twins.ours(twins.clean[k]), twins.a_clean[k], anchor_state=twins.states[k],
                          anchor_state_mask=state_mask, action_mask=mask)
    assert ours.chunk_idx == live.chunk_idx == chunks


def test_a_plain_residual_mirror_is_not_the_attn_res_stack():
    """Negative control: on a policy without AttnRes the mirror's plain stack is what the parity checks -- dropping its
    MLP residual (one wrong sum) misses the training forward."""

    twins = Twins(views=3, window=1, chunks=4, memory=MEMORY_VIDEO_ONLY, first_frame=True, physical=True, attn_res=False)
    training = twins.two_stream(4)
    for block in twins.mirror.blocks:
        block.forward_mlp_sublayer = lambda hidden, modulation: torch.zeros_like(hidden)
    streaming = twins.streaming_first_frame(4)
    gap = max(_max_abs(streaming[k][i], training[k][i]) for k in range(4) for i in (0, 1))
    assert gap > 1e-6, gap


HORIZON_CASES = [
    # views, stride, chunk, window, length, memory, first_frame, observation, state, normalizer, attn_res
    (3, 4, 32, 1, 10, MEMORY_VIDEO_ONLY, True, True, True, False, False),       # the 7ed0d28bf sana_latent no-AttnRes N1
    (1, 4, 32, 1, 10, MEMORY_VIDEO_ONLY, True, True, True, False, True),        # the sana_pixel obs-block N1 recipe
    (3, 4, 32, 2, 11, MEMORY_VIDEO_ONLY, True, False, False, False, True),      # no observation latents
    (1, 4, 32, 3, 11, MEMORY_VIDEO_AND_ACTION, False, True, True, True, True),  # padded window, video_and_action
    (3, 4, 32, 7, 10, MEMORY_VIDEO_ONLY, True, True, False, True, True),        # W = H: the horizon is the window's start
    (1, 1, 16, 2, 10, MEMORY_VIDEO_ONLY, True, True, True, False, False),       # f_c = 2
]
HORIZON_IDS = ["V3-W1-obs+state-num-plain", "V1-W1-obs+state-num", "V3-W2-noobs-num", "V1-W3-va-obs+state-pad",
               "V3-W7-obs", "V1-s1c16-W2-obs+state-num-plain"]


@pytest.mark.parametrize("views,stride,chunk,window,length,memory,first_frame,observation,state,normalizer,attn_res",
                         HORIZON_CASES, ids=HORIZON_IDS)
def test_a_sink_horizon_deploy_chain_equals_two_stream_training_forward(views, stride, chunk, window, length, memory,
                                                                       first_frame, observation, state, normalizer,
                                                                       attn_res):
    """Sana 0c88c11ab: every noisy chunk of the training canvas on the physical clock with the softmax sink read at
    chunk max(0, c - 7) (``_sink_shift_phases`` over ``sink_read_chunks``) == the deploy chain's chunk, its reads
    numbered from that chunk; past the horizon (chunks 8 ..) included, windows 1 / 2 / 3 / 7, with and without
    observation latents and their state rows, both memory contents and phase windows, f_c 1 and 2, AttnRes or not."""

    twins = Twins(views=views, stride=stride, chunk=chunk, window=window, chunks=length, memory=memory,
                  first_frame=first_frame, normalizer=normalizer, observation=observation, state=state,
                  attn_res=attn_res, horizon=True)
    assert twins.live.sink_horizon == twins.mirror.sink_horizon == 7 and twins.mirror.physical_rope
    training = twins.two_stream(length) if first_frame else twins.two_stream(length)[1:]
    streaming = twins.streaming_first_frame(length) if first_frame else twins.streaming(length)
    close = OBSERVED_CLOSE if observation else CLOSE
    assert len(training) == len(streaming) and length - 1 > 7
    for k, ((x_live, a_live), (x_ours, a_ours)) in enumerate(zip(training, streaming)):
        torch.testing.assert_close(x_ours, x_live, **close, msg=lambda m: f"chunk {k} video: {m}")
        torch.testing.assert_close(a_ours, a_live, **close, msg=lambda m: f"chunk {k} action: {m}")


@pytest.mark.parametrize("window", [1, 7])
def test_the_window_start_sink_is_caught_on_a_sink_horizon_policy(window):
    """Negative control: the mirror reading the sink at its window's start (the trees before 0c88c11ab, on the physical
    clock) against a sink-horizon policy. With W = 1 the two rules agree on chunks 0 and 1 only and the parity misses
    every later chunk; with W = 7 the horizon is the window's start and they agree everywhere."""

    length = 10
    twins = Twins(views=3, window=window, chunks=length, memory=MEMORY_VIDEO_ONLY, first_frame=True, observation=True,
                  state=True, physical=True, horizon=False, live_horizon=True)
    assert twins.mirror.sink_horizon is None and twins.live.sink_horizon == 7
    training = twins.two_stream(length)
    streaming = twins.streaming_first_frame(length)
    passes = [all(torch.allclose(streaming[k][i], training[k][i], **OBSERVED_CLOSE) for i in (0, 1)) for k in range(length)]
    gaps = [max(_max_abs(streaming[k][i], training[k][i]) for i in (0, 1)) for k in range(length)]
    assert passes == ([True] * length if window == 7 else [True, True] + [False] * (length - 2)), gaps


HORIZON_SESSION_CASES = [
    # views, memory, video_cfg, action_cfg, position mode, observation, state, normalizer, attn_res
    (3, MEMORY_VIDEO_ONLY, 6.0, 1.0, "rank", True, True, False, False),
    (1, MEMORY_VIDEO_ONLY, 6.0, 1.0, "absolute", True, True, False, True),
    (3, MEMORY_VIDEO_AND_ACTION, 2.5, 1.7, "rank", False, False, True, True),
]


@pytest.mark.parametrize("views,memory,video_cfg,action_cfg,position_mode,observation,state,normalizer,attn_res",
                         HORIZON_SESSION_CASES)
def test_a_sink_horizon_session_matches_the_live_deploy_session_at_a_seven_chunk_window(
        views, memory, video_cfg, action_cfg, position_mode, observation, state, normalizer, attn_res):
    """Sana's cached forms run a sink-horizon policy only at W = 7, where its store's window-start sink is the horizon:
    a first-frame episode of the mirror's session == Sana's CausalPolicyChunkSession past chunk 7, per-stream CFG."""

    steps, chunks = 2, 10
    twins = Twins(views=views, window=7, chunks=chunks, memory=memory, first_frame=True, observation=observation,
                  state=state, normalizer=normalizer, attn_res=attn_res, horizon=True)
    live = _live_session(twins, steps, video_cfg, action_cfg, position_mode)
    ours = twins.session(steps=steps, video_cfg=video_cfg, action_cfg=action_cfg)
    close = OBSERVED_CLOSE if observation else CLOSE
    mask = torch.ones(1, twins.chunk, 80, dtype=torch.bool)
    mask[..., 60:] = False
    state_mask = torch.ones(1, 80, dtype=torch.bool)
    live.commit_sink(twins.sink)
    ours.commit_sink(twins.ours(twins.sink))
    g = torch.Generator().manual_seed(31)
    origins = []
    for k in range(chunks):
        origins.append(ours.softmax_origin())
        video_noise = torch.randn(1, C_LAT, twins.f_chunk, 1, twins.width, generator=g, dtype=torch.float64)
        action_noise = torch.randn(1, twins.chunk, 80, generator=g, dtype=torch.float64)
        extra_live, extra_ours = {}, {}
        if observation:
            seen = twins.sink if k == 0 else twins.observations[k]   # chunk 0's observation is the sink's latent
            extra_live, extra_ours = {"observation": seen}, {"observation": twins.ours(seen)}
        a_live, v_live = live.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask, video_noise=video_noise,
            action_noise=action_noise, **extra_live,
        )
        a_ours, v_ours = ours.generate_chunk(
            anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask,
            video_noise=twins.ours(video_noise), action_noise=action_noise, **extra_ours,
        )
        torch.testing.assert_close(a_ours, a_live, **close, msg=lambda m: f"chunk {k} action: {m}")
        torch.testing.assert_close(v_ours, twins.ours(v_live), **close, msg=lambda m: f"chunk {k} video: {m}")
        live.commit_chunk(twins.clean[k], twins.a_clean[k], anchor_state=twins.states[k], anchor_state_mask=state_mask, action_mask=mask)
        ours.commit_chunk(twins.ours(twins.clean[k]), twins.a_clean[k], anchor_state=twins.states[k],
                          anchor_state_mask=state_mask, action_mask=mask)
    assert ours.chunk_idx == live.chunk_idx == chunks
    assert origins == [max(0, k - 7) for k in range(chunks)]
