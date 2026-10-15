"""Live parity of the sparse-memory mirror against Sana's causal policy at rwm/sparse-memory (``SANA_SM_REPO``).

The live class and the mirror get identical random weights (tiny dims, fp64, CPU, xformers off, ``flash_attn`` bound to an
fp64 reference varlen as Sana's own tests do; softmax attention in fp64 on both sides). Checked, for both memory types
(strided s = 25, fixed m = 16) and n = 0, 1, 3 memory latents:

* the training forward (``model(x, t, y, data_info)`` on [S | M_1..M_n | T] + rows) == ``forward_sm(form='sequence')``,
  every token;
* Sana's deploy entries (``forward_memory_prefix`` then ``forward_memory_target``) == ``forward_sm(form='prefix')`` then
  ``forward_sm(form='target')``, and the caches the prefix leaves (every GDN layer's (S, z), every softmax layer's
  rotated K and V);
* prefix + target == the sequence's last chunk (the property Sana's ``test_sparse_memory_deploy.py`` asserts), and the
  sink and memory latents never read the last chunk;
* the session: a whole decision (Euler loop, both CFG settings, both deploy forms) == Sana's
  ``SparseMemoryPolicySession.decide`` / ``decide_uncached`` on the same noise.

Run on its own (the first file of a session owns the ``dev`` package).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import _paths  # noqa: F401
import _live

from sana_wam_sm import sparse_memory as sm
from sana_wam_sm.contract import DEPLOY_CACHED, DEPLOY_UNCACHED, SMContract
from sana_wam_sm.layers import GDN_KIND, GDNCache, SoftmaxCache
from sana_wam_sm.model import SMPolicyModel
from sana_wam_sm.session import Caption, SMSession
from sana_wam_min.policy_model.config import PolicyConfig

FPS = 25.0
C_LAT = 4
GRID = (10, 16)
TEXT_LEN, TEXT_CH = 7, 16
MEMORIES = {"strided": {"memory_type": "strided", "memory_stride": 25}, "fixed": {"memory_type": "fixed", "memory_frames": 16}}
CLOSE = {"rtol": 1e-9, "atol": 1e-12}
# prefix + target vs the whole sequence: the matmuls run on fewer rows (Sana's own bound is 1e-12 / 1e-14)
ROUND_OFF = {"rtol": 1e-10, "atol": 1e-12}
EXACT = {"rtol": 0.0, "atol": 0.0}


@pytest.fixture(autouse=True)
def _fake_flash_attn(monkeypatch):
    _live.live_causal()
    _live.bind_reference_varlen(monkeypatch)


class _Cfg(SimpleNamespace):
    def __getattr__(self, name):
        return None


class Twins:
    """A tiny live sparse-memory policy and the mirror with the same weights (fp64)."""

    def __init__(self, memory, *, separate=True, depth=4, seed=7):
        net = _live.live_causal()
        Live = net.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal
        self.memory = dict(memory)
        torch.manual_seed(seed)
        kwargs = dict(
            depth=depth, hidden_size=64, patch_size=(1, 1, 1), num_heads=2, in_channels=C_LAT, pred_sigma=False,
            caption_channels=TEXT_CH, model_max_length=TEXT_LEN, linear_head_dim=32, softmax_head_dim=32,
            softmax_ratio=0.5, attn_res_block_size=2, use_time_conditioning=False, y_norm=True, cross_norm=True,
        )
        policy = {"actions_per_chunk": {str(int(FPS)): sm.CHUNK_ROWS}, **memory}
        config = _Cfg(
            model=_Cfg(extra={"chunk_causal_policy": policy, "rope": "aligned"}),
            scheduler=_Cfg(action_flow_shift=1.0 if separate else None),
            data=_Cfg(extra={"multiview": "sana_pixel"}),
            vae=_Cfg(vae_stride=[8, 32, 32]),
        )
        self.live = Live(config=config, **kwargs)
        assert self.live.memory == self.memory
        with torch.no_grad():
            for parameter in self.live.parameters():
                parameter.uniform_(-0.05, 0.05)
        self.live = self.live.double().eval()
        contract = SMContract(
            actions_per_chunk=sm.CHUNK_ROWS, video_frame_stride=4, memory_type=memory["memory_type"],
            memory_frames=memory.get("memory_frames"), memory_stride=memory.get("memory_stride"), model_fps=FPS,
            tier_num_frames=33, trained_memory_latents=8, multiview="sana_pixel", visual_layout="sana_pixel_canvas",
            rope="aligned", dt_bias_init=-5.0, a_log_init=0.0, reset_legacy_beta=False,
        )
        self.mirror = SMPolicyModel(PolicyConfig.from_sana_kwargs(config=config, **kwargs), contract)
        state = {k: v for k, v in self.live.state_dict().items()
                 if k not in ("pos_embed", "y_embedder.y_embedding", "plucker_embed.weight")}
        result = self.mirror.load_state_dict(state, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        self.mirror = self.mirror.double().eval()
        for model in (self.live, self.mirror):
            for block in model.blocks:
                block.attn.fp32_attention = False

        g = torch.Generator().manual_seed(seed + 100)

        def randn(*shape):
            return torch.randn(*shape, generator=g, dtype=torch.float64)

        self.randn = randn
        self.y = randn(1, 1, 1, TEXT_LEN, TEXT_CH)
        self.y_null = randn(1, 1, 1, TEXT_LEN, TEXT_CH)
        self.mask = torch.ones(1, 1, TEXT_LEN, dtype=torch.int16)
        self.mask[..., 5:] = 0
        self.latents = randn(1, C_LAT, 5, *GRID)          # [S | M_1..M_3 | T] at most
        self.action = randn(1, sm.CHUNK_ROWS, 80)
        self.state = randn(1, 80)
        self.t_video = 700.0
        self.t_action = 900.0 if separate else 700.0

    def x(self, n):
        """[S | M_1..M_n | T]: the first n + 1 latents, then the last one as the target."""

        return torch.cat((self.latents[:, :, : n + 1], self.latents[:, :, -1:]), dim=2)

    def info(self, robot=True):
        info = {
            "model_fps": FPS,
            "rwm_task": "policy",
            "view_count": torch.tensor([1]),
            "view_latent_shape": torch.tensor([[list(GRID)]]),
            "video_frame_stride": torch.tensor([4]),
        }
        if robot:
            info.update(
                action80=self.action,
                action_mask80=torch.ones(1, sm.CHUNK_ROWS, 80, dtype=torch.bool),
                action_timestep=torch.full((1, sm.CHUNK_ROWS), self.t_action, dtype=torch.float64),
                initial_state80=self.state,
                initial_state_condition_mask80=torch.ones(1, 80, dtype=torch.bool),
            )
        return info

    def robot(self):
        return {
            "state80": self.state,
            "state_mask80": torch.ones(1, 80, dtype=torch.bool),
            "action80": self.action,
            "action_mask80": torch.ones(1, sm.CHUNK_ROWS, 80, dtype=torch.bool),
            "action_timesteps": torch.full((1, sm.CHUNK_ROWS), self.t_action, dtype=torch.float64),
        }

    def timesteps(self, frames):
        return sm.video_timesteps(torch.full((1,), self.t_video, dtype=torch.float64), frames)   # [1, 1, F]

    # -- the two sides ---------------------------------------------------------------------------------------------------

    def live_sequence(self, n, x=None):
        x = self.x(n) if x is None else x
        with torch.no_grad():
            return self.live(x, self.timesteps(x.shape[2]), self.y, mask=self.mask, data_info=self.info())

    def mirror_sequence(self, n, x=None):
        x = self.x(n) if x is None else x
        return self.mirror.forward_sm(
            x, self.timesteps(x.shape[2])[:, 0], self.y, self.mask, form=sm.FORM_SEQUENCE, model_fps=FPS,
            robot=self.robot(),
        )

    def live_deploy(self, n):
        caches = [{} for _ in self.live.blocks]
        x = self.x(n)
        self.live.forward_memory_prefix(x[:, :, :-1], self.y, mask=self.mask, kv_caches=caches, data_info=self.info(False))
        target = self.live.forward_memory_target(
            x[:, :, -1:], self.timesteps(x.shape[2])[..., -1:], self.y, mask=self.mask, kv_caches=caches,
            memory_latents=n, data_info=self.info(),
        )
        return target, caches

    def mirror_prefix(self, n):
        caches = [GDNCache(commit=True) if b.attn_type == GDN_KIND else SoftmaxCache(commit=True) for b in self.mirror.blocks]
        out = self.mirror.forward_sm(
            self.x(n)[:, :, :-1], torch.zeros(1, n + 1, dtype=torch.float64), self.y, self.mask, form=sm.FORM_PREFIX,
            model_fps=FPS, caches=caches,
        )
        assert out == {}
        return caches

    def mirror_target(self, n, prefix_caches):
        caches = [GDNCache(seed=c.out) if isinstance(c, GDNCache) else SoftmaxCache(prefix=c.out) for c in prefix_caches]
        return self.mirror.forward_sm(
            self.x(n)[:, :, -1:], torch.full((1, 1), self.t_video, dtype=torch.float64), self.y, self.mask,
            form=sm.FORM_TARGET, model_fps=FPS, robot=self.robot(), caches=caches, memory_latents=n,
        )


# -- the forwards ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("memory", MEMORIES.values(), ids=list(MEMORIES))
@pytest.mark.parametrize("n", [0, 1, 3])
def test_the_sequence_is_the_live_training_forward(memory, n):
    twins = Twins(memory)
    live, ours = twins.live_sequence(n), twins.mirror_sequence(n)
    torch.testing.assert_close(ours["x"], live["x"].reshape(ours["x"].shape), **CLOSE)
    torch.testing.assert_close(ours["action_pred"], live["action_pred"], **CLOSE)


def test_a_shared_schedule_runs_too():
    twins = Twins(MEMORIES["strided"], separate=False)
    live, ours = twins.live_sequence(2), twins.mirror_sequence(2)
    torch.testing.assert_close(ours["x"], live["x"].reshape(ours["x"].shape), **CLOSE)
    torch.testing.assert_close(ours["action_pred"], live["action_pred"], **CLOSE)


@pytest.mark.parametrize("memory", MEMORIES.values(), ids=list(MEMORIES))
@pytest.mark.parametrize("n", [0, 1, 3])
def test_the_prefix_and_the_target_are_the_live_deploy_entries(memory, n):
    twins = Twins(memory)
    live, live_caches = twins.live_deploy(n)
    prefix = twins.mirror_prefix(n)
    ours = twins.mirror_target(n, prefix)
    torch.testing.assert_close(ours["x"], live["x"].reshape(ours["x"].shape), **CLOSE)
    torch.testing.assert_close(ours["action_pred"], live["action_pred"], **CLOSE)
    for index, (cache, held) in enumerate(zip(prefix, live_caches)):
        if isinstance(cache, GDNCache):
            assert set(held) == {"S", "z"}
            torch.testing.assert_close(cache.out[0], held["S"], **CLOSE)
            torch.testing.assert_close(cache.out[1], held["z"], **CLOSE)
        else:
            assert set(held) == {"k", "v"}
            assert cache.out[0].shape == held["k"].shape == (1, sm.TOKENS_PER_FRAME * (n + 1), 2, 32)
            torch.testing.assert_close(cache.out[0], held["k"], **CLOSE, msg=lambda m: f"layer {index} k: {m}")
            torch.testing.assert_close(cache.out[1], held["v"], **CLOSE, msg=lambda m: f"layer {index} v: {m}")


@pytest.mark.parametrize("memory", MEMORIES.values(), ids=list(MEMORIES))
@pytest.mark.parametrize("n", [0, 1, 3])
def test_the_prefix_and_the_target_reproduce_the_sequences_last_chunk(memory, n):
    twins = Twins(memory)
    sequence = twins.mirror_sequence(n)
    prefix = twins.mirror_prefix(n)
    reads = [twins.mirror_target(n, prefix) for _ in range(2)]
    torch.testing.assert_close(reads[1]["x"], reads[0]["x"], **EXACT)          # a target forward writes nothing
    torch.testing.assert_close(reads[0]["x"], sequence["x"][:, :, -1:], **ROUND_OFF)
    torch.testing.assert_close(reads[0]["action_pred"], sequence["action_pred"], **ROUND_OFF)


def test_the_sink_and_memory_latents_never_read_the_last_chunk():
    twins = Twins(MEMORIES["strided"])
    reference = twins.mirror_sequence(3)
    edited = twins.x(3).clone()
    edited[:, :, -1] += 1.0
    twins.action = twins.action + 1.0
    twins.state = twins.state + 1.0
    twins.t_action, twins.t_video = 123.0, 321.0
    out = twins.mirror_sequence(3, edited)
    torch.testing.assert_close(out["x"][:, :, :-1], reference["x"][:, :, :-1], **EXACT)
    assert not torch.equal(out["x"][:, :, -1:], reference["x"][:, :, -1:])


def test_the_forms_are_checked():
    twins = Twins(MEMORIES["fixed"])
    mirror = twins.mirror
    x = twins.x(2)
    with pytest.raises(ValueError, match="only the target"):
        noisy = torch.full((1, 4), 5.0, dtype=torch.float64)
        mirror.forward_sm(x, noisy, twins.y, twins.mask, form=sm.FORM_SEQUENCE, model_fps=FPS, robot=twins.robot())
    with pytest.raises(ValueError, match="takes no layer caches"):
        mirror.forward_sm(x, twins.timesteps(4)[:, 0], twins.y, twins.mask, form=sm.FORM_SEQUENCE, model_fps=FPS,
                          robot=twins.robot(), caches=[None] * 4)
    with pytest.raises(ValueError, match="clean latents"):
        mirror.forward_sm(x[:, :, :3], torch.ones(1, 3, dtype=torch.float64), twins.y, twins.mask, form=sm.FORM_PREFIX,
                          model_fps=FPS, caches=twins.mirror_prefix(2))
    prefix = twins.mirror_prefix(2)
    with pytest.raises(ValueError, match="reads 320"):
        caches = [GDNCache(seed=c.out) if isinstance(c, GDNCache) else SoftmaxCache(prefix=c.out) for c in prefix]
        mirror.forward_sm(x[:, :, -1:], torch.ones(1, 1, dtype=torch.float64), twins.y, twins.mask, form=sm.FORM_TARGET,
                          model_fps=FPS, robot=twins.robot(), caches=caches, memory_latents=1)
    with pytest.raises(ValueError, match="reads the prefix's cache"):
        mirror.forward_sm(x[:, :, -1:], torch.ones(1, 1, dtype=torch.float64), twins.y, twins.mask, form=sm.FORM_TARGET,
                          model_fps=FPS, robot=twins.robot(), caches=prefix, memory_latents=2)
    with pytest.raises(ValueError, match="memory_latents places"):
        mirror.forward_sm(x, twins.timesteps(4)[:, 0], twins.y, twins.mask, form=sm.FORM_SEQUENCE, model_fps=FPS,
                          robot=twins.robot(), memory_latents=2)
    with pytest.raises(ValueError, match="160 tokens"):
        mirror.forward_sm(x[..., :8], twins.timesteps(4)[:, 0], twins.y, twins.mask, form=sm.FORM_SEQUENCE,
                          model_fps=FPS, robot=twins.robot())


# -- the session --------------------------------------------------------------------------------------------------------


def _sessions(twins, cfg_scale, action_cfg_scale, steps=3):
    live = _live.live_session().SparseMemoryPolicySession(
        twins.live, caption_embeds=twins.y, caption_mask=twins.mask, data_info=twins.info(False), steps=steps,
        flow_shift=5.0, unconditional_caption_embeds=twins.y_null, unconditional_caption_mask=twins.mask,
        cfg_scale=cfg_scale, action_flow_shift=1.0, action_cfg_scale=action_cfg_scale,
    )
    ours = {
        form: SMSession(
            twins.mirror, caption=Caption(twins.y, twins.mask), unconditional_caption=Caption(twins.y_null, twins.mask),
            steps=steps, flow_shift=5.0, action_flow_shift=1.0, video_cfg_scale=cfg_scale,
            action_cfg_scale=1.0 if action_cfg_scale is None else action_cfg_scale, model_fps=FPS, deploy_form=form,
        )
        for form in (DEPLOY_CACHED, DEPLOY_UNCACHED)
    }
    return live, ours


@pytest.mark.parametrize("memory", MEMORIES.values(), ids=list(MEMORIES))
@pytest.mark.parametrize("cfg_scale, action_cfg_scale", [(1.0, None), (2.0, 1.5), (6.0, None)])
def test_a_decision_is_the_live_sessions(memory, cfg_scale, action_cfg_scale):
    twins = Twins(memory)
    live, ours = _sessions(twins, cfg_scale, action_cfg_scale)
    prefix = twins.x(2)[:, :, :-1]
    action_mask = torch.ones(1, sm.CHUNK_ROWS, 80, dtype=torch.bool)
    action_mask[..., 66:] = False
    state = dict(anchor_state=twins.state, anchor_state_mask=torch.ones(1, 80, dtype=torch.bool), action_mask=action_mask)
    reference = {
        DEPLOY_CACHED: live.decide(prefix, generator=torch.Generator().manual_seed(9), **state),
        DEPLOY_UNCACHED: live.decide_uncached(prefix, generator=torch.Generator().manual_seed(9), **state),
    }
    for form, session in ours.items():
        action, target = session.decide(prefix, generator=torch.Generator().manual_seed(9), **state)
        torch.testing.assert_close(target, reference[form][1], **CLOSE, msg=lambda m: f"{form} target: {m}")
        torch.testing.assert_close(action, reference[form][0], **CLOSE, msg=lambda m: f"{form} action: {m}")
        assert target.shape == (1, C_LAT, 1, *GRID) and action.shape == (1, sm.CHUNK_ROWS, 80)
        assert not bool(action[..., 66:].any()) and session.decision == 1


def test_a_session_refuses_a_fixed_prefix_of_another_size():
    twins = Twins(MEMORIES["fixed"])
    _, ours = _sessions(twins, 6.0, None)
    mask = torch.ones(1, sm.CHUNK_ROWS, 80, dtype=torch.bool)
    with pytest.raises(ValueError, match="holds 2 latents, the prefix 3"):
        ours[DEPLOY_CACHED].decide(twins.x(3)[:, :, :-1], anchor_state=twins.state, anchor_state_mask=torch.ones(80),
                                   action_mask=mask)
