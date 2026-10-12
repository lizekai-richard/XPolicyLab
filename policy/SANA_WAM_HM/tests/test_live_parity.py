"""Live parity of the hierarchical-memory mirror against Sana's causal policy at rwm/sparse-memory (``SANA_HM_REPO``).

The live class and the mirror get identical random weights (tiny dims, fp64, CPU, xformers off, ``flash_attn`` bound to an
fp64 reference varlen as Sana's own tests do; softmax attention in fp64 on both sides). Checked, for the two recipe
memories ({gdn: dense, softmax: sparse} and {gdn: sparse, softmax: sparse}) and the two other legal ones:

* the training forward (``model(x, t, y, data_info)``) == ``forward_hm(form='train')``, K = 0..3, every token;
* Sana's deploy forms (``forward_chunk`` commit / generate) == ``forward_hm(form='commit' / 'generate')`` on the episode
  clock, and the committed GDN states;
* the deploy chain on the episode clock == the training forward of the same decision up to round-off (``EPISODE_CLOCK``) (the property
  Sana's ``test_the_episode_clock_is_the_relative_clock_up_to_round_off`` asserts for its own forms);
* the adapter's commit-once forms (every memory latent committed once through the blocks that read its memory, then the
  noisy chunk alone) == the training forward for every memory choice, and == Sana's generate form on the same clock;
* the session: a whole decision cycle (Euler loop, CFG, both deploy forms, every memory choice) == the same loop driven
  by the live training forward.

Run on its own (the first file of a session owns the ``dev`` package).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import _paths  # noqa: F401
import _live

from sana_wam_hm import memory as hm
from sana_wam_hm.contract import DEPLOY_FULL, DEPLOY_STREAMING, HMContract
from sana_wam_hm.layers import GDN_KIND, GDNCache, SoftmaxCache
from sana_wam_hm.model import HMPolicyModel
from sana_wam_hm.session import Caption, HMSession
from sana_wam_min.policy_model.config import PolicyConfig
from sana_wam_min.sampler import _action_step, _video_step, make_scheduler

FPS = 16.0
C_LAT = 4
GRID = (10, 16)
TEXT_LEN, TEXT_CH = 7, 16
M = hm.SPARSE_FRAMES
SPLIT = {"gdn": "dense", "softmax": "sparse", "sparse_frames": M}
SPARSE = {"gdn": "sparse", "softmax": "sparse", "sparse_frames": M}
DENSE_DENSE = {"gdn": "dense", "softmax": "dense", "sparse_frames": M}
SPARSE_DENSE = {"gdn": "sparse", "softmax": "dense", "sparse_frames": M}
CLOSE = {"rtol": 1e-9, "atol": 1e-12}
# the episode clock shifts every position by t - r0: the GDN's complex64 RoPE then rounds differently (measured flat in c,
# max |diff| 1e-10 .. 3e-10 on O(1) outputs); Sana's own episode-clock test uses atol 1e-10 at one shift
EPISODE_CLOCK = {"rtol": 1e-8, "atol": 1e-9}
# a whole decision on the episode clock vs the training loop: the diffusers Euler step runs in float32, so a 1e-10
# forward difference can flip the last float32 bit of the updated sample (one ulp ~ 6e-8 at |x| ~ 0.6)
EPISODE_CLOCK_LOOP = {"rtol": 1e-6, "atol": 5e-7}


@pytest.fixture(autouse=True)
def _fake_flash_attn(monkeypatch):
    _live.live_causal()
    _live.bind_reference_varlen(monkeypatch)


class _Cfg(SimpleNamespace):
    def __getattr__(self, name):
        return None


class Twins:
    """A tiny live hierarchical-memory policy and the mirror with the same weights (fp64)."""

    def __init__(self, memory, *, separate=True, depth=4, seed=7, softmax_layer_indices=None):
        net = _live.live_causal()
        Live = net.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal
        self.memory = dict(memory)
        self.streams = hm.memory_streams(self.memory)
        self.separate = separate
        torch.manual_seed(seed)
        kwargs = dict(
            depth=depth, hidden_size=64, patch_size=(1, 1, 1), num_heads=2, in_channels=C_LAT, pred_sigma=False,
            caption_channels=TEXT_CH, model_max_length=TEXT_LEN, linear_head_dim=32, softmax_head_dim=32,
            softmax_ratio=0.5, attn_res_block_size=2, use_time_conditioning=False, y_norm=True, cross_norm=True,
        )
        if softmax_layer_indices is not None:
            kwargs["softmax_layer_indices"] = list(softmax_layer_indices)
        policy = {"actions_per_chunk": {str(int(FPS)): hm.CHUNK_ROWS}, "memory_content": "video_only",
                  hm.MEMORY_KEY: dict(self.memory)}
        config = _Cfg(
            model=_Cfg(extra={"chunk_causal_policy": policy, "rope": "aligned"}),
            scheduler=_Cfg(action_flow_shift=1.0 if separate else None),
            data=_Cfg(extra={"multiview": "sana_pixel"}),
            vae=_Cfg(vae_stride=[8, 32, 32]),
        )
        self.live = Live(config=config, **kwargs)
        assert self.live.hierarchical_memory == self.memory
        with torch.no_grad():
            for parameter in self.live.parameters():
                parameter.uniform_(-0.05, 0.05)
        self.live = self.live.double().eval()
        contract = HMContract(
            actions_per_chunk=hm.CHUNK_ROWS, video_frame_stride=4, gdn_memory=self.memory["gdn"],
            softmax_memory=self.memory["softmax"], sparse_frames=M, model_fps=FPS, tier_num_frames=33,
            multiview="sana_pixel", visual_layout="sana_pixel_canvas", rope="aligned", dt_bias_init=-5.0,
            a_log_init=0.0, reset_legacy_beta=False,
        )
        self.mirror = HMPolicyModel(PolicyConfig.from_sana_kwargs(config=config, **kwargs), contract)
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
        self.dense = [randn(1, C_LAT, 1, *GRID) for _ in range(6)]          # L_0..L_5
        self.sparse = [randn(1, C_LAT, hm.sparse_latents(M), *GRID) for _ in range(6)]   # [A_0..A_2] per decision
        self.obs = [randn(1, C_LAT, 1, *GRID) for _ in range(6)]
        self.target = randn(1, C_LAT, 1, *GRID)
        self.action = randn(1, hm.CHUNK_ROWS, 80)
        self.state = randn(1, 80)
        self.t_video = 700.0
        self.t_action = 900.0 if separate else 700.0

    # -- live forwards -------------------------------------------------------------------------------------------------

    def info(self, rows, *, k, robot=True, t_action=None):
        info = {
            "model_fps": FPS,
            "rwm_task": "policy",
            "view_count": torch.tensor([1]),
            "view_latent_shape": torch.tensor([[list(GRID)]]),
            "video_frame_stride": torch.tensor([4]),
            hm.MEMORY_KEY: {"k": k, "sparse_frames": M, "frame_offsets_x8": hm.offsets_tensor(rows)},
        }
        if robot:
            info.update(
                action80=self.action,
                action_mask80=torch.ones(1, hm.CHUNK_ROWS, 80, dtype=torch.bool),
                action_timestep=torch.full((1, hm.CHUNK_ROWS), self.t_action if t_action is None else t_action, dtype=torch.float64),
                initial_state80=self.state,
                initial_state_condition_mask80=torch.ones(1, 80, dtype=torch.bool),
            )
        return info

    def robot(self, t_action=None):
        return {
            "state80": self.state,
            "state_mask80": torch.ones(1, 80, dtype=torch.bool),
            "action80": self.action,
            "action_mask80": torch.ones(1, hm.CHUNK_ROWS, 80, dtype=torch.bool),
            "action_timesteps": torch.full((1, hm.CHUNK_ROWS), self.t_action if t_action is None else t_action, dtype=torch.float64),
        }

    def training_x(self, c, target=None):
        """Decision c's training sequence [L_0..L_c | A_0.. | obs | target] (t = r0 + 32c, r0 = 0)."""

        parts = []
        if hm.DENSE in self.streams:
            parts += self.dense[: c + 1]
        if hm.SPARSE in self.streams:
            parts.append(self.sparse[c])
        return torch.cat(parts + [self.obs[c], self.target if target is None else target], dim=2)

    def timesteps(self, frames, t=None):
        out = torch.zeros(1, frames, dtype=torch.float64)
        out[:, -1] = self.t_video if t is None else t
        return out

    def live_train(self, c, target=None, t_video=None, t_action=None):
        x = self.training_x(c, target)
        rows = hm.train_rows(c, M, self.streams)
        with torch.no_grad():
            return self.live(
                x, self.timesteps(x.shape[2], t_video), self.y, mask=self.mask,
                data_info=self.info(rows, k=c, t_action=t_action),
            )

    def mirror_train(self, c, target=None):
        x = self.training_x(c, target)
        return self.mirror.forward_hm(
            x, self.timesteps(x.shape[2]), self.y, self.mask, form=hm.FORM_TRAIN, k=c,
            frame_offsets_x8=hm.offsets_tensor(hm.train_rows(c, M, self.streams)), model_fps=FPS, robot=self.robot(),
        )

    # -- the episode clock (deploy forms) -------------------------------------------------------------------------------

    def live_deploy(self, c):
        """Sana ``forward_chunk``: commit L_0..L_c at 32j, then generate decision c on the episode clock."""

        caches = [{} for _ in self.live.blocks]
        commits = []
        for j in range(c + 1):
            strip = self.dense[j].flatten(3).unsqueeze(3)
            commits.append(self.live.forward_chunk(
                strip, torch.zeros(1, 1, 1, dtype=torch.float64), self.y, mask=self.mask, kv_caches=caches,
                save_kv_cache=True, chunk_offset=j, cached_past_tokens=hm.LATENT_TOKENS * j,
                data_info=self.info([hm.commit_rows(j)], k=j, robot=False),
            ))
        window = torch.cat([self.sparse[c], self.obs[c], self.target], dim=2)
        generate = self.live.forward_chunk(
            window.flatten(3).unsqueeze(3), self.timesteps(5)[:, None], self.y, mask=self.mask, kv_caches=caches,
            save_kv_cache=False, chunk_offset=c, cached_past_tokens=hm.LATENT_TOKENS * (c + 1),
            data_info=self.info(hm.generate_rows(c, M), k=c),
        )
        return commits, generate, caches

    def mirror_deploy(self, c):
        states = {}
        gdn = [i for i, block in enumerate(self.mirror.blocks) if block.attn_type == GDN_KIND]
        for j in range(c + 1):
            caches = [GDNCache(seed=states.get(i), commit=True) if i in gdn else None for i in range(len(self.mirror.blocks))]
            out = self.mirror.forward_hm(
                self.dense[j], torch.zeros(1, 1, dtype=torch.float64), self.y, self.mask, form=hm.FORM_COMMIT, k=j,
                frame_offsets_x8=hm.offsets_tensor([hm.commit_rows(j)]), model_fps=FPS, caches=caches,
            )
            assert out == {}
            states = {i: caches[i].out for i in gdn}
        caches = [GDNCache(seed=states.get(i)) if i in gdn else None for i in range(len(self.mirror.blocks))]
        window = torch.cat([self.sparse[c], self.obs[c], self.target], dim=2)
        generate = self.mirror.forward_hm(
            window, self.timesteps(5), self.y, self.mask, form=hm.FORM_GENERATE, k=c,
            frame_offsets_x8=hm.offsets_tensor(hm.generate_rows(c, M)), model_fps=FPS,
            step_offset=hm.CHUNK_ROWS * c, robot=self.robot(), caches=caches,
        )
        return generate, states


def commit_once(twins, c, clock="deploy"):
    """Decision c through the adapter's commit-once forms: every memory latent committed once (dense L_0..L_c one by
    one, then the sparse memory of decision c), then the noisy chunk alone -- on the deploy's clock (the episode clock
    with a dense memory, the training clock without), or with ``clock='training'`` every latent at its training offset
    from decision c's t (an emulation for one fixed t, which makes the comparison with training exact)."""

    mirror, memory = twins.mirror, twins.memory
    episode = hm.episode_clock(twins.streams)
    training = hm.train_rows(c, M, twins.streams)
    dense_count = c + 1 if hm.DENSE in twins.streams else 0
    kinds = [hm.GDN if block.attn_type == GDN_KIND else hm.SOFTMAX for block in mirror.blocks]
    held = {kind: {"gdn": {}, "kv": {}} for kind in twins.streams}

    def commit(kind, latents, rows, fresh):
        state = {"gdn": {}, "kv": {}} if fresh else held[kind]
        caches = []
        for index, block in enumerate(kinds):
            if memory[block] != kind:
                caches.append(None)
            elif block == hm.GDN:
                caches.append(GDNCache(seed=state["gdn"].get(index), commit=True))
            else:
                caches.append(SoftmaxCache(prefix=state["kv"].get(index), commit=True))
        out = mirror.forward_hm(
            latents, torch.zeros(1, latents.shape[2], dtype=torch.float64), twins.y, twins.mask, form=hm.FORM_MEMORY,
            k=0, frame_offsets_x8=hm.offsets_tensor(rows), model_fps=FPS, caches=caches, memory_kind=kind,
        )
        assert out == {}
        held[kind] = {
            "gdn": {i: cache.out for i, cache in enumerate(caches) if isinstance(cache, GDNCache)},
            "kv": {i: cache.out for i, cache in enumerate(caches) if isinstance(cache, SoftmaxCache)},
        }

    if hm.DENSE in twins.streams:
        for j in range(c + 1):
            rows = [training[j]] if clock == "training" else [hm.commit_rows(j)]
            commit(hm.DENSE, twins.dense[j], rows, fresh=j == 0)
    if hm.SPARSE in twins.streams:
        rows = training[dense_count : dense_count + hm.sparse_latents(M)] if clock == "training" else hm.sparse_memory_rows(c, M, episode)
        commit(hm.SPARSE, twins.sparse[c], rows, fresh=True)
    if clock == "training":
        episode = False
    caches = [
        GDNCache(seed=held[memory[block]]["gdn"].get(i)) if block == hm.GDN else SoftmaxCache(prefix=held[memory[block]]["kv"].get(i))
        for i, block in enumerate(kinds)
    ]
    return mirror.forward_hm(
        torch.cat([twins.obs[c], twins.target], dim=2), twins.timesteps(2), twins.y, twins.mask, form=hm.FORM_NOISY,
        k=0, frame_offsets_x8=hm.offsets_tensor(hm.noisy_rows(c, episode)), model_fps=FPS,
        step_offset=hm.noisy_step_offset(c, episode), robot=twins.robot(), caches=caches,
    )


def _frames(out, lo, hi):
    return out["x"][:, :, lo:hi]


# -- the training form ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("memory", [SPLIT, SPARSE, DENSE_DENSE, SPARSE_DENSE], ids=["dense_sparse", "sparse_sparse", "dense_dense", "sparse_dense"])
@pytest.mark.parametrize("c", [0, 1, 3])
def test_the_training_forward_is_the_live_one(memory, c):
    twins = Twins(memory)
    live = twins.live_train(c)
    ours = twins.mirror_train(c)
    torch.testing.assert_close(ours["x"], live["x"].reshape(ours["x"].shape), **CLOSE)
    torch.testing.assert_close(ours["action_pred"], live["action_pred"], **CLOSE)


def test_a_shared_schedule_runs_too():
    twins = Twins(SPLIT, separate=False)
    live, ours = twins.live_train(2), twins.mirror_train(2)
    torch.testing.assert_close(ours["x"], live["x"].reshape(ours["x"].shape), **CLOSE)
    torch.testing.assert_close(ours["action_pred"], live["action_pred"], **CLOSE)


# -- the streaming forms ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0, 1, 3])
def test_commit_and_generate_are_the_live_forward_chunk_on_the_episode_clock(c):
    twins = Twins(SPLIT)
    _commits, live, live_caches = twins.live_deploy(c)
    ours, states = twins.mirror_deploy(c)
    torch.testing.assert_close(ours["x"].flatten(3), live["x"].flatten(3), **CLOSE)
    torch.testing.assert_close(ours["action_pred"], live["action_pred"], **CLOSE)
    for index, (S, z) in states.items():
        torch.testing.assert_close(S, live_caches[index]["S"], **CLOSE)
        torch.testing.assert_close(z, live_caches[index]["z"], **CLOSE)
    for index, cache in enumerate(live_caches):
        if twins.live.block_attn_types[index] == twins.live.softmax_attn_type:
            assert cache == {}


@pytest.mark.parametrize("c", [0, 1, 2, 4])
def test_the_episode_clock_reproduces_the_training_forward_up_to_round_off(c):
    """Streaming decision c (commits L_0..L_c at 32j, generate at 32c) == the training forward of the same sample."""

    twins = Twins(SPLIT)
    training = twins.live_train(c)
    ours, _ = twins.mirror_deploy(c)
    layout = hm.frame_layout(c, M, twins.streams)
    torch.testing.assert_close(ours["x"], _frames(training, layout[hm.SPARSE][0], layout["frames"]).reshape(ours["x"].shape), **EPISODE_CLOCK)
    torch.testing.assert_close(ours["action_pred"], training["action_pred"], **EPISODE_CLOCK)


@pytest.mark.parametrize(
    "memory,tolerance",
    [(SPLIT, "episode"), (SPARSE, "exact"), (DENSE_DENSE, "episode"), (SPARSE_DENSE, "episode")],
    ids=["dense_sparse", "sparse_sparse", "dense_dense", "sparse_dense"],
)
@pytest.mark.parametrize("c", [0, 1, 3])
def test_commit_once_is_the_training_forward(memory, tolerance, c):
    """Every memory latent committed once, then the noisy chunk alone == the training forward's noisy chunk (exactly on
    the training clock, up to the episode clock's round-off with a dense memory)."""

    twins = Twins(memory)
    training = twins.live_train(c)
    ours = commit_once(twins, c)
    close = CLOSE if tolerance == "exact" else EPISODE_CLOCK
    frames = training["x"].shape[2]
    torch.testing.assert_close(ours["x"][:, :, -1:], training["x"][:, :, frames - 1 :].reshape(ours["x"][:, :, -1:].shape), **close)
    torch.testing.assert_close(ours["x"][:, :, :1], training["x"][:, :, frames - 2 : frames - 1].reshape(ours["x"][:, :, :1].shape), **close)
    torch.testing.assert_close(ours["action_pred"], training["action_pred"], **close)


GDN_LED = None          # the default tiny layout: softmax blocks at 1, 3, every AttnRes group (size 2) opens with a GDN block
SOFTMAX_LED = (0, 2)   # softmax blocks at 0, 2: every AttnRes group opens with a softmax block (never so in the real trunk)


def _record_block_inputs(mirror, run):
    """Run ``run()`` and return, per block, the token subsets its attention sublayer received (one entry per forward)."""

    records = {index: [] for index in range(len(mirror.blocks))}
    for index, block in enumerate(mirror.blocks):
        original = block.forward_attn_sublayer

        def wrapped(hidden_states, *args, _original=original, _index=index, **kwargs):
            records[_index].append(hidden_states.detach().clone())
            return _original(hidden_states, *args, **kwargs)

        block.forward_attn_sublayer = wrapped
    try:
        out = run()
    finally:
        for block in mirror.blocks:
            del block.forward_attn_sublayer
    return records, out


@pytest.mark.parametrize("layout", [GDN_LED, SOFTMAX_LED], ids=["gdn-led-groups", "softmax-led-groups"])
@pytest.mark.parametrize(
    "memory", [SPLIT, SPARSE_DENSE, SPARSE, DENSE_DENSE], ids=["dense_sparse", "sparse_dense", "sparse_sparse", "dense_dense"]
)
def test_memory_tokens_see_zero_deltas_in_the_blocks_that_do_not_read_them(memory, layout):
    """Training's routing semantics, block by block: a memory token passes a block that does not read its memory with a
    ZERO delta (dense_sparse: the GDN blocks see A with zero deltas, the softmax blocks see L with zero deltas;
    sparse_dense the mirror image; sparse_sparse / dense_dense: every memory token is in every block). The commits must
    reproduce, at the input of every block that reads a memory, that memory's tokens of the training forward -- also
    where a softmax block opens an AttnRes group, so a zero-delta L must open the group's partial."""

    c = 2
    twins = Twins(memory, softmax_layer_indices=layout)
    mirror = twins.mirror
    train_inputs, _ = _record_block_inputs(mirror, lambda: twins.mirror_train(c))
    commit_inputs, ours = _record_block_inputs(mirror, lambda: commit_once(twins, c, clock="training"))
    kinds = [hm.GDN if block.attn_type == GDN_KIND else hm.SOFTMAX for block in mirror.blocks]
    sizes = {hm.DENSE: hm.LATENT_TOKENS * (c + 1), hm.SPARSE: hm.LATENT_TOKENS * hm.sparse_latents(M)}
    for index, kind in enumerate(kinds):
        read = memory[kind]
        (train,) = train_inputs[index]
        memory_rows = sizes[read]
        committed = torch.cat(commit_inputs[index][:-1], dim=1)     # every commit forward through this block
        noisy = commit_inputs[index][-1]                            # the noisy forward
        assert committed.shape[1] == memory_rows, (index, committed.shape)
        torch.testing.assert_close(committed, train[:, :memory_rows], **CLOSE, msg=lambda m: f"block {index} ({kind}) memory: {m}")
        torch.testing.assert_close(noisy, train[:, memory_rows:], **CLOSE, msg=lambda m: f"block {index} ({kind}) noisy: {m}")
    live = twins.live_train(c)
    torch.testing.assert_close(ours["action_pred"], live["action_pred"], **CLOSE)
    torch.testing.assert_close(ours["x"][:, :, -1:], live["x"][:, :, -1:].reshape(ours["x"][:, :, -1:].shape), **CLOSE)


@pytest.mark.parametrize("c", [0, 2])
def test_commit_once_is_sanas_generate_form_on_the_same_clock(c):
    """The sparse memory committed once == Sana's generate window, where it rides the forward (same episode clock)."""

    twins = Twins(SPLIT)
    generate, _ = twins.mirror_deploy(c)
    ours = commit_once(twins, c)
    torch.testing.assert_close(ours["x"], generate["x"][:, :, -2:], **CLOSE)
    torch.testing.assert_close(ours["action_pred"], generate["action_pred"], **CLOSE)


def test_sanas_forms_take_no_softmax_cache():
    twins = Twins(SPLIT)
    caches = [GDNCache() if block.attn_type == GDN_KIND else SoftmaxCache() for block in twins.mirror.blocks]
    window = torch.cat([twins.sparse[0], twins.obs[0], twins.target], dim=2)
    with pytest.raises(ValueError, match="keeps no softmax"):
        twins.mirror.forward_hm(
            window, twins.timesteps(5), twins.y, twins.mask, form=hm.FORM_GENERATE, k=0,
            frame_offsets_x8=hm.offsets_tensor(hm.generate_rows(0, M)), model_fps=FPS, robot=twins.robot(), caches=caches,
        )


def test_the_streaming_forms_are_refused_off_their_memory():
    twins = Twins(SPARSE)
    with pytest.raises(ValueError, match="streaming"):
        twins.mirror.forward_hm(
            twins.dense[0], torch.zeros(1, 1, dtype=torch.float64), twins.y, twins.mask, form=hm.FORM_COMMIT, k=0,
            frame_offsets_x8=hm.offsets_tensor([0.0]), model_fps=FPS, caches=[None] * len(twins.mirror.blocks),
        )


def test_the_window_shapes_and_the_lockstep_are_checked():
    twins = Twins(SPLIT)
    x = twins.training_x(1)
    rows = hm.offsets_tensor(hm.train_rows(1, M, twins.streams))
    with pytest.raises(ValueError, match="latent frames"):
        twins.mirror.forward_hm(x[:, :, 1:], twins.timesteps(x.shape[2] - 1), twins.y, twins.mask, form=hm.FORM_TRAIN, k=1,
                                frame_offsets_x8=rows[:, 1:], model_fps=FPS, robot=twins.robot())
    noisy_memory = twins.timesteps(x.shape[2])
    noisy_memory[:, 0] = 5.0
    with pytest.raises(ValueError, match="only the target"):
        twins.mirror.forward_hm(x, noisy_memory, twins.y, twins.mask, form=hm.FORM_TRAIN, k=1, frame_offsets_x8=rows,
                                model_fps=FPS, robot=twins.robot())
    with pytest.raises(ValueError, match="takes no"):
        twins.mirror.forward_hm(x, twins.timesteps(x.shape[2]), twins.y, twins.mask, form=hm.FORM_TRAIN, k=1,
                                frame_offsets_x8=rows, model_fps=FPS, robot=twins.robot(), caches=[None] * 4)


# -- the session -------------------------------------------------------------------------------------------------------------


def _session(twins, form, steps=3):
    return HMSession(
        twins.mirror, caption=Caption(twins.y, twins.mask), unconditional_caption=Caption(twins.y_null, twins.mask),
        steps=steps, flow_shift=5.0, action_flow_shift=1.0, video_cfg_scale=6.0, action_cfg_scale=1.0, model_fps=FPS,
        deploy_form=form,
    )


def _live_loop(twins, c, video_noise, action_noise, steps=3, video_cfg=6.0):
    """The decision's Euler loop driven by the LIVE training forward (the reference of both deploy forms)."""

    video, action = video_noise.clone(), action_noise.clone()
    mask = torch.ones(1, hm.CHUNK_ROWS, 80, dtype=torch.bool)
    vs, as_ = make_scheduler(steps, 5.0, "cpu"), make_scheduler(steps, 1.0, "cpu")
    for i, timestep in enumerate(vs.timesteps):
        a_timestep = as_.timesteps[i]
        twins.action = action
        outs = []
        for y in (twins.y, twins.y_null):
            saved = twins.y
            twins.y = y
            outs.append(twins.live_train(c, target=video, t_video=float(timestep), t_action=float(a_timestep)))
            twins.y = saved
        cond, uncond = outs
        pred = uncond["x"][:, :, -1:] + video_cfg * (cond["x"][:, :, -1:] - uncond["x"][:, :, -1:])
        video_t = timestep.reshape(1, 1).to(torch.float32)
        action_t = a_timestep.reshape(1, 1).expand(1, hm.CHUNK_ROWS).to(torch.float32)
        video = _video_step(vs, pred.reshape(video.shape), timestep, video, video_t)
        action = _action_step(as_, cond["action_pred"], a_timestep, action, action_t).masked_fill(~mask, 0)
    return action, video


@pytest.mark.parametrize(
    "memory,form,tolerance",
    [
        (SPLIT, DEPLOY_STREAMING, EPISODE_CLOCK_LOOP),
        (SPLIT, DEPLOY_FULL, CLOSE),
        (SPARSE, DEPLOY_STREAMING, CLOSE),
        (SPARSE, DEPLOY_FULL, CLOSE),
        (DENSE_DENSE, DEPLOY_STREAMING, EPISODE_CLOCK_LOOP),
        (DENSE_DENSE, DEPLOY_FULL, CLOSE),
        (SPARSE_DENSE, DEPLOY_STREAMING, EPISODE_CLOCK_LOOP),
    ],
    ids=[
        "dense_sparse-streaming", "dense_sparse-full", "sparse_sparse-streaming", "sparse_sparse-full",
        "dense_dense-streaming", "dense_dense-full", "sparse_dense-streaming",
    ],
)
def test_the_session_is_the_live_training_loop(memory, form, tolerance):
    twins = Twins(memory)
    session = _session(twins, form)
    mask = torch.ones(1, hm.CHUNK_ROWS, 80, dtype=torch.bool)
    g = torch.Generator().manual_seed(5)
    for c in range(3):
        if session.reads_dense:
            session.add_dense(twins.dense[c])
        video_noise = torch.randn(1, C_LAT, 1, *GRID, generator=g, dtype=torch.float64)
        action_noise = torch.randn(1, hm.CHUNK_ROWS, 80, generator=g, dtype=torch.float64)
        ours = session.generate(
            twins.obs[c], twins.sparse[c] if session.reads_sparse else None, anchor_state=twins.state,
            anchor_state_mask=torch.ones(80, dtype=torch.bool), action_mask=mask, video_noise=video_noise,
            action_noise=action_noise,
        )
        reference = _live_loop(twins, c, video_noise, action_noise)
        torch.testing.assert_close(ours[0], reference[0], **tolerance, msg=lambda m: f"decision {c} action: {m}")
        torch.testing.assert_close(ours[1], reference[1], **tolerance, msg=lambda m: f"decision {c} video: {m}")
    assert session.decision == 3


def test_the_session_refuses_a_broken_cycle():
    twins = Twins(SPLIT)
    session = _session(twins, DEPLOY_STREAMING)
    mask = torch.ones(1, hm.CHUNK_ROWS, 80, dtype=torch.bool)
    kwargs = dict(anchor_state=twins.state, anchor_state_mask=torch.ones(80, dtype=torch.bool), action_mask=mask)
    with pytest.raises(RuntimeError, match="L_0..L_0"):
        session.generate(twins.obs[0], twins.sparse[0], **kwargs)
    session.add_dense(twins.dense[0])
    with pytest.raises(ValueError, match="sparse memory"):
        session.generate(twins.obs[0], None, **kwargs)
    with pytest.raises(RuntimeError, match="right before decision"):
        session.add_dense(twins.dense[1])
    with pytest.raises(RuntimeError, match="no dense memory"):
        _session(Twins(SPARSE), DEPLOY_STREAMING).add_dense(twins.dense[0])
