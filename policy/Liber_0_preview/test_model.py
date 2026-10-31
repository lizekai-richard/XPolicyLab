import importlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from ..Liber_0_lite.test_model import Engine, observation
from ..Liber_0_lite.test_model import test_official_websocket_and_deploy_loop_with_mock_engine
from ..Liber_0_lite.liber0.preprocessing import pack_images
from .generation import HighLevelDecision, HighLevelEngine, HierarchyState
from .preprocessing import prepare_images

adapter = importlib.import_module('XPolicyLab.policy.Liber_0_preview.model')
base = importlib.import_module('XPolicyLab.policy.Liber_0_lite.model')


def decision(text, bbox=None):
    return HighLevelDecision('', '', text, None, bbox, bbox is not None)


@pytest.fixture
def policy(monkeypatch, tmp_path):
    cfg = yaml.safe_load((Path(__file__).parent / 'deploy.yml').read_text())
    cfg.update(device='cpu', action_type='joint', env_cfg_type='arx_x5', seed=0,
               ckpt_name=str(tmp_path), context_model_path=str(tmp_path), bench_name='RoboDojo')
    engine = Engine()
    monkeypatch.setattr(base, '_load_policy', lambda cfg, root: engine)
    monkeypatch.setattr(base, 'get_robot_action_dim_info', lambda env: {'arm_dim': [6, 6], 'ee_dim': [1, 1]})
    context = Mock()
    context.decide_batch.return_value = [decision(observation()['instruction'])]
    monkeypatch.setattr(adapter, 'HighLevelEngine', lambda *a, **kw: context)
    monkeypatch.setattr(adapter, 'infer_action_chunk', lambda p, obs, text, seed, bbox: p.infer_action_chunk(obs, text, seed))
    return adapter.Model(cfg)


def test_lock_and_reset(policy):
    policy.update_obs(observation(10))
    assert len(policy.get_action()) == 16
    policy.update_obs(observation(40))
    policy.get_action()
    assert policy.engine.decide_batch.call_count == 1
    assert np.all(policy.policy.calls[-1][0]['visual_cue_rgb'] == 10)
    assert [row[2] for row in policy.policy.calls] == [0, 1]
    policy.reset()
    policy.update_obs(observation(60))
    policy.get_action()
    assert policy.engine.decide_batch.call_count == 2
    assert np.all(policy.policy.calls[-1][0]['visual_cue_rgb'] == 60)


def test_transitions(policy):
    policy.engine.decide_batch.side_effect = [[decision('Pick up the cup.')],
        [decision('Pick up the cup.')], [decision('Place the cup.')]]
    for value in (10, 20, 30):
        policy.update_obs(observation(value))
        policy.get_action()
    assert [int(row[0]['visual_cue_rgb'][0, 0, 0]) for row in policy.policy.calls] == [10, 10, 30]
    assert policy.context.memory == ['Pick up the cup.']
    assert policy.instruction == observation()['instruction']
    assert policy.policy.calls[-1][1].endswith('Current subtask instruction: Place the cup.')


def test_initial_parse_failure_surfaces(policy):
    policy.engine.decide_batch.return_value = [decision(None)]
    policy.update_obs(observation())
    with pytest.raises(RuntimeError):
        policy.get_action()
    assert not policy.policy.calls


def test_batch_rejected(policy):
    with pytest.raises(NotImplementedError):
        policy.update_obs_batch([observation()])
    with pytest.raises(NotImplementedError):
        policy.get_action_batch()


def test_instruction_change_requires_reset(policy):
    policy.update_obs(observation())
    policy.get_action()
    obs = observation()
    obs['instruction'] = 'Another task.'
    policy.update_obs(obs)
    with pytest.raises(ValueError, match='without reset'):
        policy.get_action()


def test_image_parity_and_box_scope():
    rng = np.random.default_rng(4)
    images = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
    expected = pack_images(images, [240, 320], .95).bfloat16()
    assert torch.equal(prepare_images(images, [240, 320], .95, None, 'cpu'), expected)
    drawn = prepare_images(images, [240, 320], .95, (.2, .3, .7, .8), 'cpu')
    assert torch.equal(drawn[:, :, 144:], expected[:, :, 144:])
    assert torch.equal(drawn[:, :, :144, :192], expected[:, :, :144, :192])
    assert not torch.equal(drawn[:, :, :144, 192:], expected[:, :, :144, 192:])


def test_bootstrap_and_dynamic_state():
    engine = HighLevelEngine.__new__(HighLevelEngine)
    engine._generate = Mock(side_effect=[['<planning>["Pick", "Place"]</planning><subtask>Pick</subtask>'],
                                        ['<subtask>Pick</subtask><bbox>0.1 0.2 0.3 0.4</bbox>']])
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    state = HierarchyState('Arrange objects.')
    result = engine.decide_batch([state], [[image] * 3])[0]
    assert result.plan == ['Pick', 'Place'] and result.bootstrap_used
    assert result.bbox == (.1, .2, .3, .4)
    state = HierarchyState('Play tic-tac-toe.')
    state.apply(decision('Place in cell 1.'), image)
    assert not state.apply(decision('Place in cell 2.'), image + 1)
    assert state.active_l2 == 'Place in cell 1.'
