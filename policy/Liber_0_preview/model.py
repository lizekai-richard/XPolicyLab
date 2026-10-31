import re

import numpy as np

from XPolicyLab.utils.process_data import unpack_robot_state
from ..Liber_0_lite.model import Model as BaseModel, _required_path
from .generation import HighLevelEngine, HierarchyState
from .runtime import infer_action_chunk


class Model(BaseModel):
    def __init__(self, model_cfg):
        if model_cfg['eval_batch']:
            raise NotImplementedError('Liber_0_preview requires eval_batch=false.')
        super().__init__(model_cfg)
        self.engine = HighLevelEngine(
            _required_path(model_cfg, 'context_model_path'), device=model_cfg['device'],
            max_seq_len=1536, target_pixels=336 * 336,
            max_new_tokens=384, attn_implementation='flash_attention_2',
        )

    def reset(self):
        super().reset()
        self.context = None
        self.locked = False

    def get_action(self):
        if self.observation is None:
            raise RuntimeError('Call update_obs after reset before requesting actions.')
        if self.context is None:
            self.context = HierarchyState(l3=self.instruction)
        if self.context.l3 != self.instruction:
            raise ValueError('Instruction changed without reset.')
        observation = dict(self.observation)
        instruction = self.instruction
        bbox = None
        use_original = self.locked
        if not self.locked:
            images = [observation[key] for key in ('head_camera_rgb', 'left_camera_rgb', 'right_camera_rgb')]
            decision = self.engine.decide_batch([self.context], [images])[0]
            self.context.apply(decision, images[0])
            use_original = (
                re.sub(r'\s+', ' ', instruction.strip()).casefold().rstrip('.!?。！？ ')
                == re.sub(r'\s+', ' ', self.context.active_l2.strip()).casefold().rstrip('.!?。！？ ')
            )
            if self.replan_index == 0 and use_original:
                self.locked = True
        if not use_original:
            observation['visual_cue_rgb'] = self.context.cue_head_rgb.copy()
            bbox = self.context.cue_bbox
            instruction = f'{instruction.strip()}\nCurrent subtask instruction: {self.context.active_l2.strip()}'
        actions = np.asarray(infer_action_chunk(
            self.policy, observation, instruction, self.seed + self.replan_index, bbox,
        ), dtype=np.float32)
        if actions.shape != (self.policy.action_horizon, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid predicted action chunk: {actions.shape}')
        self.replan_index += 1
        return unpack_robot_state(actions[:self.replan_steps], self.action_type, self.robot_action_dim_info)

    def update_obs_batch(self, obs_list):
        raise NotImplementedError('Liber_0_preview requires eval_batch=false.')

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError('Liber_0_preview requires eval_batch=false.')
