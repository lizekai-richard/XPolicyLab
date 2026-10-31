import numpy as np
import torch

from ..Liber_0_lite.liber0.preprocessing import denormalize
from .preprocessing import prepare_images


@torch.no_grad()
def infer_action_chunk(policy, observation, instruction, request_seed, bbox):
    images = [observation[key] for key in ('head_camera_rgb', 'left_camera_rgb', 'right_camera_rgb', 'visual_cue_rgb')]
    image = prepare_images(images, policy.resize, policy.image_crop_scale, bbox, policy.device)
    state = torch.from_numpy(np.asarray(observation['joint_action_vector'], dtype=np.float32).reshape(-1).copy()).to(policy.device)
    state = torch.clamp(state * policy.norm['state_scale'] + policy.norm['state_offset'], -5, 5).unsqueeze(0).to(torch.bfloat16)
    output = policy.model.infer_action(
        prompt="A video recorded from a robot's point of view executing the following instruction: " + instruction,
        input_image=image, proprio=state, action_horizon=policy.action_horizon,
        num_inference_steps=policy.steps, seed=request_seed, rand_device='cpu',
    )['action'].detach().to(device='cpu', dtype=torch.float32)
    return denormalize(output, policy.norm['action_scale'], policy.norm['action_offset']).numpy()
