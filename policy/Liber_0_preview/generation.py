from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import functional as F_v2
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
IMAGE_BLOCK = '<|vision_start|><|image_pad|><|vision_end|>'
DECISION_PROMPT = '<image>\n<image>\n<image>\nThe images show the current observation from the high, left-wrist, and right-wrist cameras.\nTask: {task}\n{planning_block}{memory_block}Decide what the robot should do now. If a <planning> block is provided, follow it and output only the current executable instruction inside <subtask></subtask>. If no <planning> block is provided and a complete plan can be determined from the observation and memory, output that ordered plan inside <planning></planning> before <subtask></subtask>. For interactive or dynamic tasks whose next action depends on another agent or a changing grid, a complete plan is not required: directly output <subtask></subtask>. Tasks represented by one whole-episode compound instruction also directly output that instruction as <subtask></subtask> without a plan. When target localization is required, append its normalized bounding box inside <bbox></bbox>.'

def compact_plan(plan: Sequence[str]) -> str:
    return json.dumps([str(item) for item in plan], ensure_ascii=False, separators=(',', ':'))

def render_memory(items: Sequence[str]) -> str:
    if not items:
        return ''
    return '\n'.join(['Completed subtasks:'] + [f'{index}. {item}' for index, item in enumerate(items, start=1)])

def decision_prompt(task: str, plan: Sequence[str], memory: Sequence[str]) -> str:
    planning_block = f'<planning>{compact_plan(plan)}</planning>\n' if plan else ''
    memory_text = render_memory(memory)
    memory_block = f'<memory>{memory_text}</memory>\n' if memory_text else ''
    return DECISION_PROMPT.format(task=str(task).strip(), planning_block=planning_block, memory_block=memory_block)

def normalize_text(text: str) -> str:
    return re.sub('\\s+', ' ', str(text).strip()).casefold()

def is_dynamic_grid_task(l3: str) -> bool:
    task = normalize_text(l3).replace('‑', '-').replace('–', '-')
    return 'tic-tac-toe' in task or 'tic tac toe' in task

def leading_action_verb(l2: str) -> str:
    match = re.search('[a-z]+', normalize_text(l2))
    return match.group(0) if match else normalize_text(l2)

def l2_transition_key(l3: str, l2: str) -> str:
    if is_dynamic_grid_task(l3):
        return leading_action_verb(l2)
    return normalize_text(l2)

def tag(text: str, name: str) -> str | None:
    match = re.search(f'<{name}>\\s*(.*?)\\s*</{name}>', text, flags=re.I | re.S)
    return match.group(1).strip() if match else None

def parse_plan(text: str) -> list[str] | None:
    value = tag(text, 'planning')
    if value is None:
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list) or not all((isinstance(item, str) for item in decoded)):
        return None
    return [item.strip() for item in decoded if item.strip()]

def parse_bbox(text: str) -> list[float] | None:
    value = tag(text, 'bbox')
    if value is None:
        return None
    numbers = re.findall('[-+]?(?:\\d+\\.?\\d*|\\.\\d+)', value)
    if len(numbers) != 4:
        return None
    box = [float(number) for number in numbers]
    if not all((math.isfinite(number) for number in box)):
        return None
    return box

def sanitize_bbox(box: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    if box is None or len(box) != 4:
        return None
    values = [float(value) for value in box]
    if not all((math.isfinite(value) for value in values)):
        return None
    x_min, y_min, x_max, y_max = (min(1.0, max(0.0, value)) for value in values)
    if x_min >= x_max or y_min >= y_max:
        return None
    return (x_min, y_min, x_max, y_max)

class TrainingAlignedInputs:

    def __init__(self, model_path: Path, max_seq_len: int=1536, target_pixels: int=224 * 224):
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        config = self.processor.image_processor
        self.patch_size = int(getattr(config, 'patch_size', 16))
        self.merge_size = int(getattr(config, 'merge_size', getattr(config, 'spatial_merge_size', 2)))
        self.temporal_size = int(getattr(config, 'temporal_patch_size', 2))
        self.resize_factor = self.patch_size * self.merge_size
        self.target_pixels = int(target_pixels)
        if self.target_pixels <= 0:
            raise ValueError(f'target_pixels must be positive, got {self.target_pixels}')
        self.max_seq_len = int(max_seq_len)
        if self.max_seq_len <= 0:
            raise ValueError(f'max_seq_len must be positive, got {self.max_seq_len}')

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        scale = math.sqrt(float(self.target_pixels) / float(height * width))
        new_height = max(self.resize_factor, int(round(height * scale / self.resize_factor)) * self.resize_factor)
        new_width = max(self.resize_factor, int(round(width * scale / self.resize_factor)) * self.resize_factor)
        tensor = F_v2.pil_to_tensor(image)
        tensor = F_v2.resize(tensor, size=(new_height, new_width), interpolation=InterpolationMode.BILINEAR, antialias=True)
        tensor = tensor.to(dtype=torch.float32) / 255.0
        return F_v2.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)

    def _patchify(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        value = self._preprocess(image).unsqueeze(0)
        value = value.unsqueeze(1).repeat(1, self.temporal_size, 1, 1, 1)
        batch_size, _, _, height, width = value.shape
        grid_t = 2 // self.temporal_size
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        grid = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
        patches = value.view(batch_size, grid_t, self.temporal_size, 3, grid_h // self.merge_size, self.merge_size, self.patch_size, grid_w // self.merge_size, self.merge_size, self.patch_size)
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixels = patches.reshape(batch_size * grid_t * grid_h * grid_w, 3 * self.temporal_size * self.patch_size * self.patch_size)
        return (pixels, grid)

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.processor.tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.squeeze(0)

    def build_one(self, images: Sequence[Image.Image], prompt: str) -> dict[str, torch.Tensor]:
        if len(images) != 3:
            raise ValueError(f'High-level requires exactly three images, got {len(images)}')
        pixel_parts, grid_parts = zip(*(self._patchify(image) for image in images))
        pixel_values = torch.cat(pixel_parts, dim=0)
        image_grid_thw = torch.cat(grid_parts, dim=0)
        value = prompt
        for _ in images:
            value = value.replace('<image>', IMAGE_BLOCK, 1)
        value = value.replace('<image>', '')
        prefix_text = f'<|im_start|>user\n{value}<|im_end|>\n<|im_start|>assistant\n'
        input_ids = self._tokenize(prefix_text)
        image_pad_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        counts = [int(t * height * width // self.merge_size ** 2) for t, height, width in image_grid_thw.tolist()]
        parts: list[torch.Tensor] = []
        previous = 0
        positions = (input_ids == image_pad_id).nonzero(as_tuple=True)[0]
        if len(positions) != len(images):
            raise RuntimeError(f'Expected {len(images)} image placeholders, found {len(positions)}')
        for image_index, position in enumerate(positions):
            current = int(position)
            parts.append(input_ids[previous:current])
            parts.append(torch.full((counts[image_index],), image_pad_id, dtype=input_ids.dtype))
            previous = current + 1
        parts.append(input_ids[previous:])
        input_ids = torch.cat(parts)
        raw_length = int(input_ids.numel())
        truncated = raw_length > self.max_seq_len
        print('[HighLevel/input_length] ' + json.dumps({'event': 'highlevel_input_length', 'level': 'WARNING' if truncated else 'INFO', 'input_tokens_before_truncation': raw_length, 'input_tokens_used': min(raw_length, self.max_seq_len), 'max_seq_len': self.max_seq_len, 'truncated': truncated, 'discarded_tokens': max(0, raw_length - self.max_seq_len)}), flush=True)
        input_ids = input_ids[:self.max_seq_len]
        return {'input_ids': input_ids, 'pixel_values': pixel_values, 'image_grid_thw': image_grid_thw}

    def build_batch(self, image_batches: Sequence[Sequence[Image.Image]], prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        if len(image_batches) != len(prompts) or not prompts:
            raise ValueError('High-level image and prompt batches must be non-empty and aligned')
        rows = [self.build_one(images, prompt) for images, prompt in zip(image_batches, prompts)]
        max_length = max((int(row['input_ids'].numel()) for row in rows))
        pad_id = int(self.processor.tokenizer.pad_token_id)
        input_ids = torch.full((len(rows), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            length = int(row['input_ids'].numel())
            input_ids[index, max_length - length:] = row['input_ids']
            attention_mask[index, max_length - length:] = 1
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': torch.cat([row['pixel_values'] for row in rows], dim=0), 'image_grid_thw': torch.cat([row['image_grid_thw'] for row in rows], dim=0)}

@dataclass
class HighLevelDecision:
    prompt: str
    raw_prediction: str
    subtask: str | None
    plan: list[str] | None
    bbox: tuple[float, float, float, float] | None
    bbox_was_present: bool
    bootstrap_used: bool = False
    planning_prediction: str | None = None

@dataclass
class HierarchyState:
    l3: str
    plan: list[str] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)
    active_l2: str | None = None
    cue_head_rgb: np.ndarray | None = None
    cue_bbox: tuple[float, float, float, float] | None = None
    transitions: int = 0
    parse_failures: int = 0

    def apply(self, decision: HighLevelDecision, current_head_rgb: np.ndarray) -> bool:
        if decision.plan:
            self.plan = list(decision.plan)
        if not decision.subtask:
            self.parse_failures += 1
            if self.active_l2 is None:
                raise RuntimeError('High-level failed to produce the initial <subtask>')
            return False
        changed = self.active_l2 is None or l2_transition_key(self.l3, decision.subtask) != l2_transition_key(self.l3, self.active_l2)
        if not changed:
            return False
        if self.active_l2 is not None:
            self.memory.append(self.active_l2)
            self.transitions += 1
        self.active_l2 = decision.subtask.strip()
        self.cue_head_rgb = np.array(current_head_rgb, dtype=np.uint8, order='C', copy=True)
        self.cue_bbox = decision.bbox
        return True

class HighLevelEngine:

    def __init__(self, model_path: str | Path, *, device: str='cuda', max_new_tokens: int=384, max_seq_len: int=1536, target_pixels: int=224 * 224, attn_implementation: str='flash_attention_2') -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f'High-level model directory not found: {self.model_path}')
        self.device = torch.device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.inputs = TrainingAlignedInputs(self.model_path, max_seq_len=max_seq_len, target_pixels=target_pixels)
        self.tokenizer = self.inputs.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation, device_map=str(self.device), local_files_only=True).eval()

    @staticmethod
    def _pil_images(images: Sequence[np.ndarray]) -> list[Image.Image]:
        output = []
        for index, image in enumerate(images):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
                raise ValueError(f'High-level image {index} must be uint8 HWC RGB, got {array.shape}/{array.dtype}')
            output.append(Image.fromarray(array, mode='RGB'))
        return output

    def _generate(self, image_batches: Sequence[Sequence[np.ndarray]], prompts: Sequence[str]) -> list[str]:
        pil_batches = [self._pil_images(images) for images in image_batches]
        inputs = self.inputs.build_batch(pil_batches, prompts)
        prefix_length = int(inputs['input_ids'].shape[1])
        model_inputs = {'input_ids': inputs['input_ids'].to(self.device), 'attention_mask': inputs['attention_mask'].to(self.device), 'pixel_values': inputs['pixel_values'].to(self.device, dtype=torch.bfloat16), 'image_grid_thw': inputs['image_grid_thw'].to(self.device)}
        with torch.inference_mode():
            generated = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False, eos_token_id=self.model.generation_config.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
        return [self.tokenizer.decode(row[prefix_length:].cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() for row in generated]

    @staticmethod
    def _parse(prompt: str, prediction: str) -> HighLevelDecision:
        raw_box = parse_bbox(prediction)
        return HighLevelDecision(prompt=prompt, raw_prediction=prediction, subtask=tag(prediction, 'subtask'), plan=parse_plan(prediction), bbox=sanitize_bbox(raw_box), bbox_was_present=tag(prediction, 'bbox') is not None)

    def decide_batch(self, states: Sequence[HierarchyState], image_batches: Sequence[Sequence[np.ndarray]]) -> list[HighLevelDecision]:
        if len(states) != len(image_batches) or not states:
            raise ValueError('High-level state and image batches must be non-empty and aligned')
        prompts = [decision_prompt(state.l3, state.plan, state.memory) for state in states]
        predictions = self._generate(image_batches, prompts)
        decisions = [self._parse(prompt, prediction) for prompt, prediction in zip(prompts, predictions)]
        bootstrap_indices = [index for index, decision in enumerate(decisions) if decision.plan and decision.bbox is None]
        if bootstrap_indices:
            bootstrap_prompts = [decision_prompt(states[index].l3, decisions[index].plan or states[index].plan, states[index].memory) for index in bootstrap_indices]
            bootstrap_predictions = self._generate([image_batches[index] for index in bootstrap_indices], bootstrap_prompts)
            for index, prompt, prediction in zip(bootstrap_indices, bootstrap_prompts, bootstrap_predictions):
                initial = decisions[index]
                updated = self._parse(prompt, prediction)
                if not updated.plan:
                    updated.plan = initial.plan
                updated.bootstrap_used = True
                updated.planning_prediction = initial.raw_prediction
                decisions[index] = updated
        return decisions
__all__ = ['HighLevelDecision', 'HighLevelEngine', 'HierarchyState', 'TrainingAlignedInputs', 'decision_prompt', 'normalize_text', 'parse_bbox', 'parse_plan', 'sanitize_bbox', 'tag']
