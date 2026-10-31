from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence
import torch

@dataclass(frozen=True)
class ImageCropWindow:
    top: int
    left: int
    height: int
    width: int
    source_height: int
    source_width: int

    def __post_init__(self) -> None:
        values = (self.top, self.left, self.height, self.width, self.source_height, self.source_width)
        if any((isinstance(value, bool) or int(value) != value for value in values)):
            raise ValueError(f'Crop-window values must be integers, got {values!r}.')
        if self.source_height <= 0 or self.source_width <= 0:
            raise ValueError(f'Invalid crop source size: {(self.source_height, self.source_width)}.')
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f'Invalid crop size: {(self.height, self.width)}.')
        if self.top < 0 or self.left < 0:
            raise ValueError(f'Crop offsets must be non-negative, got {self!r}.')
        if self.top + self.height > self.source_height:
            raise ValueError(f'Crop exceeds source height: {self!r}.')
        if self.left + self.width > self.source_width:
            raise ValueError(f'Crop exceeds source width: {self!r}.')

def clip_bboxes(boxes: Sequence[Sequence[float]], *, image_height: int, image_width: int) -> tuple[tuple[float, float, float, float], ...]:
    image_height = int(image_height)
    image_width = int(image_width)
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f'Invalid image size: {(image_height, image_width)}.')
    clipped = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError(f'BBox must contain four xyxy coordinates, got {box!r}.')
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        if not all((math.isfinite(value) for value in (x_min, y_min, x_max, y_max))):
            raise ValueError(f'BBox contains a non-finite coordinate: {box!r}.')
        x_min = min(float(image_width), max(0.0, x_min))
        x_max = min(float(image_width), max(0.0, x_max))
        y_min = min(float(image_height), max(0.0, y_min))
        y_max = min(float(image_height), max(0.0, y_max))
        if x_min < x_max and y_min < y_max:
            clipped.append((x_min, y_min, x_max, y_max))
    return tuple(clipped)

def resize_bboxes(boxes: Sequence[Sequence[float]], *, source_height: int, source_width: int, output_height: int, output_width: int) -> tuple[tuple[float, float, float, float], ...]:
    source_height = int(source_height)
    source_width = int(source_width)
    output_height = int(output_height)
    output_width = int(output_width)
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f'Invalid bbox source size: {(source_height, source_width)}.')
    scale_x = output_width / float(source_width)
    scale_y = output_height / float(source_height)
    return clip_bboxes(tuple(((float(x_min) * scale_x, float(y_min) * scale_y, float(x_max) * scale_x, float(y_max) * scale_y) for x_min, y_min, x_max, y_max in boxes)), image_height=output_height, image_width=output_width)

def map_bboxes_through_crop(boxes: Sequence[Sequence[float]], window: ImageCropWindow, *, output_height: int, output_width: int) -> tuple[tuple[float, float, float, float], ...]:
    output_height = int(output_height)
    output_width = int(output_width)
    scale_x = output_width / float(window.width)
    scale_y = output_height / float(window.height)
    mapped = tuple((((float(x_min) - window.left) * scale_x, (float(y_min) - window.top) * scale_y, (float(x_max) - window.left) * scale_x, (float(y_max) - window.top) * scale_y) for x_min, y_min, x_max, y_max in boxes))
    return clip_bboxes(mapped, image_height=output_height, image_width=output_width)

def draw_bboxes(image: torch.Tensor, boxes: Sequence[Sequence[float]], *, line_width: int=8) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f'Expected image [3,H,W], got {tuple(image.shape)}')
    line_width = int(line_width)
    if line_width <= 0:
        raise ValueError(f'line_width must be positive, got {line_width}')
    if not boxes:
        return image
    output = image.clone()
    height, width = (int(output.shape[-2]), int(output.shape[-1]))
    red = output.new_tensor((1.0, 0.0, 0.0)).view(3, 1, 1)
    for box in boxes:
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        left = max(0, min(width - 1, int(round(x_min))))
        top = max(0, min(height - 1, int(round(y_min))))
        right = max(left, min(width - 1, int(round(x_max))))
        bottom = max(top, min(height - 1, int(round(y_max))))
        thickness = min(line_width, right - left + 1, bottom - top + 1)
        output[:, top:top + thickness, left:right + 1] = red
        output[:, bottom - thickness + 1:bottom + 1, left:right + 1] = red
        output[:, top:bottom + 1, left:left + thickness] = red
        output[:, top:bottom + 1, right - thickness + 1:right + 1] = red
    return output
