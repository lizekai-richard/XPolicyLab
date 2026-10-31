import torch

from ..Liber_0_lite.liber0.preprocessing import pack_images
from .geometry import ImageCropWindow, resize_bboxes, map_bboxes_through_crop, draw_bboxes


def prepare_images(images, resize, crop_scale, bbox, device):
    image = pack_images(images, resize, crop_scale).to(device, torch.bfloat16)
    if bbox is None:
        return image
    source_h, source_w = images[3].shape[:2]
    height, width = resize
    boxes = ((bbox[0] * source_w, bbox[1] * source_h,
              bbox[2] * source_w, bbox[3] * source_h),)
    boxes = resize_bboxes(boxes, source_height=source_h, source_width=source_w,
                          output_height=height, output_width=width)
    if crop_scale != 1.0:
        crop_h = max(1, min(height, int(round(height * crop_scale))))
        crop_w = max(1, min(width, int(round(width * crop_scale))))
        window = ImageCropWindow((height - crop_h) // 2, (width - crop_w) // 2,
                                 crop_h, crop_w, height, width)
        boxes = map_bboxes_through_crop(boxes, window, output_height=height, output_width=width)
    boxes = resize_bboxes(boxes, source_height=height, source_width=width,
                          output_height=144, output_width=192)
    if boxes:
        tile = (image[0, :, :144, 192:].float() + 1) / 2
        tile = draw_bboxes(tile, boxes, line_width=max(1, round(8 * min(144/source_h, 192/source_w))))
        image[0, :, :144, 192:] = (tile * 2 - 1).to(image.dtype)
    return image
