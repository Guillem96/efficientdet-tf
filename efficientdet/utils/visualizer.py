from typing import Union, Tuple, Sequence

import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw

ImageType = Union[tf.Tensor, np.ndarray, 'Image']
Box = Union[Tuple[int, int, int, int], tf.Tensor]
Boxes = Union[Sequence[Box], tf.Tensor, np.ndarray]
Color = Tuple[int, int, int]


def _image_to_pil(image: ImageType) -> 'Image':

    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, tf.Tensor):
        image = image.numpy()
    
    if image.dtype == 'float32' or image.dtype == 'float64':
        image = (image * 255.).astype('uint8')
    elif image.dtype != 'uint8':
        print(image.dtype)
        raise ValueError('Image dtype not supported')

    return Image.fromarray(image)


def _parse_box(box: Box) -> Box:
    if isinstance(box, tf.Tensor):
        return tuple(box.numpy().astype('int32').tolist())
    elif isinstance(box, np.ndarray):
        return tuple(box.astype('int32').tolist())
    else:
        return tuple(map(int, box))


def _parse_boxes(boxes: Boxes):
    if isinstance(boxes, tf.Tensor):
        boxes = boxes.numpy().astype('int32').tolist()
    elif isinstance(boxes, np.ndarray):
        boxes = boxes.astype('int32').tolist()
    
    return [_parse_box(b) for b in boxes]


def draw_boxes(image: ImageType, 
               boxes: Boxes,
               colors: Sequence[Color] = ((0, 255, 0),)) -> 'Image':
    image = _image_to_pil(image)
    boxes = _parse_boxes(boxes)
    labels = [None] * len(boxes)

    n_colors = len(colors)
    draw = ImageDraw.Draw(image)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        draw.rectangle(box, outline=colors[i % n_colors], width=2)
    
    return image

