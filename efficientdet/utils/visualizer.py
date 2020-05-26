from itertools import cycle
from typing import Union, Tuple, Sequence, Mapping

import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw

Color = Tuple[int, int, int]
ImageType = Union[tf.Tensor, np.ndarray, 'Image']
Box = Union[Tuple[int, int, int, int], tf.Tensor]
Boxes = Union[Sequence[Box], tf.Tensor, np.ndarray]
FloatSequence = Union[tf.Tensor, np.ndarray, Sequence[float]]


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


def _parse_boxes(boxes: Boxes) -> Boxes:
    if isinstance(boxes, tf.Tensor):
        boxes = boxes.numpy().astype('int32').tolist()
    elif isinstance(boxes, np.ndarray):
        boxes = boxes.astype('int32').tolist()
    
    return [_parse_box(b) for b in boxes]


def colors_per_labels(labels: Sequence[str]) -> Sequence[Color]:
    """
    Creates an list of colors associated to a label.

    I recommend to call this function to associate a color to labels and then
    use the draw_boxes function passing the output of this function as the 
    list of colors

    Examples
    --------
    >>> labels = ['cat', 'dog', 'dog', 'cat']
    >>> colors_per_labels(labels)
    ... ['green', 'red, 'red, 'green']

    >>> labels = ['cat', 'dog', 'dog', 'cat']
    >>> colors = colors_per_labels(labels)
    >>> draw_boxes(image, labels, colors=colors)
    """
    
    import matplotlib.colors as mcolors
    
    colors = [mcolors.to_rgb(c) for c in mcolors.TABLEAU_COLORS]
    colors = (tuple([int(255 * c) for c in color]) for color in colors)
    
    unique_labels = set(labels)
    color_x_label: Mapping[str, Color] = dict(
        zip(labels, cycle(colors))) # type: ignore[arg-type]
        
    return [color_x_label[o] for o in labels]


def draw_boxes(image: ImageType, 
               boxes: Boxes,
               labels: Sequence[str] = None,
               scores: FloatSequence = None,
               colors: Sequence[Color] = ((0, 255, 0),)) -> 'Image':
    """
    Draw a set of boxes formatted as [x1, y1, x2, y2] to the image `image`
    
    Parameters
    ----------
    image: ImageType
        Image where the boxes are going to be drawn
    boxes: Boxes
        Set of boxes to draw. Boxes must have the format [x1, y1, x2, y2]
    labels: Sequence[str], default None
        Classnames corresponding to boxes
    scores: FloatSequence, defalt None

    colors: Sequence[Color], default [(0, 255, 0)]
        Colors to cycle through

    Returns
    -------
    PIL.Image
    """
    image = _image_to_pil(image)
    boxes = _parse_boxes(boxes)

    # Fill scores and labels with None if needed
    if labels is None:
        labels = [''] * len(boxes)
    
    if scores is None:
        scores = [''] * len(boxes)
    elif isinstance(scores, np.ndarray):
        scores = scores.reshape(-1).tolist()
    elif isinstance(scores, tf.Tensor):
        scores = scores.numpy().reshape(-1).tolist()

    # Check if scores and labels are correct
    assert len(labels) == len(boxes), \
        'Labels and boxes must have the same length'
    assert len(scores) == len(boxes), \
        'Scores and boxes must have the same length'

    n_colors = len(colors)
    draw = ImageDraw.Draw(image)
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        c = colors[i % n_colors]
        if label is not '' or score is not '':
            text = label + (f' {score:.2f}' if score else '')
            draw.rectangle([x1, y1 - 10, x2, y1], fill=c)
            draw.text([x1 + 5, y1 - 10], text)

        draw.rectangle(box, outline=c, width=2)
    
    return image
