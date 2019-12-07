from typing import Tuple

import tensorflow as tf


def scale_boxes(boxes: tf.Tensor,
                from_size: Tuple[int, int],
                to_size: Tuple[int, int]) -> tf.Tensor:
    """
    Scale boxes to a new image size. 
    Converts boxes generated in an image with `from_size` dimensions
    to boxes that fit inside an image with `to_size` dimensions

    Parameters
    ----------
    boxes: tf.Tensor
        Boxes to be scale. Boxes must be declared in 
        [xmin, ymin, xmax, ymax] format
    from_size: Tuple[int, int], (height, width)
        Dimensions of the image where the boxes are decalred
    to_size: Tuple[int, int]

    Returns
    -------
    tf.Tensor
        Scaled boxes
    """
    ratio_w = from_size[1] / to_size[1]
    ratio_h = from_size[0] / to_size[0]

    x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
    x1 *= ratio_w
    x2 *= ratio_w
    y1 *= ratio_h
    y2 *= ratio_h
    
    return tf.concat([x1, y1, x2, y2], axis=1)


def normalize_bndboxes(boxes: tf.Tensor, 
                       image_size: Tuple[int, int]) -> tf.Tensor:
    """
    Normalizes boxes so they can be image size agnostic

    Parameters
    ----------
    boxes: tf.Tensor of shape [N_BOXES, 4]
        Boxes to be normalize. Boxes must be declared in 
        [xmin, ymin, xmax, ymax] format
    image_size: Tuple[int, int], (height, width)
        Dimensions of the image where the boxes are decalred

    Returns
    -------
    tf.Tensor of shape [N_BOXES, 4]
        Normalized boxes with values in range [0, 1] 
    """
    h, w = image_size
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
    x1 /= (w - 1)
    x2 /= (w - 1)
    y1 /= (h - 1)
    y2 /= (h - 1)
    return tf.concat([x1, y1, x2, y2], axis=1)
