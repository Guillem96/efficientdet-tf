from typing import Tuple

import tensorflow as tf


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
