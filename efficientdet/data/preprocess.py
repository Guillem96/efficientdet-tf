import random
from typing import Tuple

import tensorflow as tf

import efficientdet.utils.bndbox as bb_utils


_Annots = Tuple[tf.Tensor, tf.Tensor]


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize the image according imagenet mean and std

    Parameters
    ----------
    image: tf.Tensor of shape [H, W, C]
        Image in [0, 1] range
    
    Returns
    -------
    tf.Tensor
        Normalized image
    """
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return (image - mean) / std


def unnormalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Unnormalize the image according imagenet mean and std

    Parameters
    ----------
    image: tf.Tensor of shape [H, W, C]
        Image in [0, 1] range
    
    Returns
    -------
    tf.Tensor
        Unnormalized image
    """
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return image * std + mean


def horizontal_flip(image: tf.Tensor, 
                    annots: _Annots) -> Tuple[tf.Tensor, _Annots]:
    labels, boxes = annots

    # Flip the image
    image = tf.image.flip_left_right(image)
    
    # Flip the box
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)

    bb_w = x2 - x1
    delta_W = tf.expand_dims(boxes[:, 0], axis=-1)

    x1 = image.shape[1] - delta_W - bb_w
    x2 = image.shape[1] - delta_W

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])

    return image, (labels, boxes)


def crop(image: tf.Tensor, 
         annots: _Annots) -> Tuple[tf.Tensor, _Annots]:
    
    labels, boxes = annots

    # Get random crop dims
    crop_factor = tf.linspace(.4, .8, 4)
    idx = tf.random.uniform([1], maxval=4, dtype=tf.int32)
    crop_factor = tf.gather(crop_factor, idx)
    crop_width = tf.cast(image.shape[1] * crop_factor, tf.int32)
    crop_height = tf.cast(image.shape[0] * crop_factor, tf.int32)

    # Pick coordinates to start the crop
    x = tf.random.uniform(
        shape=[1], maxval=(image.shape[1] - crop_width)[0], dtype=tf.int32)
    y = tf.random.uniform(
        shape=[1], maxval=(image.shape[0] - crop_height)[0], dtype=tf.int32)

    # Crop the image and resize it back to original size
    crop_im = tf.image.crop_to_bounding_box(
        image, x[0], y[0], crop_height[0], crop_width[0])
    crop_im = tf.image.resize(crop_im, image.shape[:-1])
    
    # Clip the boxes to fit inside the crop
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    
    x = tf.cast(x, tf.float32)
    crop_width = tf.cast(crop_width, tf.float32)
    y = tf.cast(y, tf.float32)
    crop_height = tf.cast(crop_height, tf.float32)
    
    widths = x2 - x1
    heights = y2 - y1

    x1 = x1 - y
    y1 = y1 - x
    x2 = x1 + widths
    y2 = y1 + heights

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])
    boxes = bb_utils.clip_boxes(tf.expand_dims(boxes, 0), 
                                (crop_height, crop_width))
    boxes = tf.reshape(boxes, [-1, 4])

    # Create a mask to avoid tiny boxes
    areas = widths * heights
    # Min area is the 5 per cent of the whole area
    min_area = .05 * (crop_height * crop_height)
    large_areas = tf.reshape(tf.greater_equal(areas, min_area), [-1])

    # Get only large enough boxes
    boxes = tf.boolean_mask(boxes, large_areas, axis=0)

    labels = tf.boolean_mask(labels, large_areas)

    # Scale the boxes to original image
    boxes = bb_utils.scale_boxes(
        boxes, tuple(image.shape[:-1]), (crop_height, crop_width))
    return crop_im, (labels, boxes)


def no_transform(image, annots):
    return image, annots


def augment(image: tf.Tensor, 
            annots: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, 
                                                          tf.Tensor, 
                                                          tf.Tensor]:

    image, annots = tf.cond(tf.random.uniform([1]) < .5,
                            lambda: horizontal_flip(image, annots),
                            lambda: no_transform(image, annots))

    image, annots = tf.cond(tf.random.uniform([1]) < .5,
                            lambda: crop(image, annots),
                            lambda: no_transform(image, annots))
                            
    return image, annots