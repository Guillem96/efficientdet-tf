import abc
from typing import Tuple

import tensorflow as tf

import efficientdet.utils.bndbox as bb_utils
from efficientdet.typing import Annotation, ObjectDetectionInstance


@tf.function
def horizontal_flip(image: tf.Tensor, 
                    annots: Annotation) -> ObjectDetectionInstance:
    labels = annots[0]
    boxes = annots[1]

    im_shape = tf.shape(image)
    h, w = im_shape[0], im_shape[1]

    # Flip the image
    image = tf.image.flip_left_right(image)
    
    # Flip the box
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)

    bb_w = x2 - x1
    delta_W = tf.expand_dims(boxes[:, 0], axis=-1)

    x1 = w - delta_W - bb_w
    x2 = w - delta_W

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])

    return image, (labels, boxes)


@tf.function
def crop(image: tf.Tensor, 
         annots: Annotation) -> ObjectDetectionInstance:
    
    labels = annots[0]
    boxes = annots[1]

    im_shape = tf.shape(image)
    h, w = im_shape[0], im_shape[1]

    # Get random crop dims
    crop_factor_w = tf.random.uniform(shape=(1,), minval=.4, maxval=1.)
    crop_factor_h = tf.random.uniform(shape=(1,), minval=.4, maxval=1.)

    crop_width = tf.cast(w * crop_factor_w, tf.int32)
    crop_height = tf.cast(h * crop_factor_h, tf.int32)

    # Pick coordinates to start the crop
    x = tf.random.uniform(shape=[1], maxval=w - crop_width, dtype=tf.int32)
    y = tf.random.uniform(shape=[1], maxval=h - crop_height, dtype=tf.int32)

    # Crop the image and resize it back to original size
    crop_im = tf.image.crop_to_bounding_box(
        image, x, y, crop_height, crop_width)
    crop_im = tf.image.resize(crop_im, (h, w))
    
    # Clip the boxes to fit inside the crop
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    
    # Cast crop coordinates to float, so they can be used for clipping
    x = tf.cast(x, tf.float32)
    crop_width = tf.cast(crop_width, tf.float32)
    y = tf.cast(y, tf.float32)
    crop_height = tf.cast(crop_height, tf.float32)
    
    # Adjust boxes coordinates after the crop
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
    widths = tf.gather(boxes, 2, axis=-1) - tf.gather(boxes, 0, axis=-1)
    heights = tf.gather(boxes, 3, axis=-1) - tf.gather(boxes, 1, axis=-1)
    areas = widths * heights
    
    # Min area is the 1 per cent of the whole area
    min_area = .01 * (crop_height * crop_width)
    large_areas = tf.reshape(tf.greater_equal(areas, min_area), [-1])

    # Get only large enough boxes
    boxes = tf.boolean_mask(boxes, large_areas, axis=0)
    labels = tf.boolean_mask(labels, large_areas)

    # Scale the boxes to original image
    boxes = bb_utils.scale_boxes(
        boxes, image.shape[:-1], (crop_height, crop_width))

    return crop_im, (labels, boxes)


@tf.function
def erase(image: tf.Tensor, 
          annots: Annotation,
          patch_aspect_ratio: Tuple[float, float] = (.15, .15)) \
              -> ObjectDetectionInstance:

    im_shape = tf.shape(image)
    h, w = im_shape[0], im_shape[1]
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    
    # Generate patch
    h_prop = tf.random.uniform(shape=[1], 
                               minvalue=0, 
                               maxvalue=patch_aspect_ratio[0])
    w_prop = tf.random.uniform(shape=[1], 
                               minvalue=0, 
                               maxvalue=patch_aspect_ratio[1])
    patch_h = tf.cast(tf.multiply(h, h_prop), tf.int32)
    patch_w = tf.cast(tf.multiply(w, w_prop), tf.int32)
    patch = tf.zeros([patch_h, patch_w], tf.float32)

    # Generate random location for patches
    x = tf.random.uniform(shape=[1], maxval=w - patch_w, dtype=tf.int32)
    y = tf.random.uniform(shape=[1], maxval=h - patch_h, dtype=tf.int32)
    
    # Pad patch with ones so it has the same shape as the image
    paddings = tf.constant([
        [x, y], 
        [tf.cast(w, tf.int32) - x - patch_w, 
         tf.cast(h, tf.int32) - y - patch_h]], tf.int32)
    patch = tf.pad(patch, paddings, constant_values=1.)

    return tf.multiply(image, patch), annots


@tf.function
def no_transform(image: tf.Tensor, 
                 annots: Annotation) -> ObjectDetectionInstance:
    return image, annots


class Augmentation(abc.ABC):

    def __init__(self, prob: float = .5) -> None:
        self.prob = prob

    @abc.abstractmethod
    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        raise NotImplementedError

    def __call__(self,
                 image: tf.Tensor, 
                 annot: Annotation) -> ObjectDetectionInstance:

        image, annot = tf.cond(tf.random.uniform([1]) < self.prob,
                               lambda: horizontal_flip(image, annot),
                               lambda: no_transform(image, annot))

        return image, annot


class RandomHorizontalFlip(Augmentation):

    def __init__(self, prob: float = .5) -> None:
        super(RandomHorizontalFlip, self).__init__(prob=prob)


    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        return horizontal_flip(image, annot)


class RandomCrop(Augmentation):

    def __init__(self, prob: float = .5) -> None:
        super(RandomCrop, self).__init__(prob=prob)

    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        return crop(image, annot)
