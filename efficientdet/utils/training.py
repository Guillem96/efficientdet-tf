import functools
from typing import Tuple

import tensorflow as tf

from .. import config
from . import anchors as anchors_utils
from ..typing import ObjectDetectionInstance


def _compute_gt(images: tf.Tensor, 
                annots: Tuple[tf.Tensor, tf.Tensor], 
                anchors: tf.Tensor,
                num_classes: int) -> ObjectDetectionInstance:

    labels = annots[0]
    boxes = annots[1]

    target_reg, target_clf = anchors_utils.anchor_targets_bbox(
            anchors, images, boxes, labels, num_classes)

    return images, (target_reg, target_clf)


def _generate_anchors(anchors_config: config.AnchorsConfig,
                      im_shape: int) -> tf.Tensor:

    anchors_gen = [anchors_utils.AnchorGenerator(
            size=anchors_config.sizes[i - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.strides[i - 3]) 
            for i in range(3, 8)]

    shapes = [im_shape // (2 ** x) for x in range(3, 8)]

    anchors = [g((size, size, 3))
               for g, size in zip(anchors_gen, shapes)]

    return tf.concat(anchors, axis=0)


def wrap_detection_dataset(ds: tf.data.Dataset,
                           im_size: Tuple[int, int],
                           num_classes: int) -> tf.data.Dataset:

    anchors = _generate_anchors(config.AnchorsConfig(), im_size[0])

    # Wrap datasets so they return the anchors labels
    dataset_training_head_fn = functools.partial(
        _compute_gt, anchors=anchors, num_classes=num_classes)

    return ds.map(dataset_training_head_fn)
