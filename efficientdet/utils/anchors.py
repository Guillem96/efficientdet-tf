"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
from typing import List, Union, Tuple, Sequence

import tensorflow as tf
import numpy as np # TODO: Migrate to tf

from . import compute_overlap as iou


class AnchorGenerator(object):
    
    def __init__(self, 
                 size: float,
                 aspect_ratios: Sequence[float],
                 stride: int = 1):
        """
        RetinaNet input examples:
            size: 32
            aspect_ratios: [0.5, 1, 2]
        """
        self.size = size
        self.stride = stride

        self.aspect_ratios = aspect_ratios
        self.anchor_scales = [
            2 ** 0,
            2 ** (1 / 3.0),
            2 ** (2 / 3.0)]

        self.anchors = self._generate()
    
    def __call__(self, *args, **kwargs):
        return self.tile_anchors_over_feature_map(*args, **kwargs)

    def tile_anchors_over_feature_map(self, feature_map_shape):
        """
        Tile anchors over all feature map positions

        Parameters
        ----------
        feature_map: Tuple[int, int, int] H, W , C
            Feature map where anchors are going to be tiled
        
        Returns
        --------
        tf.Tensor of shape [BATCH, N_BOXES, 4]
        """
        def arange(limit):
            return tf.range(0, limit, dtype=tf.float32)
        
        h = feature_map_shape[0]
        w = feature_map_shape[1]

        shift_x = (arange(w) + 0.5) * self.stride
        shift_y = (arange(h) + 0.5) * self.stride

        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
        shifts = tf.transpose(shifts)

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = len(self)
        K = shifts.shape[0]
    
        all_anchors = (tf.reshape(self.anchors, [1, A, 4]) 
                       + tf.cast(tf.reshape(shifts, [K, 1, 4]), tf.float64))
        all_anchors = tf.reshape(all_anchors, [K * A, 4])

        return all_anchors

    def _generate(self) -> tf.Tensor:
        num_anchors = len(self)
        ratios = np.array(self.aspect_ratios)
        scales = np.array(self.anchor_scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = self.size * np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def __len__(self):
        return len(self.aspect_ratios) * len(self.anchor_scales)


def anchor_targets_bbox(anchors: np.ndarray,
                        images: np.ndarray,
                        bndboxes: np.ndarray,
                        labels: np.ndarray,
                        num_classes: int,
                        padding_value: int = -1,
                        negative_overlap: float = 0.4,
                        positive_overlap: float = 0.5) -> Tuple[np.ndarray,
                                                                np.ndarray]:
    """ 
    Generate anchor targets for bbox detection.

    Parameters
    ----------
    anchors: np.ndarray 
        Annotations of shape (N, 4) for (x1, y1, x2, y2).
    images: np.ndarray
        Array of shape [BATCH, H, W, C] containing images.
    bndboxes: np.ndarray
        Array of shape [BATCH, N, 4] contaning ground truth boxes
    labels: np.ndarray
        Array of shape [BATCH, N] containing the labels for each box
    num_classes: int
        Number of classes to predict.
    negative_overlap: float, default 0.4
        IoU overlap for negative anchors 
        (all anchors with overlap < negative_overlap are negative).
    positive_overlap: float, default 0.5
        IoU overlap or positive anchors 
        (all anchors with overlap > positive_overlap are positive).
    padding_value: int
        Value used to pad labels
    Returns
    --------
    Tuple[np.ndarray, np.ndarray]
        labels_batch: 
            batch that contains labels & anchor states 
            (np.array of shape (batch_size, N, num_classes + 1),
            where N is the number of anchors for an image and the last 
            column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: 
            batch that contains bounding-box regression targets for an 
            image & anchor states (np.array of shape (batch_size, N, 4 + 1),
            where N is the number of anchors for an image, the first 4 columns 
            define regression targets for (x1, y1, x2, y2) and the
            last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """
    batch_size = images.shape[0]

    regression_batch = np.zeros((batch_size, anchors.shape[0], 4 + 1))
    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1))

    # compute labels and regression targets
    for batch_idx, (image, bboxes, label) in enumerate(zip(images, bndboxes, labels)):
        if bboxes.shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            result = compute_gt_annotations(anchors,
                                            bboxes,
                                            negative_overlap, 
                                            positive_overlap)
            positive_indices, ignore_indices, argmax_overlaps_inds = result
            
            # If any anchor overlaps with padding box
            padding_indices = label[argmax_overlaps_inds] == padding_value
            
            positive_indices = positive_indices & ~padding_indices
            ignore_indices = ignore_indices | padding_indices

            labels_batch[batch_idx, ignore_indices, -1] = -1 # Ignore
            labels_batch[batch_idx, positive_indices, -1] = 1 # Foreground

            regression_batch[batch_idx, ignore_indices, -1] = -1 # Ignore
            regression_batch[batch_idx, positive_indices, -1] = 1 # Foreground

            # compute target class labels
            labels_batch[batch_idx, 
                         positive_indices,
                         label[argmax_overlaps_inds[positive_indices]]] = 1

            regression_batch[batch_idx, :, :-1] = \
                bbox_transform(anchors, 
                               bboxes[argmax_overlaps_inds])

        # ignore annotations outside of image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, 
                                         (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], 
                                    anchors_centers[:, 1] >= image.shape[0])

            labels_batch[batch_idx, indices, -1] = -1
            regression_batch[batch_idx, indices, -1] = -1

    return regression_batch, labels_batch


def compute_gt_annotations(anchors: np.ndarray,
                           annotations: np.ndarray,
                           negative_overlap=0.4,
                           positive_overlap=0.5) -> Tuple[np.ndarray,
                                                          np.ndarray,
                                                          np.ndarray]:
    """ 
    Obtain indices of gt annotations with the greatest overlap.
    
    Parameters
    ----------
        anchors: np.array
            Annotations of shape [N, 4] for (x1, y1, x2, y2).
        annotations: np.array 
            shape [N, 5] for (x1, y1, x2, y2).
        negative_overlap: float, default 0.4
            IoU overlap for negative anchors 
            (all anchors with overlap < negative_overlap are negative).
        positive_overlap: float, default 0.5
            IoU overlap or positive anchors 
            (all anchors with overlap > positive_overlap are positive).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """

    overlaps = iou.compute_overlap(anchors.astype(np.float64), 
                                   annotations.astype(np.float64))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
    max_overlap_boxes = annotations[argmax_overlaps_inds]

    # Compute areas of boxes so we can ignore 'ridiculous' sized boxes
    widths = (max_overlap_boxes[:, 2] - max_overlap_boxes[:, 0])
    heights = (max_overlap_boxes[:, 3] - max_overlap_boxes[:, 1])
    areas = widths * heights
    small_areas = areas < 1.

    # assign "dont care" labels
    positive_indices = (max_overlaps >= positive_overlap) & ~small_areas
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices
    ignore_indices = ignore_indices | small_areas

    return positive_indices, ignore_indices, argmax_overlaps_inds


_ListOfInt = List[int]
_TupleOfInt = Tuple[int, int, int, int]
_MeanStd = Union[np.ndarray, _ListOfInt, _TupleOfInt]

def bbox_transform(anchors: np.ndarray, 
                   gt_boxes: np.ndarray, 
                   mean: _MeanStd = None, 
                   std: _MeanStd = None) -> np.ndarray:
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    targets = (targets - mean) / std

    return targets
