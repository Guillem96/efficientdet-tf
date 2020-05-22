from functools import partial
from typing import Tuple, List, Union

import tensorflow as tf


def to_tf_format(boxes: tf.Tensor):
    """
    Convert xmin, ymin, xmax, ymax boxes to ymin, xmin, ymax, xmax
    and viceversa
    """
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)
    return tf.concat([x1, y1, x2, y2], axis=-1)


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
    h = image_size[0]
    w = image_size[1]

    x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
    x1 /= (w - 1)
    x2 /= (w - 1)
    y1 /= (h - 1)
    y2 /= (h - 1)
    return tf.concat([x1, y1, x2, y2], axis=1)


def regress_bndboxes(boxes: tf.Tensor,
                     regressors: tf.Tensor) -> tf.Tensor:
    """
    Apply scale invariant regression to boxes.

    Parameters
    ----------
    boxes: tf.Tensor of shape [BATCH, N_BOXES, 4]
        Boxes to apply the regressors
    regressors: tf.Tensor of shape [BATCH, N_BOXES, 4]
        Scale invariant regressions
    
    Returns
    -------
    tf.Tensor
        Regressed boxes
    """ 
    boxes = tf.cast(boxes, tf.float32)
    regressors = tf.cast(regressors, tf.float32)
    
    mean = tf.constant([0., 0., 0., 0.], dtype=tf.float32)
    std = tf.constant([0.2, 0.2, 0.2, 0.2], dtype=tf.float32)

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (regressors[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (regressors[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (regressors[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (regressors[:, :, 3] * std[3] + mean[3]) * height

    return tf.stack([x1, y1, x2, y2], axis=2)


def clip_boxes(boxes: tf.Tensor, 
               im_size: Tuple[int, int]) -> tf.Tensor:
    # TODO: Document this
    h = im_size[0]
    w = im_size[1]

    h = tf.cast(h - 1, tf.float32)
    w = tf.cast(w - 1, tf.float32)

    x1 = tf.clip_by_value(boxes[..., 0], 0., w)
    y1 = tf.clip_by_value(boxes[..., 1], 0., h)
    x2 = tf.clip_by_value(boxes[..., 2], 0., w)
    y2 = tf.clip_by_value(boxes[..., 3], 0., h)

    return tf.stack([x1, y1, x2, y2], axis=-1)


def single_image_nms(boxes: tf.Tensor, 
                     scores: tf.Tensor,
                     score_threshold: float = 0.05):

    """
    Perform detection filters using Non maxima supression on the predictions of
    a single image

    Parameters
    ----------
    boxes: tf.Tensor of shape [N_BOXES, 4]
    scores: tf.Tensor of shape [N_BOXES, N_CLASSES]

    Returns
    -------
    tf.Tensor of shape [N, 2]
        N indices pairs of box_idx, label. 
        Get box indices with indices[:,0] and labels by [:,1]
    """
    def per_class_nms(class_idx):
        class_scores = tf.gather(scores, class_idx, axis=-1)
        
        indices = tf.image.non_max_suppression(
            boxes=boxes,
            scores=class_scores,
            max_output_size=100,
            score_threshold=score_threshold)
        
        n_indices = tf.constant(tf.shape(indices)[0])
        labels = tf.tile([class_idx], [n_indices])
        return tf.stack([indices, labels], axis=-1)

    boxes = to_tf_format(boxes)
    return tf.concat(
        [per_class_nms(c) for c in range(tf.shape(scores)[-1])], axis=0)


def nms(boxes: tf.Tensor, 
        class_scores: tf.Tensor,
        score_threshold: float = 0.5) -> tf.Tensor:

    """
    Parameters
    ----------
    boxes: tf.Tensor of shape [BATCH, N, 4]

    class_scores: tf.Tensor of shape [BATCH, N, NUM_CLASSES]
    
    score_threshold: float, default 0.1
        Classification score to keep the box

    Returns
    -------
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]
        The list len is equal to batch size. 
        list[0] contains the boxes and corresponding label of the first element
        of the batch
        boxes List[tf.Tensor of shape [N, 4]]
        labels: List[tf.Tensor of shape [N]]
        scores: List[tf.Tensor of shape [N]]
    """
    iou_threshold = .5
    
    batch_size = tf.shape(boxes)[0]
    num_classes = tf.shape(class_scores)[-1]
    
    boxes = tf.cast(boxes, tf.float32)
    class_scores = tf.cast(class_scores, tf.float32)
    
    all_boxes = []
    all_labels = []
    all_scores = []

    for batch_idx in range(batch_size):
        c_scores = tf.gather(class_scores, batch_idx)
        c_boxes = tf.gather(boxes, batch_idx)

        indices = single_image_nms(c_boxes, c_scores, score_threshold)
        
        batch_labels = indices[:, 1]
        batch_scores = tf.gather_nd(c_scores, indices)
        batch_boxes = tf.gather(c_boxes, indices[:, 0])
        
        all_boxes.append(batch_boxes)
        all_scores.append(batch_scores)
        all_labels.append(batch_labels)

    return all_boxes, all_labels, all_scores
    

def bbox_overlap(boxes, gt_boxes):
    """
    Calculates the overlap between proposal and ground truth boxes.
    Some `gt_boxes` may have been padded. The returned `iou` tensor for these
    boxes will be -1.
    
    Parameters
    ----------
    boxes: tf.Tensor with a shape of [batch_size, N, 4]. 
        N is the number of proposals before groundtruth assignment. The
        last dimension is the pixel coordinates in [xmin, ymin, xmax, ymax] form.
    gt_boxes: tf.Tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. 
        This tensor might have paddings with a negative value.
    
    Returns
    -------
    tf.FloatTensor 
        A tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
    """
    bb_x_min, bb_y_min, bb_x_max, bb_y_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    i_xmin = tf.math.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.math.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.math.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.math.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = (tf.math.maximum(i_xmax - i_xmin, 0) * 
              tf.math.maximum(i_ymax - i_ymin, 0))

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    # Fills -1 for IoU entries between the padded ground truth boxes.
    gt_invalid_mask = tf.less(
        tf.reduce_max(gt_boxes, axis=-1, keepdims=True), 0.0)
    padding_mask = tf.logical_or(
        tf.zeros_like(bb_x_min, dtype=tf.bool),
        tf.transpose(gt_invalid_mask, [0, 2, 1]))
    iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

    return iou
