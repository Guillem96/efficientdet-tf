from typing import Tuple, List, Union

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


_ListOfInt = List[int]
_TupleOfInt = Tuple[int, int, int, int]
_MeanStd = Union[_ListOfInt, _TupleOfInt]


def regress_bndboxes(boxes: tf.Tensor,
                     regressors: tf.Tensor,
                     mean: _MeanStd = None, 
                     std: _MeanStd = None) -> tf.Tensor:
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
    
    if mean is None:
        mean = tf.constant([0., 0., 0., 0.], dtype=tf.float32)
    if std is None:
        std = tf.constant([0.2, 0.2, 0.2, 0.2], dtype=tf.float32)

    if isinstance(mean, (list, tuple)):
        mean = tf.constant(mean, dtype=tf.float32)
    elif not isinstance(mean, tf.Tensor):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = tf.constant(std, dtype=tf.float32)
    elif not isinstance(std, tf.Tensor):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

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
    h, w = im_size

    x1 = tf.clip_by_value(boxes[:, :, 0], 0, w - 1)
    y1 = tf.clip_by_value(boxes[:, :, 1], 0, h - 1)
    x2 = tf.clip_by_value(boxes[:, :, 2], 0, w - 1)
    y2 = tf.clip_by_value(boxes[:, :, 3], 0, h - 1)

    return tf.stack([x1, y1, x2, y2], axis=2)


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
    
    boxes = tf.cast(boxes, tf.float32)
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    boxes = tf.stack([y1, x1, y2, x2], axis=-1)
    boxes = tf.reshape(boxes, [boxes.shape[0], -1, 4])

    class_scores = tf.cast(class_scores, tf.float32)
    
    all_boxes = []
    all_labels = []
    all_scores = []

    for batch_idx in range(boxes.shape[0]):
        batch_boxes = []
        batch_labels = []
        batch_scores = []

        for c in range(class_scores.shape[-1]):
            box_scores = class_scores[batch_idx, :, c]
            indices = tf.image.non_max_suppression(
                boxes[batch_idx],
                box_scores,
                max_output_size=100,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold)
                
            if indices.shape[0] > 0:
                batch_boxes.append(tf.gather(boxes[batch_idx], indices))
                batch_scores.append(tf.gather(box_scores, indices))
                batch_labels.extend([c] * len(indices))
        
        if batch_boxes:
            batch_boxes = tf.concat(batch_boxes, axis=0)
            y1, x1, y2, x2 = tf.split(batch_boxes, 4, axis=-1)
            batch_boxes = tf.stack([x1, y1, x2, y2], axis=-1)
            batch_boxes = tf.reshape(batch_boxes, [-1, 4])
            
            batch_scores = tf.concat(batch_scores, axis=0)
            
            all_boxes.append(batch_boxes)
            all_scores.append(batch_scores)
            all_labels.append(tf.constant(batch_labels, dtype=tf.int32))
        else:
            all_boxes.append(tf.constant([]))
            all_labels.append(tf.constant([]))
            all_scores.append(tf.constant([]))

    return all_boxes, all_labels, all_scores
