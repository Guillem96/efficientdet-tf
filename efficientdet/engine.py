import copy
from typing import Callable, Tuple, Mapping, Sequence

import tensorflow as tf

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import efficientdet.utils as utils

LossFn = Callable[[tf.Tensor] * 4, Tuple[tf.Tensor, tf.Tensor]]
import time


def get_lr(optimizer):
    lr = optimizer.learning_rate
    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        return lr.current_lr
    return lr


def _train_step(model: tf.keras.Model,
                optimizer: tf.optimizers.Optimizer,
                loss_fn: LossFn,
                images: tf.Tensor, 
                regress_targets: tf.Tensor, 
                labels: tf.Tensor) -> Tuple[float, float]:
    
    with tf.GradientTape() as tape:
        regressors, clf_probas = model(images)

        reg_loss, clf_loss = loss_fn(labels, clf_probas, 
                                    regress_targets, regressors)
        l2_loss = 4e-5 * sum([tf.reduce_sum(tf.pow(w, 2)) 
                              for w in model.trainable_variables])
        loss = reg_loss + clf_loss + l2_loss

    grads = tape.gradient(loss, model.trainable_variables)
    return reg_loss, clf_loss, grads


def train_single_epoch(model: tf.keras.Model,
                       anchors: tf.Tensor,
                       dataset: tf.data.Dataset,
                       optimizer: tf.optimizers.Optimizer,
                       grad_accum_steps: int,
                       loss_fn: LossFn,
                       steps: int,
                       epoch: int,
                       num_classes: int,
                       print_every: int = 10):
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None, None, 3], 
                                       dtype=tf.float32),
                         tf.TensorSpec(shape=[None, None, 5], 
                                       dtype=tf.float32),
                         tf.TensorSpec(shape=[None, None, num_classes + 1], 
                                       dtype=tf.float32)])
    def train_step(images, r_targets, c_targets):
        return _train_step(
            model=model, optimizer=optimizer, loss_fn=loss_fn,
            images=images, regress_targets=r_targets, labels=c_targets)

    acc_gradients = []

    running_loss = tf.metrics.Mean()
    running_clf_loss = tf.metrics.Mean()
    running_reg_loss = tf.metrics.Mean()

    for i, (images, (labels, bbs)) in enumerate(dataset):

        target_reg, target_clf = utils.anchors.anchor_targets_bbox(
            anchors, images, bbs, labels, num_classes)

        reg_loss, clf_loss, grads = train_step(
            images=images, r_targets=target_reg, c_targets=target_clf)
        
        if tf.math.is_nan(reg_loss) or tf.math.is_nan(clf_loss):
            print('Loss NaN, skipping training step')
            continue
        
        if len(acc_gradients) == 0:
            acc_gradients = grads 
        else:
            acc_gradients = [g1 + g2 for g1, g2 in zip(acc_gradients, grads)]

        if (i + 1) % grad_accum_steps == 0:
            optimizer.apply_gradients(
                zip(acc_gradients, model.trainable_variables))
            acc_gradients = []
        
        running_loss(reg_loss + clf_loss)
        running_clf_loss(clf_loss)
        running_reg_loss(reg_loss)

        if (i + 1) % print_every == 0:
            lr = get_lr(optimizer)
            print(f'Epoch[{epoch}] [{i}/{steps}] '
                  f'loss: {running_loss.result():.6f} '
                  f'clf. loss: {running_clf_loss.result():.6f} '
                  f'reg. loss: {running_reg_loss.result():.6f} '
                  f'learning rate: {lr:.6f}')

    if len(acc_gradients) > 0:
        optimizer.apply_gradients(
                zip(acc_gradients, model.trainable_variables))


def _COCO_result(image_id: int,
                 labels: tf.Tensor,
                 bboxes: tf.Tensor,
                 scores: tf.Tensor):

    b_h = bboxes[:, 3] - bboxes[:, 1]
    b_w = bboxes[:, 2] - bboxes[:, 0]
    coco_bboxes = tf.stack([bboxes[:, 0], bboxes[:, 1], b_w, b_h])
    coco_bboxes = tf.transpose(coco_bboxes).numpy().tolist()

    labels = labels.numpy().tolist()
    scores = scores.numpy().tolist()

    return [dict(image_id=image_id, category_id=l, bbox=b,score=s) 
            for l, b, s in zip(labels, coco_bboxes, scores)]


def _COCO_gt_annot(image_id: int,
                   annot_id: int,
                   image_shape: Tuple[int, int], 
                   labels: tf.Tensor, 
                   bboxes: tf.Tensor):
    
    im_h, im_w = image_shape
    
    b_h = bboxes[:, 3] - bboxes[:, 1]
    b_w = bboxes[:, 2] - bboxes[:, 0]
    areas = tf.reshape(b_h * b_w, [-1])
    areas = areas.numpy().tolist()

    coco_bboxes = tf.stack([bboxes[:, 0], bboxes[:, 1], b_w, b_h])
    coco_bboxes = tf.transpose(coco_bboxes).numpy().tolist()

    labels = labels.numpy().tolist()

    image = dict(id=image_id, height=im_h, width=im_w)

    it = zip(coco_bboxes, areas, labels)
    annotations = [
        dict(id=id_, image_id=image_id, bbox=bbox, iscrowd=0, 
             area=a, category_id=l)
        for id_, (bbox, a, l) in enumerate(it, start=annot_id)]

    return image, annotations
    

def evaluate(model: tf.keras.Model, 
             dataset: tf.data.Dataset,
             class2idx: Mapping[str, int],
             steps: int,
             print_every: int = 10):

    gt_coco = dict(images=[], annotations=[])
    results_coco = []
    image_id = 1
    annot_id = 1

    # Create COCO categories
    categories = [dict(supercategory='instance', id=i, name=n)
                  for n, i in class2idx.items()]
    gt_coco['categories'] = categories
    
    for i, (images, (labels, bbs)) in enumerate(dataset):
        
        bboxes, categories, scores = model(images, training=False)
        h, w = images.shape[1: 3]
        
        # Iterate through images in batch, and for each one
        # create the ground truth coco annotation

        for batch_idx in range(len(bboxes)):
            gt_labels, gt_boxes = labels[batch_idx], bbs[batch_idx]
            no_padding_mask = gt_labels != -1
            
            gt_labels = tf.boolean_mask(gt_labels, no_padding_mask)
            gt_boxes = tf.boolean_mask(gt_boxes, no_padding_mask)

            im_annot, annots = _COCO_gt_annot(image_id, annot_id, 
                                              (h, w), gt_labels, gt_boxes)
            gt_coco['annotations'].extend(annots)
            gt_coco['images'].append(im_annot)
            
            preds = categories[batch_idx], bboxes[batch_idx], scores[batch_idx]
            pred_labels, pred_boxes, pred_scores = preds

            if pred_labels.shape[0] > 0:
                results = _COCO_result(image_id, 
                                       pred_labels, pred_boxes, pred_scores)
                results_coco.extend(results)
            
            annot_id += len(annots)
            image_id += 1

        if i % print_every == 0:
            print(f'Validating[{i}/{steps}]...')

    # Convert custom annotations to COCO annots
    gtCOCO = COCO()
    gtCOCO.dataset = gt_coco
    gtCOCO.createIndex()

    resCOCO = COCO()
    resCOCO.dataset['images'] = [img for img in gt_coco['images']]
    resCOCO.dataset['categories'] = copy.deepcopy(gt_coco['categories'])

    for i, ann in enumerate(results_coco):
        bb = ann['bbox']
        x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
        if not 'segmentation' in ann:
            ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        ann['area'] = bb[2]*bb[3]
        ann['id'] = i + 1
        ann['iscrowd'] = 0

    resCOCO.dataset['annotations'] = results_coco
    resCOCO.createIndex()

    coco_eval = COCOeval(gtCOCO, resCOCO, 'bbox')
    coco_eval.params.imgIds = sorted(gtCOCO.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
