import copy
from typing import Callable, Tuple, Mapping

import tensorflow as tf

import efficientdet.utils as utils

LossFn = Callable[[tf.Tensor] * 4, Tuple[tf.Tensor, tf.Tensor]]


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

