from typing import Callable, Tuple

import tensorflow as tf

import efficientdet.utils as utils


LossFn = Callable[[tf.Tensor] * 4, Tuple[tf.Tensor, tf.Tensor]]


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
        loss = (reg_loss + clf_loss) * 0.5

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return reg_loss, clf_loss


def train_single_epoch(model: tf.keras.Model,
                       anchors: tf.Tensor,
                       dataset: tf.data.Dataset,
                       optimizer: tf.optimizers.Optimizer,
                       loss_fn: LossFn,
                       epoch: int,
                       num_classes: int,
                       print_every: int = 10):
    
    running_loss = tf.metrics.Mean()
    running_clf_loss = tf.metrics.Mean()
    running_reg_loss = tf.metrics.Mean()

    for i, (images, (labels, bbs)) in enumerate(dataset):
        
        # TODO: Handle padding
        target_reg, target_clf = \
            utils.anchors.anchor_targets_bbox(anchors.numpy(), 
                                              images.numpy(), 
                                              bbs.numpy(), 
                                              labels.numpy(), 
                                              num_classes)

        reg_loss, clf_loss = _train_step(model=model, 
                                         optimizer=optimizer, 
                                         loss_fn=loss_fn,
                                         images=images, 
                                         regress_targets=target_reg, 
                                         labels=target_clf)

        running_loss((reg_loss + clf_loss) * 0.5)
        running_clf_loss(clf_loss)
        running_reg_loss(reg_loss)

        if (i + 1) % print_every == 0:
            print(f'Epoch[{epoch}] '
                  f'loss: {running_loss.result():.4f} '
                  f'clf. loss: {running_clf_loss.result():.4f} '
                  f'reg. loss: {running_reg_loss.result():.4f} ')