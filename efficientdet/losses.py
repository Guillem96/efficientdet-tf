import abc
from typing import Callable

import tensorflow as tf


def focal_loss(y_true: tf.Tensor,
               y_pred: tf.Tensor,
               gamma: int = 1.5,
               alpha: float = 0.25,
               from_logits: bool = False,
               reduction: str = 'sum'):

    if from_logits:
        y_pred = tf.sigmoid(y_pred)
    
    epsilon = 1e-6
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_true = tf.cast(y_true, tf.float32)

    alpha = tf.ones_like(y_true) * alpha 
    alpha = tf.where(tf.equal(y_true, 1.), alpha, 1 - alpha)
    
    pt = tf.where(tf.equal(y_true, 1.), y_pred, 1 - y_pred)
    
    loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
    loss = tf.reduce_sum(loss, axis=-1)
    
    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)

    return loss


def huber_loss(y_true: tf.Tensor, 
               y_pred: tf.Tensor, 
               clip_delta: float = 1.0,
               reduction: str = 'sum'):

    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    error = y_true - y_pred
    cond  = tf.abs(error) < clip_delta

    squared_loss = 0.5 * tf.square(error)
    linear_loss  = clip_delta * (tf.abs(error) - 0.5 * clip_delta)

    loss = tf.where(cond, squared_loss, linear_loss)
    loss = tf.reduce_mean(loss, axis=-1)

    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)

    return loss


class _EfficientDetLoss(abc.ABC, tf.keras.losses.Loss):
    
    def __init__(self) -> None:
        super(_EfficientDetLoss, self).__init__()
        self.name = 'efficientdet_loss_'
        self.name += 'clf' if self.is_clf else 'reg'
    
    @abc.abstractproperty
    def is_clf(self) -> tf.Tensor:
        raise NotImplemented
    
    @abc.abstractproperty
    def loss_fn(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        raise NotImplemented

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_shape = tf.shape(y_true)
        batch = y_shape[0]
        n_anchors = y_shape[1]

        anchors_states = y_true[:, :, -1]
        not_ignore_idx = tf.where(tf.not_equal(anchors_states, -1.))
        true_idx = tf.where(tf.equal(anchors_states, 1.))

        normalizer = tf.shape(true_idx)[0]
        normalizer = tf.cast(normalizer, tf.float32)

        # We only regress true boxes, but we classify positive and negative
        # instances
        indexer = tf.cond(self.is_clf, lambda: not_ignore_idx, lambda: true_idx)

        y_true = tf.gather_nd(y_true[:, :, :-1], indexer)
        y_pred = tf.gather_nd(y_pred, indexer)

        return tf.divide(self.loss_fn(y_true, y_pred), normalizer)


class EfficientDetFocalLoss(_EfficientDetLoss):

    @property
    def is_clf(self) -> tf.Tensor: return tf.constant(True, dtype=tf.bool)

    @property
    def loss_fn(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        return focal_loss


class EfficientDetHuberLoss(_EfficientDetLoss):

    @property
    def is_clf(self) -> tf.Tensor: return tf.constant(False, dtype=tf.bool)

    @property
    def loss_fn(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        return tf.losses.Huber(reduction=tf.losses.Reduction.SUM)
