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
    alpha = tf.where(y_true == 1, alpha, 1 - alpha)
    
    pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
    
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
