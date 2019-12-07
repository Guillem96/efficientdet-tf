import tensorflow as tf


def focal_loss(y_true: tf.Tensor,
               y_pred: tf.Tensor,
               gamma: int = 2,
               from_logits: bool = False,
               reduction: str = 'sum'):

    bce_fn = tf.losses.BinaryCrossentropy(
        from_logits=from_logits,
        reduction=tf.losses.Reduction.NONE)

    bce = bce_fn(y_true, y_pred)

    if from_logits:
        y_pred = tf.sigmoid(y_pred)

    y_true = tf.cast(y_true, tf.float32)
    
    # if y == 1:
    #   pt = prob
    # else
    #   pt = 1 - prob
    pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    pt = tf.reshape(pt, [-1])

    weight = (1 - pt) ** gamma
    loss = tf.multiply(weight, bce)

    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)

    return loss

