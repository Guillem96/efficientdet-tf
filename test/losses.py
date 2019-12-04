import tensorflow as tf

from efficientdet import losses


def focal_loss_test():
    y_true = tf.random.uniform(shape=(32, 1),
                               minval=0, maxval=2, 
                               dtype=tf.int32)
    y_pred = tf.random.uniform(shape=(32, 1))
    
    loss = losses.binary_focal_loss(y_true, y_pred)

    print(loss)
    

focal_loss_test()