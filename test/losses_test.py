import unittest

import tensorflow as tf

from efficientdet import losses


class FocalLossTest(unittest.TestCase):
    
    def test_focal_loss(self):
        y_clf_true = tf.random.uniform(shape=(32, 1),
                                       minval=0, maxval=2, 
                                       dtype=tf.int32)
        y_clf_pred = tf.random.uniform(shape=(32, 1))
        
        loss = losses.focal_loss(y_clf_true, y_clf_pred)

        print(loss)
    
    def test_huber_loss(self):
        y_reg_true = tf.random.uniform(shape=(32, 10, 4))
        y_reg_pred = tf.random.uniform(shape=(32, 10, 4))

        print(losses.huber_loss(y_reg_true, y_reg_pred))


if __name__ == "__main__":
    unittest.main()