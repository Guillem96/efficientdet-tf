import unittest

import tensorflow as tf

from efficientdet import losses


class FocalLossTest(unittest.TestCase):
    def test_focal_loss(self):
        y_true = tf.random.uniform(shape=(32, 1),
                                minval=0, maxval=2, 
                                dtype=tf.int32)
        y_pred = tf.random.uniform(shape=(32, 1))
        
        loss = losses.focal_loss(y_true, y_pred)

        print(loss)
    

if __name__ == "__main__":
    unittest.main()