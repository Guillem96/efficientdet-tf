import unittest

import tensorflow as tf

import efficientdet.models as models

class EfficientDetTest(unittest.TestCase):

    def _compare_shapes(self, shape1, shape2):
        for s1, s2 in zip(shape1, shape2):
            if s1 is not None and s2 is not None:
                self.assertEqual(s1, s2)

    def test_forward(self):
        num_classes = 2
        batch_size = 2
        model = models.EfficientDet(num_classes=num_classes,
                                    D=0, 
                                    weights=None)

        input_size = model.config.input_size
        inputs = tf.random.uniform([batch_size, input_size, input_size, 3], dtype=tf.float32)

        bb, clf = model(inputs)

        self._compare_shapes(bb.shape, [batch_size, None, 9, 4])
        self._compare_shapes(clf.shape, [batch_size, None, 9, num_classes])
        

if __name__ == "__main__":
    unittest.main()
