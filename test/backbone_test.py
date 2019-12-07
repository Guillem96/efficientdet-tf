import unittest

import tensorflow as tf
import efficientdet.models as models


def _get_res_at_level_i(res, i):
    return int(res / (2**i))


def _create_backbone(B = 0):
    return models.build_efficient_net_backbone(B = 0, 
                                                weights=None)


class BackboneTest(unittest.TestCase):

    def test_forward(self):
        input_size = 640
        inputs = tf.random.uniform([2, input_size, input_size, 3])

        model = _create_backbone(1)
        
        features = model(inputs)
        P3, P4, P5, P6, P7 = features

        # TODO: Features between efficientnet conv block are not being
        # reduced
        # for level in range(3, 8):
        #     self.assertEqual(features[level - 3].shape[1], 
        #                      _get_res_at_level_i(input_size, level),
        #                      msg='Unexpected shape at level {}'.format(level))

if __name__ == "__main__":
    unittest.main()