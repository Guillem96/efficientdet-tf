import unittest

import tensorflow as tf
from efficientdet.models import BiFPN


def _get_res_at_level_i(res, i):
    return int(res / (2**i))


class BiFPNTest(unittest.TestCase):

    def test_forward(self):
        batch_size = 2
        input_res = 640
        model = BiFPN()

        resolutions = [_get_res_at_level_i(input_res, i)
                    for i in range(3, 8)]
        Ps = [tf.random.uniform([batch_size, r, r, 64]) 
            for r in resolutions]

        P_out = model(Ps)
        print([p.shape for p in P_out])


if __name__ == "__main__":
    unittest.main()