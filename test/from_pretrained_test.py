import unittest

import tensorflow as tf

from efficientdet.utils import io
from efficientdet import EfficientDet
from efficientdet.data import voc
from efficientdet.data.preprocess import normalize_image


class PretrainedTest(unittest.TestCase):

    def test_pretrained(self):
        # Load efficientdet pretrained on VOC2007
        effdet = EfficientDet.from_pretrained('D0-VOC2007', 
                                              score_threshold=.6)
        
        image = io.load_image(
            'test/data/VOC2007/JPEGImages/000001.jpg',
            (effdet.config.im_size,) * 2)
        image = normalize_image(image)
        image = tf.expand_dims(image, 0)

        class_2_idx = voc.IDX_2_LABEL
        

        

if __name__ == "__main__":
    unittest.main()