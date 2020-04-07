import unittest

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from efficientdet.utils import io
from efficientdet import EfficientDet
from efficientdet.data import voc
from efficientdet.data.preprocess import normalize_image


class PretrainedTest(unittest.TestCase):

    def test_pretrained(self):
        # Load efficientdet pretrained on VOC2007
        effdet = EfficientDet.from_pretrained('D0-VOC2007', 
                                              score_threshold=.1)
        
        image = io.load_image(
            'data/VOC2007/images/000004.jpg',
            (effdet.config.input_size,) * 2)
        n_image = normalize_image(image)
        n_image = tf.expand_dims(image, 0)

        classes = voc.IDX_2_LABEL
        
        boxes, labels, scores = effdet(n_image, training=False)
        labels = [classes[l] for l in labels[0]]

        im = image.numpy()
        for l, box, s in zip(labels, boxes[0].numpy(), scores[0]):
            print('Hellooo')
            x1, y1, x2, y2 = box.astype('int32')

            cv2.rectangle(im, 
                        (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, l + ' {:.2f}'.format(s), 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_PLAIN, 
                        2, (0, 255, 0), 2)
                        
        plt.imshow(im)
        plt.axis('off')
        plt.show(block=True)


if __name__ == "__main__":
    unittest.main()