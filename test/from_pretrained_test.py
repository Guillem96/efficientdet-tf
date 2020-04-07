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
        model = EfficientDet.from_pretrained('D0-VOC', 
                                            score_threshold=.6)
        print('Done loading...')
        image = io.load_image(
            'test/data/VOC2007/JPEGImages/000002.jpg',
            (model.config.input_size,) * 2)
        n_image = normalize_image(image)
        n_image = tf.expand_dims(n_image, axis=0)

        classes = voc.IDX_2_LABEL

        boxes, labels, scores = model(n_image, training=False)
        labels = [classes[l] for l in labels[0]]

        im = image.numpy()
        for l, box, s in zip(labels, boxes[0].numpy(), scores[0]):
            x1, y1, x2, y2 = box.astype('int32')

            cv2.rectangle(im, 
                        (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, l + ' {:.2f}'.format(s), 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_PLAIN, 
                        2, (0, 255, 0), 2)
                        
        plt.imshow(im)
        plt.axis('off')
        plt.savefig('test.png')
        plt.show(block=True)


if __name__ == "__main__":
    unittest.main()