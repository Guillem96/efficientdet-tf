import unittest

import tensorflow as tf
import matplotlib.pyplot as plt

from efficientdet.utils import io
from efficientdet.data import voc
from efficientdet import visualizer
from efficientdet import EfficientDet
from efficientdet.data.preprocess import normalize_image


class PretrainedTest(unittest.TestCase):

    def test_pretrained(self):
        # Load efficientdet pretrained on VOC2007
        model = EfficientDet.from_pretrained('D0-VOC', 
                                            score_threshold=.6)
        print('Done loading...')
        image = io.load_image('imgs/cat-dog.jpg', (model.config.input_size,) * 2)
        n_image = normalize_image(image)
        n_image = tf.expand_dims(n_image, axis=0)

        classes = voc.IDX_2_LABEL

        boxes, labels, scores = model(n_image, training=False)
        labels = [classes[l] for l in labels[0]]

        colors = visualizer.colors_per_labels(labels)
        im = visualizer.draw_boxes(
            image, boxes[0], labels, scores[0], colors=colors)
         
        plt.imshow(im)
        plt.axis('off')
        plt.show(block=True)


if __name__ == "__main__":
    unittest.main()