import unittest

import matplotlib.pyplot as plt
import tensorflow as tf

from efficientdet import data
from efficientdet import visualizer


class AugmentationTest(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(AugmentationTest, self).__init__(*args, **kwargs)
        self.classes = ['treecko', 'psyduck', 'greninja', 'solgaleo', 'mewtwo']
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.ds = data.labelme.build_dataset('test/data/pokemon',
                                             'test/data/pokemon',
                                             self.class2idx,
                                             im_input_size=(512, 512))

    def plot_single(self, aug_fn):
        image, (labels, boxes) = next(iter(self.ds.take(1)))
        aug_image, (aug_labels, aug_boxes) = aug_fn(image, (labels, boxes))

        image = data.preprocess.unnormalize_image(image)
        aug_image = data.preprocess.unnormalize_image(aug_image)
        

        labels = [self.classes[l] for l in labels.numpy().tolist()]
        aug_labels = [self.classes[l] for l in aug_labels.numpy().tolist()]
        
        colors = visualizer.colors_per_labels(labels)
        aug_colors = visualizer.colors_per_labels(aug_labels)

        image = visualizer.draw_boxes(image, boxes, labels, colors=colors)
        aug_image = visualizer.draw_boxes(
            aug_image, aug_boxes, aug_labels, colors=aug_colors)

        plt.subplot(121)
        plt.title('No Image augmentation')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(122)
        plt.title('Image augmentation')
        plt.imshow(aug_image)
        plt.axis('off')

        plt.show(block=True)

    def test_flip_horizontal(self):
        self.plot_single(aug_fn=data.preprocess.horizontal_flip)

    def test_crop(self):
        self.plot_single(aug_fn=data.preprocess.crop)


if __name__ == "__main__":
    unittest.main()