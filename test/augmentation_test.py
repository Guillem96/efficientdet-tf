import unittest

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from efficientdet import data


class AugmentationTest(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(AugmentationTest, self).__init__(*args, **kwargs)
        self.classes = ['treecko', 'psyduck', 'greninja', 'solgaleo', 'mewtwo']
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        ds = data.labelme.build_dataset('data/pokemon',
                                        'data/pokemon',
                                        self.class2idx,
                                        im_input_size=(512, 512),
                                        batch_size=1)
        self.ds = ds.unbatch()

    def plot_single(self, aug_fn):

        for image, (labels, boxes) in self.ds.shuffle(20).take(1):
            aug_image, (aug_labels, aug_boxes) = aug_fn(image, (labels, boxes))

            image = data.preprocess.unnormalize_image(image) * 255
            aug_image = data.preprocess.unnormalize_image(aug_image) * 255
            
            image = image.numpy().astype('uint8')
            aug_image = aug_image.numpy().astype('uint8')
            
            labels = [self.classes[l] for l in labels.numpy().tolist()]
            print('WO aug:', labels)

            for box, l in zip(boxes.numpy(), labels):
                box = box.astype('int32')
                cv2.rectangle(image, 
                              (box[0], box[1]), 
                              (box[2], box[3]), (0, 0, 255), 3)
                cv2.putText(image, l , 
                            (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            2, (0, 255, 0), 2)

            plt.subplot(121)
            plt.title('Without image augmentation')
            plt.imshow(image)
            plt.axis('off')

            labels = [self.classes[l] for l in aug_labels.numpy().tolist()]
            print('W aug:', labels)

            for box, l in zip(aug_boxes.numpy(), labels):
                box = box.astype('int32')
                cv2.rectangle(aug_image, 
                              (box[0], box[1]), 
                              (box[2], box[3]), (0, 0, 255), 3)
                cv2.putText(aug_image, l , 
                            (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            2, (0, 255, 0), 2)
            plt.subplot(122)
            plt.title('Image augmentation')
            plt.imshow(aug_image)
            plt.axis('off')

        plt.show(block=True)

    def test_flip_horizontal(self):
        self.plot_single(aug_fn=data.preprocess.horizontal_flip)

    def test_crop(self):
        for i in range(5):
            self.plot_single(aug_fn=data.preprocess.crop)


if __name__ == "__main__":
    unittest.main()