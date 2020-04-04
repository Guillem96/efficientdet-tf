import unittest

import cv2

import tensorflow as tf

from efficientdet import data


class AugmentationTest(unittest.TestCase):
    
    def test_flip_horizontal(self):
        classes = ['treecko', 'psyduck', 'greninja', 'solgaleo', 'mewtwo']
        class2idx = {c: i for i, c in enumerate(classes)}
        ds = data.labelme.build_dataset('test/data/pokemon',
                                        'test/data/pokemon',
                                        class2idx=class2idx,
                                        im_input_size=(512, 512),
                                        batch_size=1)
        ds = ds.unbatch()
        aug_ds = ds.map(data.preprocess.crop)

        for image, (labels, boxes) in ds.take(1):
            print(image.shape)
            
            

            image = data.preprocess.unnormalize_image(image) * 255
            image = image.numpy()[..., ::-1].copy().astype('uint8')
            for box in boxes.numpy():
                box = box.astype('int32')
                cv2.rectangle(image, 
                              (box[0], box[1]), 
                              (box[2], box[3]), (0, 0, 255), 1)
            
            cv2.imwrite('test.png', image)

        for image, (labels, boxes) in aug_ds.take(1):
            print(image.shape)
            
            image = data.preprocess.unnormalize_image(image) * 255
            image = image.numpy()[..., ::-1].copy().astype('uint8')
            for box in boxes.numpy():
                box = box.astype('int32')
                cv2.rectangle(image, 
                              (box[0], box[1]), 
                              (box[2], box[3]), (0, 0, 255), 1)
            
            cv2.imwrite('test_2.png', image)


if __name__ == "__main__":
    unittest.main()