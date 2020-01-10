import unittest

import unittest

import cv2
import numpy as np
import tensorflow as tf

import efficientdet.utils as utils
import efficientdet.config as config
import efficientdet.data.labelme as labelme


class LabelmeDatasetTest(unittest.TestCase):

    def generate_anchors(self,
                         anchors_config: config.AnchorsConfig,
                         im_shape: int) -> tf.Tensor:

        anchors_gen = [utils.anchors.AnchorGenerator(
                size=anchors_config.sizes[i - 3],
                aspect_ratios=anchors_config.ratios,
                stride=anchors_config.strides[i - 3]) 
                for i in range(3, 8)]
        
        shapes = [im_shape // (2 ** (x - 2)) for x in range(3, 8)]

        anchors = [g((size, size, 3))
                for g, size in zip(anchors_gen, shapes)]

        return tf.concat(anchors, axis=0)

    def test_compute_gt(self):
        classes = ['treecko', 'psyduck', 'greninja', 'solgaleo', 'mewtwo']
        class2idx = {c: i for i, c in enumerate(classes)}
        ds = labelme.build_dataset('test/data/pokemon',
                                   'test/data/pokemon',
                                   class2idx=class2idx,
                                   im_input_size=(512, 512))

        anchors = self.generate_anchors(config.AnchorsConfig(), 512)
        
        for im, (l, bbs) in ds.take(1):

            gt_reg, gt_labels = \
                utils.anchors.anchor_targets_bbox(anchors.numpy(), 
                                                  im.numpy(), 
                                                  bbs.numpy(), l.numpy(), 
                                                  len(classes))
            nearest_anchors = anchors[gt_reg[0, :, -1] == 1].numpy()
            
            im_random = im[0].numpy()[..., ::-1].copy()
            for box in nearest_anchors:
                box = box.astype('int32')
                cv2.rectangle(im_random, 
                              (box[0], box[1]), 
                              (box[2], box[3]), (0, 255, 0), 1)

            for box in bbs.numpy()[0]:
                box = box.astype('int32')
                cv2.rectangle(im_random, 
                              (box[0], box[1]), 
                              (box[2], box[3]), (0, 0, 255), 1)
            
            for label in l[0]:
                print(classes[int(label)])

            cv2.imshow('', im_random)
            cv2.waitKey()

            print('GT shapes:', gt_labels.shape, gt_reg.shape)
            print('Found any overlapping anchor?', 
                  np.any(gt_labels[:, :, -1] == 1.))


if __name__ == "__main__":
    unittest.main()