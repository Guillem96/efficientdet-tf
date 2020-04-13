import unittest

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import efficientdet.data.voc as voc
import efficientdet.config as config
from efficientdet import utils, visualizer
from efficientdet.data.preprocess import unnormalize_image

def _get_res_at_level_i(res, i):
    return int(res / (2**i))

anchors_config = config.AnchorsConfig()

class AnchorsTest(unittest.TestCase):
    
    def generate_anchors(self,
                         anchors_config: config.AnchorsConfig,
                         im_shape: int) -> tf.Tensor:

        anchors_gen = [utils.anchors.AnchorGenerator(
                size=anchors_config.sizes[i - 3],
                aspect_ratios=anchors_config.ratios,
                stride=anchors_config.strides[i - 3]) 
                for i in range(3, 8)]
        
        shapes = [im_shape // (2 ** x) for x in range(3, 8)]

        anchors = [g((size, size, 3))
                   for g, size in zip(anchors_gen, shapes)]

        return tf.concat(anchors, axis=0)

    def test_tile_anchors(self):
        level = 3
        feature_size = 512
        im_random = np.zeros((512, 512, 3))

        boxes = self.generate_anchors(config.AnchorsConfig(), 
                                      im_random.shape[0])

        im_random = visualizer.draw_boxes(im_random, boxes)

        plt.imshow(im_random)
        plt.show(block=True)

    def test_compute_gt(self):
        level = 3
        ds = voc.build_dataset('test/data/VOC2007',
                               im_input_size=(512, 512))

        anchors = self.generate_anchors(config.AnchorsConfig(), 
                                        512)
        
        for im, (l, bbs) in ds.take(1):

            gt_reg, gt_labels = utils.anchors.anchor_targets_bbox(
                anchors, im, bbs, l, len(voc.IDX_2_LABEL))

            nearest_anchors = anchors[gt_reg[0, :, -1] == 1].numpy()
            im_random =  unnormalize_image(im[0])
            im_random = visualizer.draw_boxes(im_random, nearest_anchors)
            im_random = visualizer.draw_boxes(
                im_random, bbs[0], colors=[0, 0, 255])
            
            for label in l[0]:
                print(voc.IDX_2_LABEL[int(label)])

            plt.imshow(im_random)
            plt.show(block=True)

            print('GT shapes:', gt_labels.shape, gt_reg.shape)
            print('Found any overlapping anchor?', 
                  np.any(gt_labels[:, :, -1] == 1.))

    def test_regress_boxes(self):
        print('Regress anchors test')

        level = 3
        ds = voc.build_dataset('test/data/VOC2007',
                               im_input_size=(512, 512))

        anchors = self.generate_anchors(config.AnchorsConfig(), 512)
        
        for im, (l, bbs) in ds.take(1):
            
            gt_reg, gt_labels = utils.anchors.anchor_targets_bbox(
                anchors, im, bbs, l, len(voc.IDX_2_LABEL))

            near_mask = gt_reg[0, :, -1] == 1
            nearest_regressors = tf.expand_dims(
                tf.boolean_mask(gt_reg[0], near_mask)[:, :-1], 0)
            nearest_anchors = tf.expand_dims(anchors[near_mask], 0)

            # apply regression to boxes
            regressed_boxes = utils.bndbox.regress_bndboxes(nearest_anchors, 
                                                            nearest_regressors)

            im_random = unnormalize_image(im[0])
            im_random = visualizer.draw_boxes(im_random, regressed_boxes[0])
            
            plt.imshow(im_random)
            plt.show(block=True)


if __name__ == "__main__":
    unittest.main()