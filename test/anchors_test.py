import unittest

import cv2
import numpy as np
import tensorflow as tf

import efficientdet.utils as utils
import efficientdet.data.voc as voc
import efficientdet.config as config

def _get_res_at_level_i(res, i):
    return int(res / (2**i))

anchors_config = config.AnchorsConfig()

class AnchorsTest(unittest.TestCase):
    
    def test_tile_anchors(self):
        level = 3
        feature_size = _get_res_at_level_i(512, level)
        im_random = np.zeros((feature_size, feature_size, 3))

        anchors_gen = utils.anchors.AnchorGenerator(
            size=anchors_config.sizes[level - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.strides[level - 3])
        
        boxes = anchors_gen.tile_anchors_over_feature_map(im_random.shape)

        for box in boxes.numpy():
            box = box.astype('int32')
            cv2.rectangle(im_random, 
                          (box[0], box[1]), 
                          (box[2], box[3]), (0, 255, 0), 1)

        cv2.imshow('', im_random)
        cv2.waitKey()

    def test_compute_gt(self):
        level = 3
        ds = voc.build_dataset('test/data/VOC2007',
                               im_input_size=(256, 256))

        anchors_gen = utils.anchors.AnchorGenerator(
            size=anchors_config.sizes[level - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.strides[level - 3])
        
        for im, (l, bbs) in ds.take(1):
            anchors = anchors_gen.tile_anchors_over_feature_map(im[0].shape)
            
            gt_reg, gt_labels = \
                utils.anchors.anchor_targets_bbox(anchors.numpy(), 
                                                  im.numpy(), 
                                                  bbs.numpy(), l.numpy(), 
                                                  len(voc.IDX_2_LABEL))
            nearest_anchors = anchors[gt_reg[0, :, -1] == 1].numpy()
            
            im_random = im[0].numpy()
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
                print(voc.IDX_2_LABEL[int(label)])

            cv2.imshow('', im_random)
            cv2.waitKey()

            print('GT shapes:', gt_labels.shape, gt_reg.shape)
            print('Found any overlapping anchor?', 
                  np.any(gt_labels[:, :, -1] == 1.))

    def test_regress_boxes(self):
        level = 3
        ds = voc.build_dataset('test/data/VOC2007',
                               im_input_size=(256, 256))

        anchors_gen = utils.anchors.AnchorGenerator(
            size=anchors_config.sizes[level - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.strides[level - 3])
        
        for im, (l, bbs) in ds.take(1):
            anchors = anchors_gen.tile_anchors_over_feature_map(im[0].shape)
            
            gt_reg, gt_labels = \
                utils.anchors.anchor_targets_bbox(anchors.numpy(), 
                                                  im.numpy(), 
                                                  bbs.numpy(), l.numpy(), 
                                                  len(voc.IDX_2_LABEL))
            
            near_mask = gt_reg[0, :, -1] == 1
            nearest_regressors = tf.expand_dims(gt_reg[0, near_mask][:, :-1], 0)
            nearest_anchors = tf.expand_dims(anchors[near_mask], 0)

            # apply regression to boxes
            regressed_boxes = utils.bndbox.regress_bndboxes(nearest_anchors, 
                                                            nearest_regressors)

            im_random = im[0].numpy()
            for box in regressed_boxes[0].numpy():
                box = box.astype('int32')
                cv2.rectangle(im_random, 
                              (box[0], box[1]), 
                              (box[2], box[3]), (0, 255, 0), 1)
            
            cv2.imshow('', im_random)
            cv2.waitKey()


if __name__ == "__main__":
    unittest.main()