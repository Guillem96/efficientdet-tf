import unittest 

import cv2
import tensorflow as tf

import efficientdet.utils as utils
import efficientdet.data.voc as voc
import efficientdet.config as config
import efficientdet.utils.bndbox as bb_utils

anchors_config = config.AnchorsConfig()


class NMSTest(unittest.TestCase):

    def test_nms(self):
        batch_size = 2
        level = 3
        n_classes = len(voc.LABEL_2_IDX)

        ds = voc.build_dataset('test/data/VOC2007',
                               batch_size=batch_size,
                               im_input_size=(256, 256))
        
        anchors_gen = utils.anchors.AnchorGenerator(
            size=anchors_config.sizes[level - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.strides[level - 3])
        
        for im, (l, bbs) in ds.take(1):
            anchors = anchors_gen.tile_anchors_over_feature_map(im[0])
            
            gt_reg, gt_labels = \
                utils.anchors.anchor_targets_bbox(anchors.numpy(), 
                                                  im.numpy(), 
                                                  bbs.numpy(), l.numpy(), 
                                                  n_classes)

            box_score = gt_labels[0]
            before_nms_shape = anchors.shape

            boxes, labels = bb_utils.nms(tf.expand_dims(anchors, 0), 
                                         tf.expand_dims(box_score, 0))
            after_nms_shape = boxes[0].shape

            self.assertTrue(after_nms_shape[0] < before_nms_shape[0],
                            'After nms boxes should be reduced')

            im_random = im[0].numpy()
            for box in boxes[0].numpy():
                box = box.astype('int32')
                cv2.rectangle(im_random, 
                              (box[0], box[1]), 
                              (box[2], box[3]), (0, 255, 0), 1)
            
            cv2.imshow('', im_random)
            cv2.waitKey()

