import unittest

import cv2
import numpy as np

import efficientdet.utils as utils
import efficientdet.data.voc as voc


def _get_res_at_level_i(res, i):
    return int(res / (2**i))

sizes = [32, 64, 128, 256, 512]
strides = [8, 16, 32, 64, 128]

class AnchorsTest(unittest.TestCase):
    
    def test_tile_anchors(self):
        level = 3
        feature_size = _get_res_at_level_i(512, level)
        im_random = np.zeros((feature_size, feature_size, 3))

        anchors_gen = utils.anchors.AnchorGenerator(
            size=sizes[level - 3],
            aspect_ratios=[.5, 1, 2],
            stride=strides[level - 3])
        
        boxes = anchors_gen.tile_anchors_over_feature_map(im_random)

    #     print(boxes.shape)
    #     for box in boxes:
    #         box = box.astype('int32')
    #         cv2.rectangle(im_random, 
    #                     (box[0], box[1]), 
    #                     (box[2], box[3]), (0, 255, 0), 1)

    #     # cv2.imshow('', im_random)
    #     # cv2.waitKey()

    def test_compute_gt(self):
        level = 3
        ds = voc.build_dataset('test/data/VOC2007',
                               im_input_size=(256, 256))

        anchors_gen = utils.anchors.AnchorGenerator(
            size=sizes[level - 3],
            aspect_ratios=[.5, 1, 2],
            stride=strides[level - 3])
        
        for im, (l, bbs) in ds.take(1):
            anchors = anchors_gen.tile_anchors_over_feature_map(im[0])
            
            gt_reg, gt_labels = \
                utils.anchors.anchor_targets_bbox(anchors.numpy(), 
                                                  im.numpy(), 
                                                  bbs.numpy(), l.numpy(), 
                                                  len(voc.IDX_2_LABEL))
            nearest_anchors = anchors[gt_reg[0, :, -1] == 1].numpy()
            
    #         im_random = im[0].numpy()
    #         for box in nearest_anchors:
    #             box = box.astype('int32')
    #             cv2.rectangle(im_random, 
    #                           (box[0], box[1]), 
    #                           (box[2], box[3]), (0, 255, 0), 1)

    #         for box in bbs.numpy()[0]:
    #             box = box.astype('int32')
    #             cv2.rectangle(im_random, 
    #                           (box[0], box[1]), 
    #                           (box[2], box[3]), (0, 0, 255), 1)
            
    #         for label in l[0]:
    #             print(voc.IDX_2_LABEL[int(label)])

    #         cv2.imshow('', im_random)
    #         cv2.waitKey()

            print('GT shapes:', gt_labels.shape, gt_reg.shape)
            print('Found any overlapping anchor?', 
                  np.any(gt_labels[:, :, -1] == 1.))

if __name__ == "__main__":
    unittest.main()