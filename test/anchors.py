import cv2
import numpy as np
from efficientdet.utils import anchors


def _get_res_at_level_i(res, i):
    return int(res / (2**i))


def tile_anchors_test():
    level = 3
    feature_size = _get_res_at_level_i(512, level)
    sizes = [32, 64, 128, 256, 512]
    strides = [8, 16, 32, 64, 128]
    im_random = np.zeros((feature_size, feature_size, 3))

    anchors_gen = anchors.AnchorGenerator(
        size=sizes[level - 3],
        aspect_ratios=[.5, 1, 2],
        stride=strides[level - 3])
    
    boxes = anchors_gen.tile_anchors_over_feature_map(im_random)

    # boxes = boxes[:, 0: 2] + feature_size / 2
    print(boxes.shape)
    for box in boxes:
        box = box.astype('int32')
        cv2.rectangle(im_random, 
                      (box[0], box[1]), 
                      (box[2], box[3]), (0, 255, 0), 1)

    cv2.imshow('', im_random)
    cv2.waitKey()


tile_anchors_test()