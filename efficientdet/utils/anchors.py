import math
from typing import List

import tensorflow as tf
import numpy as np # TODO: Migrate to tf

class AnchorGenerator(object):
    
    def __init__(self, 
                 size: float,
                 aspect_ratios: List[float],
                 stride: int = 1):
        """
        RetinaNet input examples:
            size: 32
            aspect_ratios: [0.5, 1, 2]
        """
        self.size = size
        self.stride = stride

        self.aspect_ratios = aspect_ratios
        self.anchor_scales = [
            2 ** 0,
            2 ** (1 / 3.0),
            2 ** (2 / 3.0)]

        self.anchors = self._generate()
    
    def tile_anchors_over_feature_map(self, feature_map):
        shift_x = (np.arange(0, feature_map.shape[0], step=self.stride) + 0.5)
        shift_y = (np.arange(0, feature_map.shape[1], step=self.stride) + 0.5)

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = len(self)
        K = shifts.shape[0]
        all_anchors = (self.anchors.reshape((1, A, 4)) 
                       + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))

        return all_anchors

    def _generate(self) -> tf.Tensor:
        num_anchors = len(self)
        ratios = np.array(self.aspect_ratios)
        scales = np.array(self.anchor_scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = self.size * np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def __len__(self):
        return len(self.aspect_ratios) * len(self.anchor_scales)
    