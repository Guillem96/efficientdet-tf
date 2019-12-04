import math
from typing import List

import tensorflow as tf


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
        self.anchor_sizes = [
            size,
            size * 2 ** (1 / 3.0),
            size * 2 ** (2 / 3.0)]

        self.anchors = self._generate()
    
    def tile_anchors_over_feature_map(self):
        shifts_x = tf.range(self.size * self.stride, delta=self.stride)
        shifts_y = tf.range(self.size * self.stride, delta=self.stride)
        shift_y, shift_x = tf.meshgrid(shifts_y, shifts_x)
        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])
        
        shifts = tf.stack([shift_x, shift_y, ] * 2, axis=1)
        shifts = tf.reshape(shifts, [-1, 1, 4])
        shifts = tf.cast(shifts, tf.float32)
        
        boxes = tf.expand_dims(self.anchors, axis=0)

        tiled_boxes = tf.reshape(shifts + boxes, [-1, 4])

        return tiled_boxes 

    def _generate(self) -> tf.Tensor:
        anchors = []
        for size in self.anchor_sizes:
            area = size ** 2.0
            for aspect_ratio in self.aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return tf.stack(anchors)

    def __len__(self):
        return len(self.aspect_ratios) * len(self.anchor_sizes)
    