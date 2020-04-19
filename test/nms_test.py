import unittest 

import tensorflow as tf
import matplotlib.pyplot as plt

import efficientdet.utils as utils
import efficientdet.config as config
import efficientdet.data.labelme as labelme
import efficientdet.utils.bndbox as bb_utils

anchors_config = config.AnchorsConfig()


class NMSTest(unittest.TestCase):

    def test_nms(self):
        anchors_config = config.AnchorsConfig()

        classes, class2idx = utils.io.read_class_names(
            'test/data/pokemon/classes.names') 
        n_classes = len(classes)
        ds = labelme.build_dataset('test/data/pokemon',
                                   'test/data/pokemon',
                                   class2idx=class2idx,
                                   im_input_size=(512, 512))
        
        anchors_gen = [utils.anchors.AnchorGenerator(
            size=anchors_config.sizes[i - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.strides[i - 3]) 
            for i in range(3, 8)]
        
        sizes = (80, 40, 20, 10, 5)
        im, (l, bbs) = next(iter(ds.take(1)))

        anchors = [anchor_gen((size, size, 3))
                   for anchor_gen, size in zip(anchors_gen, sizes)]
        anchors = tf.concat(anchors, axis=0)
        
        gt_reg, gt_labels = utils.anchors.anchor_targets_bbox(
            anchors, 
            tf.expand_dims(im, 0),
            tf.expand_dims(bbs, 0),
            tf.expand_dims(l, 0), 
            n_classes)

        box_score = gt_labels[0]
        true_idx = tf.reshape(tf.where(box_score[:, -1] == 1), [-1])
        
        box_score = tf.gather(box_score, true_idx)
        anchors = tf.gather(anchors, true_idx)

        before_nms_shape = anchors.shape

        anchors = tf.expand_dims(anchors, 0)
        box_score = tf.expand_dims(box_score[:, :-1], 0)
        boxes, labels, scores = bb_utils.nms(anchors, box_score)
        after_nms_shape = boxes[0].shape
        
        if anchors.shape[0] != 0:
            self.assertTrue(after_nms_shape[0] < before_nms_shape[0],
                            'After nms boxes should be reduced')
        else: 
            print('No ground truth anchors')
            
        im_random = utils.visualizer.draw_boxes(im, boxes[0])
        plt.imshow(im_random)
        plt.axis('off')
        plt.show(block=True)


if __name__ == "__main__":
    unittest.main()