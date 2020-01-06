from typing import List

import tensorflow as tf

import efficientdet.utils as utils
import efficientdet.config as config
from efficientdet import models


class EfficientDet(tf.keras.Model):
    """
    Parameters
    ----------
    num_classes: int
        Number of classes to classify
    D: int, default 0
        EfficientDet architecture based on a compound scaling,
        to better understand this parameter refer to EfficientDet 
        paper 4.2 section
    weights: str, default 'imagenet'
        If set to 'imagenet' then the backbone will be pretrained
        on imagenet. If set to None, the backbone will be random
        initialized
    """
    def __init__(self, 
                 num_classes: int,
                 D : int = 0, 
                 weights : str = 'imagenet'):
        super(EfficientDet, self).__init__()
        self.config = config.EfficientDetCompudScaling(D=D)
        self.anchors_config = config.AnchorsConfig()
        self.num_classes = num_classes

        self.backbone = (models
                         .build_efficient_net_backbone(self.config.B, 
                                                       weights))

        self.bifpn = models.BiFPN(self.config.Wbifpn, self.config.Dbifpn)

        self.class_head = models.RetinaNetClassifier(self.config.Wbifpn,
                                                     self.config.Dclass,
                                                     num_classes)
        self.bb_head = models.RetinaNetBBPredictor(self.config.Wbifpn,
                                                   self.config.Dclass)

        self.anchors_gen = [utils.anchors.AnchorGenerator(
            size=self.anchors_config.sizes[i - 3],
            aspect_ratios=self.anchors_config.ratios,
            stride=self.anchors_config.strides[i - 3]
        ) for i in range(3, 8)] # 3 to 7 pyramid levels

    def _compute_gt_anchors(self, feature_anchors, inputs):
        images, (labels, bndboxes) = inputs
        
        # TODO: No depend on numpy while training
        reg, clf = \
            utils.anchors.anchor_targets_bbox(feature_anchors.numpy(), 
                                              images.numpy(), 
                                              bndboxes.numpy(), 
                                              labels.numpy(), 
                                              self.num_classes)

        return tf.constant(reg), tf.constant(clf)

    def call(self, inputs: List[tf.Tensor], training: bool = True):
        """
        EfficientDet forward step

        Parameters
        ----------
        inputs: List[tf.Tensor]
            List with 2 positions containing batch of images
            and the annotations for each image.
            Annotations is a tuple of two elements:
                annotations[0] == labels, annotations[1] == bounding boxes
        training: bool
            Wether if model is training or it is in inference mode

        Examples
        --------
        >>> model = EfficientDet(num_classes=2)
        >>> images = tf.random.uniform([2, 128, 128, 128]) # Mock images
        >>> labels = tf.random.uniform([2, 1], maxval=2, dtype=tf.int32)
        >>> boxes = ... # Random boxes with shape [2, 1, 4] # One box per label
        >>> model([images, (labels, boxes)], training=True)
        """
        images, (labels, bndboxes) = inputs
        features = self.backbone(images)

        # List of [BATCH, H, W, C]
        bifnp_features = self.bifpn(features)

        # List of [BATCH, A * 4]
        bboxes = [self.bb_head(bf) for bf in bifnp_features]

        # List of [BATCH, A * num_classes]
        class_scores = [self.class_head(bf) for bf in bifnp_features]

        # # [BATCH, H, W, A * 4]
        bboxes = tf.concat(bboxes, axis=1)
        class_scores = tf.concat(class_scores, axis=1)

        # Create the anchors
        anchors = [anchor_gen(f[0])
                  for anchor_gen, f in zip(self.anchors_gen, bifnp_features)]
        anchors = tf.concat(anchors, axis=0)

        if training:
            # # In case we are in training mode, we have access to ground truths
            # reg_targets, clf_targets = self._compute_gt_anchors(anchors, inputs)
            return bboxes, class_scores, anchors
        else:
            # Tile anchors over batches, so they can be regressed
            batch_size = bboxes.shape[0]
            anchors = tf.tile(tf.expand_dims(anchors, 0), 
                              [bboxes.shape[0], 1, 1])
            
            class_scores = tf.reshape(class_scores, 
                                      [batch_size, -1, self.num_classes])
            bboxes = tf.reshape(bboxes, 
                                [batch_size, -1, 4])

            boxes = utils.bndbox.regress_bndboxes(anchors, bboxes)
            boxes = utils.bndbox.clip_boxes(boxes, images.shape[1:3])
            boxes, labels = utils.bndbox.nms(boxes, class_scores)
            # TODO: Pad output
            return boxes, labels