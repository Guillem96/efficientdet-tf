from pathlib import Path
from typing import List, Union

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
    freeze_backbone: bool, default False
    weights: str, default 'imagenet'
        If set to 'imagenet' then the backbone will be pretrained
        on imagenet. If set to None, the backbone will be random
        initialized
    """
    def __init__(self, 
                 num_classes: int,
                 D : int = 0, 
                 bidirectional: bool = True,
                 freeze_backbone: bool = False,
                 score_threshold: float = .01,
                 weights : str = 'imagenet'):
                 
        super(EfficientDet, self).__init__()
        self.config = config.EfficientDetCompudScaling(D=D)
        self.anchors_config = config.AnchorsConfig()
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.backbone = (models
                         .build_efficient_net_backbone(self.config.B, 
                                                       weights))
        self.backbone.trainable = not freeze_backbone

        if bidirectional:
            self.neck = models.BiFPN(self.config.Wbifpn, self.config.Dbifpn)
        else:
            self.neck = models.FPN(self.config.Wbifpn)

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

    def call(self, images, training: bool = True):
        """
        EfficientDet forward step

        Parameters
        ----------
        images: tf.Tensor
        training: bool
            Wether if model is training or it is in inference mode

        """
        features = self.backbone(images, training=training)
        
        # List of [BATCH, H, W, C]
        bifnp_features = self.neck(features, training=training)

        # List of [BATCH, A * 4]
        bboxes = [self.bb_head(bf, training=training) 
                  for bf in bifnp_features]

        # List of [BATCH, A * num_classes]
        class_scores = [self.class_head(bf, training=training) 
                        for bf in bifnp_features]

        # # [BATCH, H, W, A * 4]
        bboxes = tf.concat(bboxes, axis=1)
        class_scores = tf.concat(class_scores, axis=1)

        if training:
            return bboxes, class_scores

        else:
            # Create the anchors
            anchors = [g(f[0].shape)
                       for g, f in zip(self.anchors_gen, bifnp_features)]
            anchors = tf.concat(anchors, axis=0)
            
            # Tile anchors over batches, so they can be regressed
            batch_size = bboxes.shape[0]
            anchors = tf.tile(tf.expand_dims(anchors, 0), 
                              [batch_size, 1, 1])
            
            class_scores = tf.reshape(class_scores, 
                                      [batch_size, -1, self.num_classes])
            bboxes = tf.reshape(bboxes, 
                                [batch_size, -1, 4])

            boxes = utils.bndbox.regress_bndboxes(anchors, bboxes)
            boxes = utils.bndbox.clip_boxes(boxes, images.shape[1:3])
            boxes, labels, scores = utils.bndbox.nms(
                boxes, class_scores, score_threshold=self.score_threshold)
            # TODO: Pad output
            return boxes, labels, scores
    
    @staticmethod
    def from_pretrained(checkpoint_path: Union[Path, str], 
                        num_classes: int = None,
                        **kwargs) -> 'EfficientDet':
        """
        Instantiates an efficientdet model with pretreined weights.
        For transfer learning, the classifier head can be overwritten by
        a new randomly initialized one.

        Parameters
        ----------
        checkpoint_path: Union[Path, str]
            Checkpoint directory
        num_classes: int, default None
            If left to None the model will have the checkpoint head, 
            otherwise the head will be overwrite with a new randomly initialized
            classification head. Useful when training on your own dataset
        
        Returns
        ------- 
        EfficientDet
        """
        AVAILABLE_MODELS = {
            'D0-VOC': 'gs://ml-generic-purpose-tf-models/D0-VOC',
            'D0-VOC-FPN': 'gs://ml-generic-purpose-tf-models/D0-VOC-FPN'
        }

        # TODO: Make checkpoint path also a reference to a path.
        # For example: EfficientDet.from_pretrained('voc')        
        from efficientdet.checkpoint import load

        if str(checkpoint_path) in AVAILABLE_MODELS:
            checkpoint_path = AVAILABLE_MODELS[checkpoint_path]

        model, _ = load(checkpoint_path, **kwargs)

        if num_classes is not None:
            print('Loading a custom classification head...')
            model.num_classes = num_classes
            model.class_head = models.RetinaNetClassifier(
                model.config.Wbifpn, model.config.D, num_classes)

        return model
