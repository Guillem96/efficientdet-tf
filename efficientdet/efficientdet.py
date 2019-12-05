import tensorflow as tf

import efficientdet.head as head
import efficientdet.bifpn as bifpn
import efficientdet.config as config
import efficientdet.backbone as backbone


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
        self.config = config.EfficientDetCompudScalig(D)
        self.backbone = (backbone
                         .build_efficient_net_backbone(self.config.B, 
                                                       weights))

        self.bifp = bifpn.BiFPN(self.config.Wbipn, self.config.Dbifpn)

        self.class_head = head.RetinaNetClassifier(self.config.Wbipn,
                                                   self.config.Dclass)
        self.bb_head = head.RetinaBBClassifier(self.config.Wbipn,
                                               self.config.Dclass)


    def call(self, images):
        features = self.backbone(images)

        # List of [BATCH, H, W, C]
        bifnp_features = self.bifp(features)

        # List of [BATCH, H, W, A * 4]
        bboxes = tf.map_fn(self.bb_head, features)
        # List of [BATCH, H, W, A * num_classes]
        class_scores = tf.map_fn(self.class_head, features)

        # [BATCH, H, W, A * 4]
        bboxes = tf.concat(bboxes, axis=1)
        class_scores = tf.concat(class_scores, axis=1)