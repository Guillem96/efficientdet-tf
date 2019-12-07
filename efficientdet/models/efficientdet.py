import tensorflow as tf

from efficientdet import models
from efficientdet import config


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
        self.config = config.EfficientDetCompudScalig(D=D)
        self.backbone = (models
                         .build_efficient_net_backbone(self.config.B, 
                                                       weights))

        self.bifp = models.BiFPN(self.config.Wbifpn, self.config.Dbifpn)

        self.class_head = models.RetinaNetClassifier(self.config.Wbifpn,
                                                     self.config.Dclass,
                                                     num_classes)
        self.bb_head = models.RetinaNetBBPredictor(self.config.Wbifpn,
                                                   self.config.Dclass)


    def call(self, images):
        features = self.backbone(images)

        # List of [BATCH, H, W, C]
        bifnp_features = self.bifp(features)

        # List of [BATCH, A * 4]
        bboxes = [self.bb_head(bf) for bf in bifnp_features]

        # List of [BATCH, A * num_classes]
        class_scores = [self.class_head(bf) for bf in bifnp_features]

        # # [BATCH, H, W, A * 4]
        bboxes = tf.concat(bboxes, axis=1)
        class_scores = tf.concat(class_scores, axis=1)

        return bboxes, class_scores