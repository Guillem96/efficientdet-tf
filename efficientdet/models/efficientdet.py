import json
from pathlib import Path
from typing import List, Union

import tensorflow as tf

import efficientdet.config as config
from efficientdet.utils import anchors, bndbox

from efficientdet import models


_AVAILABLE_WEIGHTS = {None, 'imagenet', 'D0-VOC'}
_WEIGHTS_PATHS = {
    'D0-VOC': 'gs://ml-generic-purpose-tf-models/D0-VOC',
    # 'D0-VOC-FPN': 'gs://ml-generic-purpose-tf-models/D0-VOC-FPN'
}


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
        on imagenet. If set to None, the backbone and the bifpn will be random
        initialized. If set to other value, both the backbone and bifpn
        will be initialized with pretrained weights
    """
    def __init__(self, 
                 num_classes: int = None,
                 D : int = 0, 
                 bidirectional: bool = True,
                 freeze_backbone: bool = False,
                 score_threshold: float = .1,
                 weights : str = 'imagenet',
                 custom_head_classifier: bool = False):
                 
        super(EfficientDet, self).__init__()

        # Check arguments coherency 
        if custom_head_classifier is True and num_classes is None:
            raise ValueError('If include_top is False, you must specify '
                             'the num_classes')
        
        if weights not in _AVAILABLE_WEIGHTS:
            raise ValueError(f'Weights {weights} not available.\n'
                             f'The available weights are '
                             f'{list(_AVAILABLE_WEIGHTS)}')
        
        if ((weights is 'imagenet' or weights is None) 
            and custom_head_classifier):
            raise ValueError('Custom Head does not make sense when '
                             'training the model from scratch. '
                             'Set custom_head_classifier to False or specify '
                             'other weights.')

        # If weights related to efficientdet are set,
        # update the model hyperparameters according to the checkpoint,
        # but printing a warning
        if weights != 'imagenet' and weights is not None:
            from efficientdet.utils.checkpoint import download_folder
            checkpoint_path = _WEIGHTS_PATHS[weights]
            save_dir = download_folder(checkpoint_path)

            print(f'Since weights {weights} are being loaded, '
                  f'the settings regarding to efficientdet are going to be '
                  f'overrided with the ones defined in the weights checkpoint')
            
            params = json.load((save_dir/'hp.json').open())

            # If num_classes is specified it must be the same as in the 
            # weights checkpoint except if the custom head classfier is set
            # to true
            if (num_classes is not None and not custom_head_classifier and
                num_classes != params['num_classes']):
                raise ValueError(f'Weights {weights} num classes are different'
                                  'from num_classes argument, please leave it '
                                  ' as None or specify the correct classes')
            
            bidirectional = params['bidirectional']
            D = params['efficientdet']

        # Declare the model architecture
        self.config = config.EfficientDetCompudScaling(D=D)
        self.anchors_config = config.AnchorsConfig()
        
        # Setup efficientnet backbone
        backbone_weights = 'imagenet' if weights == 'imagenet' else None
        self.backbone = (models
                         .build_efficient_net_backbone(
                             self.config.B, backbone_weights))
        for l in self.backbone.layers:
            l.trainable = not freeze_backbone
        self.backbone.trainable = not freeze_backbone
        
        # Setup the feature extractor neck
        if bidirectional:
            self.neck = models.BiFPN(self.config.Wbifpn, self.config.Dbifpn)
        else:
            self.neck = models.FPN(self.config.Wbifpn)

        # Setup the heads
        self.num_classes = num_classes
        self.class_head = models.RetinaNetClassifier(self.config.Wbifpn,
                                                     self.config.Dclass,
                                                     num_classes)
        self.bb_head = models.RetinaNetBBPredictor(self.config.Wbifpn,
                                                   self.config.Dclass)
        
        # Inference variables, won't be used during training
        self.anchors_gen = [anchors.AnchorGenerator(
            size=self.anchors_config.sizes[i - 3],
            aspect_ratios=self.anchors_config.ratios,
            stride=self.anchors_config.strides[i - 3]
        ) for i in range(3, 8)] # 3 to 7 pyramid levels

        self.score_threshold = score_threshold

        # Load the weights if needed
        if weights is not None and weights != 'imagenet':
            self.load_weights(str(save_dir / 'model.tf'))
            
            # Append a custom classifier
            if custom_head_classifier:
                self.class_head = models.RetinaNetClassifier(self.config.Wbifpn,
                                                             self.config.Dclass,
                                                             num_classes)

    def call(self, images: tf.Tensor, training: bool = True):
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

        # [BATCH, H, W, A * 4]
        bboxes = tf.concat(bboxes, axis=1)
        class_scores = tf.concat(class_scores, axis=1)

        if training:
            return bboxes, class_scores

        else:
            im_shape = tf.shape(images)
            batch_size, h, w = im_shape[0], im_shape[1], im_shape[2]
            
            # Create the anchors
            anchors = [g(tf.shape(f[0]))
                       for g, f in zip(self.anchors_gen, bifnp_features)]
            anchors = tf.concat(anchors, axis=0)
            
            # Tile anchors over batches, so they can be regressed
            anchors = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])
            
            class_scores = tf.reshape(class_scores, 
                                      [batch_size, -1, self.num_classes])
            bboxes = tf.reshape(bboxes, 
                                [batch_size, -1, 4])

            boxes = bndbox.regress_bndboxes(anchors, bboxes)
            boxes = bndbox.clip_boxes(boxes, [h, w])
            boxes, labels, scores = bndbox.nms(
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
        from efficientdet.utils.checkpoint import load

        if str(checkpoint_path) in _AVAILABLE_WEIGHTS:
            checkpoint_path = _WEIGHTS_PATHS[checkpoint_path]

        model, _ = load(checkpoint_path, **kwargs)

        if num_classes is not None:
            print('Loading a custom classification head...')
            model.num_classes = num_classes
            model.class_head = models.RetinaNetClassifier(
                model.config.Wbifpn, model.config.D, num_classes)

        return model
