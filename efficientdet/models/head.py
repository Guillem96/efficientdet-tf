import tensorflow as tf
import math

from . import layers
from efficientdet.utils import tf_utils


class RetinaNetBBPredictor(tf.keras.Model):

    def __init__(self, 
                 width: int, 
                 depth: int, 
                 num_anchors: int = 9):
        super(RetinaNetBBPredictor, self).__init__()
        self.num_anchors = num_anchors
        self.feature_extractors = [
            layers.ConvBlock(width, 
                             kernel_size=3,
                             activation='swish',
                             padding='same')
            for _ in range(depth)]

        self.bb_regressor = tf.keras.layers.Conv2D(num_anchors * 4,
                                                   kernel_size=3,
                                                   padding='same')

    def call(self, features, training=True):
        batch_size = tf.shape(features)[0]

        x = tf_utils.call_cascade(
            self.feature_extractors, features, training=training)
        return tf.reshape(self.bb_regressor(x), [batch_size, -1, 4])


class RetinaNetClassifier(tf.keras.Model):

    def __init__(self, 
                 width: int, 
                 depth: int,
                 num_classes: int, 
                 num_anchors: int = 9):
        super(RetinaNetClassifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.feature_extractors = [
            layers.ConvBlock(width, 
                             kernel_size=3,
                             activation='swish',
                             padding='same')
            for _ in range(depth)]
        
        prob = 0.01
        w_init = tf.constant_initializer(-math.log((1 - prob) / prob))
        self.cls_score = tf.keras.layers.Conv2D(num_anchors * num_classes,
                                                kernel_size=3,
                                                activation='sigmoid',
                                                padding='same',
                                                bias_initializer=w_init)

    def call(self, features, training=True):
        batch_size = tf.shape(features)[0]
        x = tf_utils.call_cascade(
            self.feature_extractors, features, training=training)
        return tf.reshape(self.cls_score(x), 
                          [batch_size, -1, self.num_classes])

