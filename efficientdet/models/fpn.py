from typing import Sequence

import tensorflow as tf

from . import layers


class Merge(tf.keras.layers.Layer):

    def __init__(self, features: int = 64) -> None:
        super(Merge, self).__init__()
        
        self.resize = layers.Resample(features)
        self.conv = layers.ConvBlock(features,
                                     kernel_size=3,
                                     strides=1,
                                     separable=True,
                                     activation='swish',
                                     padding='same')
                                     
    def call(self, features: tf.Tensor, training: bool = None) -> tf.Tensor:
        a, b = features
        b = self.resize(b, a.shape, training=training)
        return self.conv(a + b)


class FPN(tf.keras.Model):

    def __init__(self, features: int = 64) -> None:
        super(FPN, self).__init__()

        self.features = features
        self.pointwises = [tf.keras.layers.Conv2D(features, 
                                                  kernel_size=1, 
                                                  strides=1, 
                                                  padding='same')
                           for _ in range(3)]

        self.merge1 = Merge(features)
        self.merge2 = Merge(features)
        
        self.gen_P6 = tf.keras.layers.Conv2D(features, 
                                             kernel_size=3, 
                                             strides=2, 
                                             padding='same')
        
        self.relu = tf.keras.layers.Activation('relu')

        self.gen_P7 = tf.keras.layers.Conv2D(features, 
                                             kernel_size=3, 
                                             strides=2, 
                                             padding='same')
        
    def call(self, 
             features: tf.Tensor, 
             training: bool = None) -> Sequence[tf.Tensor]:
        _, _, *C = features

        P3, P4, P5 = [self.pointwises[i](C[i]) for i in range(3)]

        P4 = self.merge1([P4, P5], training=training)
        P3 = self.merge2([P3, P4], training=training)
        
        P6 = self.gen_P6(C[-1])
        P7 = self.gen_P7(self.relu(P6))

        return [P3, P4, P5, P6, P7]
