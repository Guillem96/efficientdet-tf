from typing import Sequence

import tensorflow as tf

from . import layers
from efficientdet.utils import tf_utils


EPSILON = 1e-5


class FastFusion(tf.keras.layers.Layer):
    def __init__(self, size: int, features: int):
        super(FastFusion, self).__init__()

        self.size = size
        w_init = tf.keras.initializers.constant(1. / size)
        self.w = tf.Variable(name='w', 
                             initial_value=w_init(shape=(size,)),
                             trainable=True)
        self.relu = tf.keras.layers.Activation('relu')
        
        self.conv = layers.ConvBlock(features,
                                     separable=True,
                                     kernel_size=3, 
                                     strides=1, 
                                     padding='same', 
                                     activation='swish')
        self.resize = layers.Resize(features)

    def call(self, 
             inputs: Sequence[tf.Tensor], 
             training: bool = True) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs: List[tf.Tensor] of shape (BATCH, H, W, C)
        """
        # The last feature map has to be resized according to the
        # other inputs
        inputs[-1] = self.resize(
            inputs[-1], tf.shape(inputs[0]), training=training)

        # wi has to be larger than 0 -> Apply ReLU
        w = self.relu(self.w)
        w_sum = EPSILON + tf.reduce_sum(w, axis=0)

        # [INPUTS, BATCH, H, W, C]
        weighted_inputs = [w[i] * inputs[i] for i in range(self.size)]

        # Sum weighted inputs
        # (BATCH, H, W, C)
        weighted_sum = tf.reduce_sum(weighted_inputs, axis=0) / w_sum
        return self.conv(weighted_sum, training=training)
        

class BiFPNBlock(tf.keras.Model):

    def __init__(self, features: int):
        super(BiFPNBlock, self).__init__()

        # Feature fusion for intermediate level
        # ff stands for Feature fusion
        # td refers to intermediate level
        self.ff_6_td = FastFusion(2, features)
        self.ff_5_td = FastFusion(2, features)
        self.ff_4_td = FastFusion(2, features)

        # Feature fusion for output
        self.ff_7_out = FastFusion(2, features)
        self.ff_6_out = FastFusion(3, features)
        self.ff_5_out = FastFusion(3, features)
        self.ff_4_out = FastFusion(3, features)
        self.ff_3_out = FastFusion(2, features)

    def call(self, 
             features: Sequence[tf.Tensor], 
             training: bool = True) -> Sequence[tf.Tensor]:
        """
        Computes the feature fusion of bottom-up features comming
        from the Backbone NN

        Parameters
        ----------
        features: List[tf.Tensor]
            Feature maps of each convolutional stage of the
            backbone neural network
        """
        P3, P4, P5, P6, P7 = features

        # Compute the intermediate state
        # Note that P3 and P7 have no intermediate state
        P6_td = self.ff_6_td([P6, P7], training=training)
        P5_td = self.ff_5_td([P5, P6_td], training=training)
        P4_td = self.ff_4_td([P4, P5_td], training=training)

        # Compute out features maps
        P3_out = self.ff_3_out([P3, P4_td], training=training)
        P4_out = self.ff_4_out([P4, P4_td, P3_out], training=training)
        P5_out = self.ff_5_out([P5, P5_td, P4_out], training=training)
        P6_out = self.ff_6_out([P6, P6_td, P5_out], training=training)
        P7_out = self.ff_7_out([P7, P6_td], training=training)

        return [P3_out, P4_out, P5_out, P6_out, P7_out]


class BiFPN(tf.keras.Model):
    
    def __init__(self, features: int = 64, n_blocks: int = 3):
        super(BiFPN, self).__init__()

        # One pixel-wise for each feature comming from the 
        # bottom-up path
        self.pixel_wise = [layers.ConvBlock(features, kernel_size=1)
                            for _ in range(3)] 

        self.gen_P6 = layers.ConvBlock(features, 
                                       kernel_size=3, 
                                       strides=2, 
                                       padding='same')
        
        self.relu = tf.keras.layers.Activation('relu')

        self.gen_P7 = layers.ConvBlock(features, 
                                       kernel_size=3, 
                                       strides=2, 
                                       padding='same')

        self.blocks = [BiFPNBlock(features) for i in range(n_blocks)]

    def call(self, 
             inputs: Sequence[tf.Tensor], 
             training: bool = True) -> Sequence[tf.Tensor]:
        
        # Each Pin has shape (BATCH, H, W, C)
        # We first reduce the channels using a pixel-wise conv
        _, _, *C = inputs
        P3, P4, P5 = [self.pixel_wise[i](C[i], training=training) 
                      for i in range(len(C))]
        P6 = self.gen_P6(C[-1], training=training)
        P7 = self.gen_P7(self.relu(P6), training=training)

        features = [P3, P4, P5, P6, P7]
        features = tf_utils.call_cascade(self.blocks, 
                                         features, 
                                         training=training)
        return features

