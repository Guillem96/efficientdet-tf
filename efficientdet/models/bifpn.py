from typing import List

import tensorflow as tf

from . import layers
from efficientdet.utils import tf_utils


EPSILON = 1e-5


class FastFusion(tf.keras.layers.Layer):
    
    def __init__(self, 
                 features: int, 
                 prefix: str = '',
                 fusion_type: str = 'sum') -> None:
        super(FastFusion, self).__init__()

        self.features = features
        self.prefix = prefix
        self.fusion_type = fusion_type

        self.relu = tf.keras.layers.Activation('relu', name=prefix + 'relu')
        
        self.conv = layers.ConvBlock(features,
                                     separable=True,
                                     kernel_size=3, 
                                     strides=1, 
                                     padding='same', 
                                     activation='swish',
                                     prefix=prefix + 'conv_block/')

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        _, h, w, c = input_shape[0].as_list()

        self.size = len(input_shape)
        
        if self.fusion_type == 'fast_attn':
            self.w = self.add_weight(name=self.prefix + 'w',
                                     shape=(self.size,),
                                     initializer=tf.initializers.Ones(),
                                     trainable=True,
                                     dtype=tf.float32)

        elif self.fusion_type == 'fast_attn_channel':
            self.w = self.add_weight(name=self.prefix + 'w',
                                     shape=(self.size, c),
                                     initializer=tf.initializers.Ones(),
                                     trainable=True,
                                     dtype=tf.float32)
        elif self.fusion_type == 'sum':
            self.w = self.add_weight(name=self.prefix + 'w',
                                     shape=(self.size,),
                                     initializer=tf.initializers.Ones(),
                                     trainable=False,
                                     dtype=tf.float32)
        else:
            raise ValueError(f'Invalid fusion type {self.fusion_type}')

        self.resize = layers.Resample((h, w), self.features, 
                                      prefix=self.prefix + 'resize/')

    def call(self, 
             inputs: List[tf.Tensor], 
             training: bool = None) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs: List[tf.Tensor] of shape (BATCH, H, W, C)
        """
        # The last feature map has to be resized according to the
        # other inputs
        resampled_feature = self.resize(
            inputs[-1], training=training)

        resampled_features = inputs[:-1] + [resampled_feature]

        # wi has to be larger than 0 -> Apply ReLU
        w = self.relu(self.w)
        w_sum = EPSILON + tf.reduce_sum(w)

        # [INPUTS, BATCH, H, W, C]
        weighted_inputs = [
            tf.divide(tf.multiply(w[i], resampled_features[i]), w_sum)
            for i in range(self.size)]

        # Sum weighted inputs
        # (BATCH, H, W, C)
        weighted_sum = tf.add_n(weighted_inputs)
        return self.conv(weighted_sum, training=training)
        

class BiFPNBlock(tf.keras.layers.Layer):

    def __init__(self, features: int, prefix: str = '') -> None:
        super(BiFPNBlock, self).__init__()

        # Feature fusion for intermediate level
        # ff stands for Feature fusion
        # td refers to intermediate level
        self.ff_6_td = FastFusion(features, 
                                  prefix=prefix + 'ff_6_td_P6-P7_/')
        self.ff_5_td = FastFusion(features,
                                  prefix=prefix + 'ff_5_td_P5_P6_td/')
        self.ff_4_td = FastFusion(features, 
                                  prefix=prefix + 'ff_4_td_P4_P5_td/')

        # Feature fusion for output
        self.ff_7_out = FastFusion(features,
                                   prefix=prefix + 'ff_7_out_P7_P6_td/')
        self.ff_6_out = FastFusion(features,
                                   prefix=prefix + 'ff_6_out_P6_P6_td_P7_out/')
        self.ff_5_out = FastFusion(features,
                                   prefix=prefix + 'ff_5_out_P5_P5_td_P4_out/')
        self.ff_4_out = FastFusion(features,
                                   prefix=prefix + 'ff_4_out_P4_P4_td_P3_out/')
        self.ff_3_out = FastFusion(features,
                                   prefix=prefix + 'ff_3_out_P3_P4_td/')

    def call(self, 
             features: List[tf.Tensor], 
             training: bool = None) -> List[tf.Tensor]:
        """
        Computes the feature fusion of bottom-up features comming
        from the Backbone NN

        Parameters
        ----------
        features: List[tf.Tensor]
            Feature maps of each convolution stage of the
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
    
    def __init__(self, 
                 features: int = 64, 
                 n_blocks: int = 3, 
                 prefix: str = '') -> None:
        super(BiFPN, self).__init__()

        # One pixel-wise for each feature comming from the 
        # bottom-up path
        self.pixel_wise = [layers.ConvBlock(features, kernel_size=1, 
                                            prefix=prefix + f'pixel_wise_{i}/')
                            for i in range(3)] 

        self.gen_P6 = layers.ConvBlock(features, 
                                       kernel_size=3, 
                                       strides=2, 
                                       padding='same',
                                       prefix=prefix + 'gen_P6/')
        
        self.relu = tf.keras.layers.Activation('relu', name=prefix + 'relu')

        self.gen_P7 = layers.ConvBlock(features, 
                                       kernel_size=3, 
                                       strides=2, 
                                       padding='same',
                                       prefix=prefix + 'gen_P7/')

        self.blocks = [BiFPNBlock(features, prefix=prefix + f'block_{i}/') 
                                  for i in range(n_blocks)]

    def call(self, 
             inputs: List[tf.Tensor], 
             training: bool = None) -> List[tf.Tensor]:
        
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

