import tensorflow as tf

from . import layers


class Merge(tf.keras.Model):

    def __init__(self, features: int = 64):
        super(Merge, self).__init__()

        self.resize = layers.Resize(features)

    def call(self, features: tf.Tensor):
        a, b = features
        b = self.resize(b, a.shape)
        return a + b


class FPN(tf.keras.Model):

    def __init__(self, features: int = 64):
        super(FPN, self).__init__()

        self.features = features
        self.pointwises = [tf.keras.layers.Conv2D(features, 
                                                  kernel_size=1, 
                                                  strides=1, 
                                                  padding='same')
                           for _ in range(3)]

        self.resize_P5 = layers.Resize(features)
        self.resize_P4 = layers.Resize(features)
        
        self.gen_P6 = tf.keras.layers.Conv2D(features, 
                                             kernel_size=3, 
                                             strides=2, 
                                             padding='same')
        
        self.relu = tf.keras.layers.Activation('relu')

        self.gen_P7 = tf.keras.layers.Conv2D(features, 
                                             kernel_size=3, 
                                             strides=2, 
                                             padding='same')
        
    def call(self, features: tf.Tensor):
        _, _, *C = features

        P3, P4, P5 = [self.pointwises[i](C[i]) for i in range(3)]
        P5_upsampled = self.resize_P5(P5, P4.shape)
        P4 = P5_upsampled + P4

        P4_upsampled = self.resize_P4(P4, P3.shape)
        P3 = P4_upsampled + P3
        
        P6 = self.gen_P6(C[-1])
        P7 = self.gen_P7(self.relu(P6))

        return P3, P4, P5, P6, P7
