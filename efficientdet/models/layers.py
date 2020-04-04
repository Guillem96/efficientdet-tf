import tensorflow as tf


class Resize(tf.keras.Model):

    def __init__(self, features):
        super(Resize, self).__init__()
        self.antialiasing_conv = ConvBlock(separable=True,
                                           kernel_size=3, 
                                           padding='same')

    def call(self, images, target_dim=None, training=True):
        dims = target_dim[1: 3]
        x = tf.image.resize(images, dims, method='nearest')        
        x = self.antialiasing_conv(x, training=training)
        return x 


class ConvBlock(tf.keras.Model):

    def __init__(self, 
                 features: int = None, 
                 separable: bool = False, 
                 activation: str = None,
                 **kwargs):
        super(ConvBlock, self).__init__()

        if separable:
            self.conv = tf.keras.layers.DepthwiseConv2D(**kwargs)
        else:
            self.conv = tf.keras.layers.Conv2D(features, **kwargs)
        self.bn = tf.keras.layers.BatchNormalization()
        
        self.activation = None
        if activation is not None:
            self.activation = tf.keras.layers.Activation(activation)

    def call(self, x, training=True):
        x = self.bn(self.conv(x), training=training)

        if self.activation is not None:
            return self.activation(x)
        else: 
            return x
