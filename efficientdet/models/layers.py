import tensorflow as tf


class Resize(tf.keras.Model):

    def __init__(self, features: int):
        super(Resize, self).__init__()
        self.antialiasing_conv = ConvBlock(features,
                                           separable=True,
                                           kernel_size=3, 
                                           padding='same')

    def call(self, 
             images: tf.Tensor, 
             target_dim=None, 
             training: bool = True) -> tf.Tensor:
        dims = target_dim[1:3]
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
            self.conv = tf.keras.layers.SeparableConv2D(filters=features, 
                                                        **kwargs)
        else:
            self.conv = tf.keras.layers.Conv2D(features, **kwargs)
        self.bn = tf.keras.layers.BatchNormalization()
        
        if activation == 'swish':
            self.activation = tf.keras.layers.Activation(tf.nn.swish)
        elif activation is not None:
            self.activation = tf.keras.layers.Activation(activation)
        else:
            self.activation = tf.keras.layers.Activation('linear')

    def call(self, x, training=True):
        x = self.bn(self.conv(x), training=training)
        return self.activation(x)
