import tensorflow as tf


class Resize(tf.keras.Model):

    def __init__(self, features, separable: bool = False):
        super(Resize, self).__init__()
        conv_cls = (tf.keras.layers.SeparableConv2D 
                    if separable else tf.keras.layers.Conv2D)
        self.antialiasing_conv = conv_cls(features, 
                                          kernel_size=3, 
                                          padding='same')

    def call(self, images, target_dim=None):
        dims = target_dim[1: 3]
        x = tf.image.resize(images, dims) # Bilinear as default
        x = self.antialiasing_conv(x)
        return x 