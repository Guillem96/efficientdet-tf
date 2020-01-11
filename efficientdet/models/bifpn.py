import tensorflow as tf

from . import layers

EPSILON = 1e-5


class FastFusion(tf.keras.layers.Layer):
    def __init__(self, size: int, features: int):
        super(FastFusion, self).__init__()
        w_init = tf.keras.initializers.constant(1 / size)

        self.size = size
        self.w = tf.Variable(name='w', 
                             initial_value=w_init(shape=(size,)),
                             trainable=True)

        self.relu = tf.keras.layers.Activation('relu')

        self.conv = tf.keras.layers.SeparableConv2D(features, 
                                                    kernel_size=3, 
                                                    strides=1, 
                                                    padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.Activation('relu')

        self.resize = layers.Resize(features)

    def call(self, inputs, training=True):
        """
        Parameters
        ----------
        inputs: List[tf.Tensor] of shape (BATCH, H, W, C)
        """
        # The last feature map has to be resized according to the
        # other inputs
        inputs[-1] = self.resize(inputs[-1], inputs[0].shape)

        # wi has to be larger than 0 -> Apply ReLU
        w = self.relu(self.w)
        w_sum = EPSILON + tf.reduce_sum(w)

        # List of (BATCH, H, W, C)
        weighted_inputs = [w[i] * inputs[i] for i in range(self.size)]
        
        # Sum weighted inputs
        # (BATCH, H, W, C)
        weighted_sum = tf.reduce_sum(weighted_inputs, axis=0)
        fusioned_features = self.conv(weighted_sum / w_sum)
        fusioned_features = self.bn(fusioned_features, training=training)

        return self.relu_2(fusioned_features)


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

    def call(self, features, training=True):
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

        return P3_out, P4_out, P5_out, P6_out, P7_out


class BiFPN(tf.keras.Model):
    
    def __init__(self, 
                 features=64,
                 n_blocks=3):
        super(BiFPN, self).__init__()

        # One pixel-wise for each feature comming from the 
        # bottom-up path
        self.pixel_wise = [tf.keras.layers.Conv2D(features, kernel_size=1)
                            for _ in range(3)] 

        self.gen_P6 = tf.keras.layers.Conv2D(features, 
                                             kernel_size=3, 
                                             strides=2, 
                                             padding='same')
        
        self.relu = tf.keras.layers.Activation('relu')

        self.gen_P7 = tf.keras.layers.Conv2D(features, 
                                             kernel_size=3, 
                                             strides=2, 
                                             padding='same')

        self.blocks = [BiFPNBlock(features) for i in range(n_blocks)]
          
    
    def call(self, inputs, training=True):
        
        # Each Pin has shape (BATCH, H, W, C)
        # We first reduce the channels using a pixel-wise conv
        _, _, *C = inputs
        P3, P4, P5 = [self.pixel_wise[i](inputs[i]) for i in range(len(C))]
        P6 = self.gen_P6(C[-1])
        P7 = self.gen_P7(self.relu(P6))

        features = P3, P4, P5, P6, P7
        for block in self.blocks:
            features = block(features, training=training)

        return features
