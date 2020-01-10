import tensorflow as tf

EPSILON = 1e-5


class Resize(tf.keras.Model):

    def __init__(self, features, separable: bool = False):
        super(Resize, self).__init__()
        conv_cls = (tf.keras.layers.SeparableConv2D 
                    if separable else tf.keras.layers.Conv2D)
        # self.antialiasing_conv = conv_cls(features, 
        #                                   kernel_size=3, 
        #                                   padding='same')

    def call(self, images, target_dim=None):
        dims = target_dim[1: 3]
        x = tf.image.resize(images, dims) # Bilinear as default
        # x = self.antialiasing_conv(x)
        return x 

class FastFusion(tf.keras.Model):
    def __init__(self, size, features):
        super(FastFusion, self).__init__()
        w_init = tf.random_normal_initializer()

        self.size = size
        self.w = tf.Variable(initial_value=w_init((size,)))
        self.conv = tf.keras.layers.SeparableConv2D(features, kernel_size=1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.resize = Resize(features)

    def call(self, inputs):
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

        # (N_INPUTS, BATCH, H, W, C)
        weighted_sum = tf.map_fn(
            lambda i: w[i] * inputs[i], tf.range(self.size), dtype=tf.float32)
        
        # (BATCH, N_INPUTS, H, W, C)
        weighted_sum = tf.transpose(weighted_sum, [1, 0 , 2, 3, 4])
        
        # Sum weighted inputs
        # (BATCH, H, W, C)
        weighted_sum = tf.reduce_sum(weighted_sum, axis=1)
        fusioned_features = self.conv(weighted_sum / w_sum)
        fusioned_features = self.bn(fusioned_features)
        return self.relu(fusioned_features)


class BiFPNBlock(tf.keras.Model):

    def __init__(self, features):
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

    def call(self, features):
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
        P6_td = self.ff_6_td([P6, P7])
        P5_td = self.ff_5_td([P5, P6_td])
        P4_td = self.ff_4_td([P4, P5_td])

        # Compute out features maps
        P3_out = self.ff_3_out([P3, P4_td])
        P4_out = self.ff_4_out([P4, P4_td, P3_out])
        P5_out = self.ff_5_out([P5, P5_td, P4_out])
        P6_out = self.ff_6_out([P6, P6_td, P5_out])
        P7_out = self.ff_7_out([P7, P6_td])

        return P3_out, P4_out, P5_out, P6_out, P7_out


class BiFPN(tf.keras.Model):
    
    def __init__(self, 
                 features=64,
                 n_blocks=3):
        super(BiFPN, self).__init__()

                # One pixel-wise for each feature comming from the 
        # bottom-up path
        self.pixel_wise = [tf.keras.layers.Conv2D(features, kernel_size=1)
                            for _ in range(5)] 
                            
        self.blocks = [BiFPNBlock(features)
                       for i in range(n_blocks)]
    
    def call(self, inputs):
        
        # Each Pin has shape (BATCH, H, W, C)
        # We first reduce the channels using a pixel-wise conv
        features = [self.pixel_wise[i](inputs[i]) for i in range(len(inputs))]

        for block in self.blocks:
            features = block(features)
        return features
