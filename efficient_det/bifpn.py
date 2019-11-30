import tensorflow as tf

EPSILON = 1e-5


class FastFusion(tf.keras.Model):
    def __init__(self, size, features):
        super(FastFusion, self).__init__()
        w_init = tf.random_normal_initializer()
        self.size = size
        self.w = tf.Variable(initial_value=w_init((size,)))
        self.conv = tf.keras.layers.SeparableConv2D(features, kernel_size=1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs: List[tf.Tensor] of shape (BATCH, H, W, C)
        """
        # wi has to be larger than 0 -> Apply ReLU
        w = self.relu(self.w)
        w_sum = EPSILON + tf.reduce_sum(w)
        weighted_sum = tf.map_fn(
            lambda i: w[i] * inputs[i], tf.range(self.size), dtype=tf.float32)
        weighted_sum = tf.reduce_sum(weighted_sum, axis=0)
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

    def _resize(self, im, dims):
        # im: [BATCH, H, W, C]
        dims = dims[1: 3]
        return tf.image.resize(im, dims) # Bilinear as default

    def call(self, inputs):
        """
        Computes the feature fusion of bottom-up features comming
        from the Backbone NN

        Parameters
        ----------
        inputs: List[tf.Tensor]
            Feature maps of each convolutional stage of the
            backbone neural network
        """
        # Each Pin has shape (BATCH, HEIGHT, WIDTH, CHANNELS)
        P3, P4, P5, P6, P7 = inputs

        # Compute the intermediate state
        # Note that P3 and P7 have no intermediate state
        P6_td = self.ff_6_td([P6, self._resize(P7, P6.shape)])
        P5_td = self.ff_5_td([P5, self._resize(P6_td, P5.shape)])
        P4_td = self.ff_4_td([P4, self._resize(P5_td, P4.shape)])

        # Compute out features maps
        P3_out = self.ff_3_out([P3,
                                self._resize(P4_td, P3.shape)])
        P4_out = self.ff_4_out([P4,
                                P4_td,
                                self._resize(P3_out, P4.shape)])
        P5_out = self.ff_5_out([P5,
                                P5_td,
                                self._resize(P4_out, P5.shape)])
        P6_out = self.ff_6_out([P6,
                                P6_td,
                                self._resize(P5_out, P6.shape)])
        P7_out = self.ff_7_out([P7,
                                self._resize(P6_td, P7.shape)])

        return P3_out, P4_out, P5_out, P6_out, P7_out


class BiFPN(tf.keras.Model):
    
    def __init__(self, 
                 features=64,
                 n_blocks=3):
        super(BiFPN, self).__init__()

        self.blocks = [BiFPNBlock(features)
                       for i in range(n_blocks)]
    
    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        return x
