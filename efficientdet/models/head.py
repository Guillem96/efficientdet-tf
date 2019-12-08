import tensorflow as tf


class RetinaNetBBPredictor(tf.keras.Model):

    def __init__(self, 
                 width: int, 
                 depth: int, 
                 num_anchors: int = 9):
        super(RetinaNetBBPredictor, self).__init__()
        self.num_anchors = num_anchors
        self.feature_extractors = [
            tf.keras.layers.Conv2D(width, 
                                   kernel_size=3,
                                   activation='relu',
                                   padding='same')
            for _ in range(depth)]
        self.bb_regressor = tf.keras.layers.Conv2D(num_anchors * 4,
                                                   kernel_size=3,
                                                   padding='same')

    def call(self, features):
        x = features
        for fe in self.feature_extractors:
            x = fe(x)
        return tf.reshape(self.bb_regressor(x), [x.shape[0], -1, self.num_anchors, 4])


class RetinaNetClassifier(tf.keras.Model):

    def __init__(self, 
                 width: int, 
                 depth: int,
                 num_classes: int, 
                 num_anchors: int = 9):
        super(RetinaNetClassifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.feature_extractors = [
            tf.keras.layers.Conv2D(width, 
                                   kernel_size=3,
                                   activation='relu',
                                   padding='same')
            for _ in range(depth)]

        self.cls_score = tf.keras.layers.Conv2D(num_anchors * num_classes,
                                                kernel_size=3,
                                                activation='sigmoid',
                                                padding='same')

    def call(self, features):
        x = features
        for fe in self.feature_extractors:
            x = fe(x)
        return tf.reshape(self.cls_score(x), [x.shape[0], -1, self.num_anchors, self.num_classes])

