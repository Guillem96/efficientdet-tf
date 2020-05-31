from typing import Any, Tuple, Sequence

import tensorflow as tf

from efficientdet.config import AnchorsConfig
from efficientdet.utils import bndbox, anchors


class Resample(tf.keras.layers.Layer):

    def __init__(self, 
                 shape: Tuple[int, int], 
                 features: int, prefix: str = '') -> None:
        super(Resample, self).__init__()
        self.target_h = shape[0]
        self.target_w = shape[1]
        self.features = features
        self.prefix = prefix

        self.antialiasing_conv = ConvBlock(features,
                                           separable=False,
                                           kernel_size=1, 
                                           padding='same',
                                           prefix=prefix + 'pixel_wise/')

    def build(self, input_shape: Tuple[int, int, int, int]) -> None:
        _, self.height, self.width, self.num_channels = input_shape.as_list()
        stride_h = (self.height - 1) // self.target_h + 1
        stride_w = (self.width - 1) // self.target_w + 1
        
        self.max_pool = tf.keras.layers.MaxPooling2D(
            pool_size=(stride_h + 1, stride_w + 1),
            strides=(stride_h, stride_w),
            padding='same',
            name=self.prefix + 'pool')
        
        h_scale = self.target_h // self.height
        w_scale = self.target_w // self.width

        self.upsample = tf.keras.layers.UpSampling2D((h_scale, w_scale))

    def _maybe_apply_1x1(self, 
                         images: tf.Tensor, 
                         training: bool = None) -> tf.Tensor:
        if self.num_channels != self.features:
            images = self.antialiasing_conv(images, training=training)
        return images

    def call(self, 
             images: tf.Tensor, 
             training: bool = None) -> tf.Tensor:

        if self.height > self.target_h and self.width > self.target_w:
            x = self.max_pool(images)
        elif self.height <= self.target_h and self.width <= self.target_w:
            x = self.upsample(images)
        else:
            input_shape = (self.height, self.width, self.num_channels)
            target_shape = (self.target_h, self.target_w, self.features)

            raise ValueError(f'Inconsistent image shapes. Input {input_shape} '
                             f'and target {target_shape}')

        x = self._maybe_apply_1x1(x, training=training)

        return x 
    
    def compute_output_shape(self, 
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape(
            [None, self.target_h, self.target_w, self.features])


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, 
                 features: int = None, 
                 separable: bool = False, 
                 activation: str = None,
                 prefix: str = '',
                 **kwargs: Any) -> None:
        super(ConvBlock, self).__init__()

        if separable:
            name = prefix + 'separable_conv'
            self.conv = tf.keras.layers.SeparableConv2D(filters=features,
                                                        name=name,
                                                        **kwargs)
        else:
            name = prefix + 'conv'
            self.conv = tf.keras.layers.Conv2D(features, name=name, **kwargs)

        self.bn = tf.keras.layers.BatchNormalization(name=prefix + 'bn')
        
        if activation == 'swish':
            self.activation = tf.keras.layers.Activation(
                tf.nn.swish, name=prefix + 'swish')
        elif activation is not None:
            self.activation = tf.keras.layers.Activation(activation,
                name=prefix + activation)
        else:
            self.activation = tf.keras.layers.Activation('linear',
                name=prefix + 'linear')

    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        x = self.bn(self.conv(x), training=training)
        return self.activation(x)

    def compute_output_shape(self, 
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        return self.conv.compute_output_shape(input_shape)


class FilterDetections(object):

    def __init__(self, 
                 anchors_config: AnchorsConfig,
                 score_threshold: float):

        self.score_threshold = score_threshold
        self.anchors_gen = [anchors.AnchorGenerator(
            size=anchors_config.sizes[i - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.strides[i - 3]
        ) for i in range(3, 8)] # 3 to 7 pyramid levels


        # Accelerate calls
        self.regress_boxes = tf.function(
            bndbox.regress_bndboxes, input_signature=[
                tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32)])

        self.clip_boxes = tf.function(
            bndbox.clip_boxes, input_signature=[
                tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                tf.TensorSpec(shape=None)])
    
    def __call__(self, 
                 images: tf.Tensor, 
                 regressors: tf.Tensor, 
                 class_scores: tf.Tensor) -> Tuple[Sequence[tf.Tensor], 
                                                   Sequence[tf.Tensor], 
                                                   Sequence[tf.Tensor]]:

        im_shape = tf.shape(images)
        batch_size, h, w = im_shape[0], im_shape[1], im_shape[2]

        # Create the anchors
        shapes = [w // (2 ** x) for x in range(3, 8)]
        anchors = [g((size, size, 3))
                   for g, size in zip(self.anchors_gen, shapes)]
        anchors = tf.concat(anchors, axis=0)
        
        # Tile anchors over batches, so they can be regressed
        anchors = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])

        # Regress anchors and clip in case they go outside of the image
        boxes = self.regress_boxes(anchors, regressors)
        boxes = self.clip_boxes(boxes, [h, w])

        # Suppress overlapping detections
        boxes, labels, scores = bndbox.nms(
            boxes, class_scores, score_threshold=self.score_threshold)

        # TODO: Pad output
        return boxes, labels, scores