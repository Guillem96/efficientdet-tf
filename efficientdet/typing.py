from typing import Tuple

import tensorflow as tf


Annotation = Tuple[tf.Tensor, tf.Tensor]
ObjectDetectionInstance = Tuple[tf.Tensor, Annotation]