from typing import Tuple

import tensorflow as tf


def load_image(im_path: str,
               im_size: Tuple[int, int]) -> tf.Tensor:
                
    # Reads the image and resizes it
    im = tf.io.read_file(im_path)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.convert_image_dtype(im, tf.float32)
    return tf.image.resize(im, im_size)