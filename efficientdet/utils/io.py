from typing import Tuple

import tensorflow as tf
from efficientdet.data import preprocess


def load_image(im_path: str,
               im_size: Tuple[int, int],
               normalize_image: bool = False) -> tf.Tensor:
                
    # Reads the image and resizes it
    im = tf.io.read_file(im_path)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.convert_image_dtype(im, tf.float32)
    if normalize_image:
        im = preprocess.normalize_image(im)
        
    return tf.image.resize(im, im_size)