from pathlib import Path
from typing import Tuple, Union, Sequence, Mapping

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


def read_class_names(fname: Union[str, Path]) -> Tuple[Sequence[str], 
                                                       Mapping[str, int]]:
    """
    Given a file where each line is a class name, returns all classes and a
    mapping to class to index.

    Parameters
    ----------
    fname: Union[str, Path]
        Path to class names file
    
    Return
    ------
    Tuple
        The first element of the tuple is the set of classes, and the second
        element is the mapping from name to index
    """
    classes = Path(fname).read_text().split('\n')
    classes = [c.strip() for c in classes]
    class2idx = {c: i for i, c in enumerate(classes)}

    return classes, class2idx