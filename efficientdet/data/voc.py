"""
Utils module to work with VOC2007 dataset

Download the dataset from here: 
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
"""

from pathlib import Path
from functools import partial
from typing import Tuple, Sequence, Union

import tensorflow as tf
import tensorflow_datasets as tfds

import xml.etree.ElementTree as ET

import efficientdet.utils.io as io_utils
import efficientdet.utils.bndbox as bb_utils
from .preprocess import normalize_image, augment


IDX_2_LABEL = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

LABEL_2_IDX = {l: i for i, l in enumerate(IDX_2_LABEL)}


def _tfds_to_efficientdet_fmt(o) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    image = o['image']
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    labels = o['objects']['label']
    labels = tf.cast(labels, tf.int32)
    
    bboxes = o['objects']['bbox']

    return image, labels, bboxes


def _permute_boxes(image, labels, bboxes):
    bboxes = bb_utils.to_tf_format(bboxes)
    return image, labels, bboxes


def _scale_boxes(image: tf.Tensor, labels: tf.Tensor, boxes: tf.Tensor):
    im_shape = tf.shape(image)
    w = tf.cast(im_shape[1], tf.float32)
    h = tf.cast(im_shape[0], tf.float32)

    x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
    x1 *= w
    x2 *= w
    y1 *= h
    y2 *= h
    
    return image, (labels, tf.concat([x1, y1, x2, y2], axis=1))


def build_dataset(dataset_path: Union[str, Path] = None,
                  im_input_size: Tuple[int, int] = None,
                  split: str = 'train',
                  shuffle: bool = True,
                  data_augmentation: bool = False) -> tf.data.Dataset:
    """
    Create model input pipeline using tensorflow datasets

    Parameters
    ----------
    dataset_path: Union[str, Path], default None
        Path to store downloaded VOC data. This may be useful to store the
        dataset ina google cloud bucket so later we can train the model
        using TPUs
    im_input_size: Tuple[int, int]
        Model input size. Images will automatically be resized to this
        shape
    data_augmentation: bool, default False
        Wether or not to apply data augmentation
    Examples
    --------
    
    >>> ds = build_dataset(im_input_size=(128, 128))

    >>> for images, (labels, bbs) in ds.take(1):
    ...   print(images.shape)
    ...   print(labels, bbs.shape)
    ...
    (2, 128, 128)
    ([[1, 0]
      [13, -1]], (2, 2, 4))

    Returns
    -------
    tf.data.Dataset

    """
    ds = tfds.load('voc', 
                   shuffle_files=shuffle,
                   split=split,
                   data_dir=dataset_path)

    normalize_image_fn = lambda im, *a: (normalize_image(im), *a)
    resize_image = lambda im, *a: (tf.image.resize(im, im_input_size), *a)
    ds = (ds
          .map(_tfds_to_efficientdet_fmt)
          .map(_permute_boxes)
          .map(normalize_image_fn)
          .map(resize_image)
          .map(_scale_boxes))

    if data_augmentation:
        ds = ds.map(augment)
    
    return ds
