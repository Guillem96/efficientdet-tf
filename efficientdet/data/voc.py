"""
Utils module to work with VOC2007 dataset

Download the dataset from here: 
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
"""

from pathlib import Path
from functools import partial
from typing import Tuple, Sequence, Union

import tensorflow as tf

import xml.etree.ElementTree as ET

import efficientdet.utils.bndbox as bb_utils

IDX_2_LABEL = [
    'person',
    # Animals
    'dog',
    'bird',
    'cat',
    'cow',
    'horse',
    'sheep',
    # Vehicle
    'aeroplane',
    'bicycle',
    'boat',
    'bus',
    'car',
    'motorbike',
    'train',
    # Indoor
    'bottle',
    'chair',
    'diningtable',
    'pottedplant',
    'sofa',
    'tvmonitor',
]

LABEL_2_IDX = {l: i for i, l in enumerate(IDX_2_LABEL)}


def _read_voc_annot(annot_path: str) -> Tuple[Sequence[int], 
                                              Sequence[tf.Tensor]]:
    # Reads a voc annotation and returns
    # a list of tuples containing the ground 
    # truth boxes and its respective label
    root = ET.parse(annot_path).getroot()
    image_size = (int(root.findtext('size/height')), 
                  int(root.findtext('size/width')))

    boxes = root.findall('object')
    bbs = []
    labels = []

    for b in boxes:
        bb = b.find('bndbox')
        bb = (int(bb.findtext('xmin')), 
              int(bb.findtext('ymin')), 
              int(bb.findtext('xmax')), 
              int(bb.findtext('ymax')))
        bbs.append(bb)
        labels.append(LABEL_2_IDX[b.findtext('name')])

    bbs = tf.stack(bbs)
    bbs = bb_utils.normalize_bndboxes(bbs, image_size)

    return labels, bbs


def _load_image(im_path: str,
                im_size: Tuple[int, int]) -> tf.Tensor:
    # Reads the image and resizes it
    im = tf.io.read_file(im_path)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.convert_image_dtype(im, tf.float32)
    return tf.image.resize(im, im_size)


def _annot_gen(annot_file: Sequence[Path]):
    for f in annot_file:
        yield _read_voc_annot(str(f))
        

def build_dataset(dataset_path: Union[str, Path],
                  im_input_size: Tuple[int, int],
                  batch_size: int = 2) -> tf.data.Dataset:
    """
    Create model input pipeline using tensorflow datasets

    Parameters
    ----------
    dataset_path: Union[Path, str]
        Path to the voc2007 dataset. The dataset path should contain
        two subdirectories, one called images and another one called 
        annots
    im_input_size: Tuple[int, int]
        Model input size. Images will automatically be resized to this
        shape
    batch_size: int, default 2
        Training model batch size
    
    Examples
    --------
    
    >>> ds = build_dataset('data/VOC2007', im_input_size=(128, 128))

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
    dataset_path = Path(dataset_path)
    im_path = dataset_path / 'images'
    annot_path = dataset_path / 'annots'

    # List sorted annotation files
    annot_files = sorted(annot_path.glob('*.xml'))
    
    # Partially evaluate image loader to resize images
    # always with the same shape
    load_im = partial(_load_image, im_size=im_input_size)

    # We assume that tf datasets list files sorted when shuffle=False
    im_ds = (tf.data.Dataset.list_files(str(im_path / '*.jpg'), 
                                        shuffle=False)
             .map(load_im))
    annot_ds = (tf.data.Dataset
                .from_generator(generator=lambda: _annot_gen(annot_files), 
                                output_types=(tf.int32, tf.float32)))

    # Join both datasets
    ds = (tf.data.Dataset.zip((im_ds, annot_ds))
          .shuffle(128)
          .padded_batch(batch_size=batch_size,
                        padded_shapes=((*im_input_size, 3), 
                                       ((None,), (None, 4))),
                        padding_values=(0., (-1, 0.)))
          .repeat())
    
    return ds
