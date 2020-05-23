"""
Utils module to work with VOC2007 dataset
Download the dataset from here: 
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
"""

from pathlib import Path
from functools import partial
from typing import Tuple, Sequence, Union, Iterator

import tensorflow as tf

import xml.etree.ElementTree as ET

import efficientdet.utils.io as io_utils
import efficientdet.utils.bndbox as bb_utils
from .preprocess import augment


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
    image_size = (int(root.findtext('size/height')), # type: ignore[arg-type]
                  int(root.findtext('size/width')))  # type: ignore[arg-type]

    boxes = root.findall('object')
    bbs = []
    labels = []

    for b in boxes:
        bb = b.find('bndbox')
        bb = (int(bb.findtext('xmin')), # type: ignore
              int(bb.findtext('ymin')), # type: ignore
              int(bb.findtext('xmax')), # type: ignore
              int(bb.findtext('ymax'))) # type: ignore
        bbs.append(bb)
        labels.append(LABEL_2_IDX[b.findtext('name')]) # type: ignore

    bbs = tf.stack(bbs)
    bbs = bb_utils.normalize_bndboxes(bbs, image_size)

    return labels, bbs


def _annot_gen(
        annot_file: Sequence[Path]) -> Iterator[Tuple[Sequence[int], 
                                                Sequence[tf.Tensor]]]:
    for f in annot_file:
        yield _read_voc_annot(str(f))


def _scale_boxes(labels: tf.Tensor, boxes: tf.Tensor, 
                 to_size: Tuple[int, int]) -> Tuple[Sequence[int], 
                                                    Sequence[tf.Tensor]]:
    h = to_size[0]
    w = to_size[1]

    x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
    x1 *= w
    x2 *= w
    y1 *= h
    y2 *= h
    
    return labels, tf.concat([x1, y1, x2, y2], axis=1)


def build_dataset(dataset_path: Union[str, Path],
                  im_input_size: Tuple[int, int],
                  shuffle: bool = True,
                  data_augmentation: bool = False) -> tf.data.Dataset:
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
    data_augmentation: bool, default False
        Wether or not to apply data augmentation
    Examples
    --------
    
    >>> ds = build_dataset('data/VOC2007', im_input_size=(128, 128))
    >>> for image, (labels, bbs) in ds.take(1):
    ...   print(image.shape)
    ...   print(labels, bbs.shape)
    ...
    (128, 128, 3)
    ([1, 13], (2, 4))
    Returns
    -------
    tf.data.Dataset
    """
    dataset_path = Path(dataset_path)
    im_path = dataset_path / 'JPEGImages'
    annot_path = dataset_path / 'Annotations'

    # List sorted annotation files
    annot_files = sorted(annot_path.glob('*.xml'))
    
    # Partially evaluate image loader to resize images
    # always with the same shape and normalize them
    load_im = lambda im, annots: (io_utils.load_image(im, im_input_size, 
                                                      normalize_image=True),
                                  annots)
    scale_boxes = partial(_scale_boxes, to_size=im_input_size)

    # We assume that tf datasets list files sorted when shuffle=False
    im_ds = tf.data.Dataset.list_files(str(im_path / '*.jpg'), shuffle=False)
    annot_ds = (tf.data.Dataset
                .from_generator(generator=lambda: _annot_gen(annot_files), 
                                output_types=(tf.int32, tf.float32))
                .map(scale_boxes))

    # Join both datasets
    ds = tf.data.Dataset.zip((im_ds, annot_ds))

    # Shuffle before loading the images
    if shuffle:
        ds = ds.shuffle(1024)

    ds = ds.map(load_im)

    if data_augmentation:
        ds = ds.map(augment)
        ds = ds.filter(lambda im, a: tf.greater(tf.shape(a[0])[0], 0))

    return ds
