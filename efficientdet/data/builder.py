from typing import Mapping, Sequence, Tuple

import tensorflow as tf

import efficientdet


AVAILABLE_FORMATS = {'VOC', 'labelme'}


def build_ds(format: str,
             annots_path: str,
             im_size: Tuple[int, int],
             class_names: Sequence[str] = [],
             data_augmentation: bool = True,
             **kwargs) -> Tuple[tf.data.Dataset, Mapping[str, int]]:
    """Build a tf.data.Dataset with an specific preprocessing
    
    This method builds a tf.data.Dataset with different preprocessing steps 
    depending on the specified format.

    Parameters
    ----------
    format: str, choice of [VOC, labelme]
        The format specifies the type of files and dataset structure

    im_size: Tuple[int, int]
        Size of the input images
    class_names: Sequence[str], default []
        Name of labels. When working with research datasets such as VOC
        there is no need to specify it, otherwise, with custom datasets 
        (labelme) it is mandatory.
    kwargs: 
        The extra arguments will be delegated to the correspoding dataset
        builder. Refer to different datasets builders parameters:
        - labelme -> efficientdet.data.labelme.build_dataset
        - VOC -> efficientdet.data.voc.build_datset
    """
    assert format in AVAILABLE_FORMATS, \
        'Format must be one of {}'.format(AVAILABLE_FORMATS) 
    
    if format == 'VOC':
        del kwargs['images_path']
        
        class2idx = efficientdet.data.voc.LABEL_2_IDX
        ds = efficientdet.data.voc.build_dataset(
            annots_path,
            data_augmentation=data_augmentation,
            im_input_size=im_size,
            **kwargs)

        return ds, class2idx

    elif format == 'labelme':
        assert len(class_names) > 0, 'You must specify class names'
        assert kwargs['images_path'] != '', 'Images base path missing'

        class2idx = {c: i 
            for i, c in enumerate(class_names)}

        ds = efficientdet.data.labelme.build_dataset(
            annots_path,
            class2idx=class2idx,
            im_input_size=im_size,
            data_augmentation=data_augmentation,
            **kwargs)
        
        return ds, class2idx