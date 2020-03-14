from typing import Tuple
from pathlib import Path

import click

import tensorflow as tf

import efficientdet
import efficientdet.utils as utils
import efficientdet.engine as engine

huber_loss_fn = tf.losses.Huber(
    reduction=tf.losses.Reduction.SUM)


def loss_fn(y_true_clf: tf.Tensor, 
            y_pred_clf: tf.Tensor, 
            y_true_reg: tf.Tensor, 
            y_pred_reg: tf.Tensor) -> Tuple[float, float]:

    batch, n_anchors, n_classes = y_true_clf.shape

    anchors_states = y_true_clf[:, :, -1]
    not_ignore_idx = tf.where(anchors_states != -1)
    true_idx = tf.where(anchors_states == 1)
    normalizer = true_idx.shape[0]
    
    y_true_clf = tf.gather_nd(y_true_clf[:, :, :-1], not_ignore_idx)
    y_pred_clf = tf.gather_nd(y_pred_clf, not_ignore_idx)
    
    y_true_reg = tf.gather_nd(y_true_reg[:, :, :-1], true_idx)
    y_pred_reg = tf.gather_nd(y_pred_reg, true_idx)
    
    reg_loss = huber_loss_fn(y_true_reg, y_pred_reg)

    clf_loss = efficientdet.losses.focal_loss(y_true_clf, 
                                              y_pred_clf,
                                              reduction='sum')

    return reg_loss / normalizer, clf_loss / normalizer


def generate_anchors(anchors_config: efficientdet.config.AnchorsConfig,
                     im_shape: int) -> tf.Tensor:

    anchors_gen = [utils.anchors.AnchorGenerator(
            size=anchors_config.sizes[i - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.strides[i - 3]) 
            for i in range(3, 8)]

    shapes = [im_shape // (2 ** x) for x in range(3, 8)]

    anchors = [g((size, size, 3))
               for g, size in zip(anchors_gen, shapes)]

    return tf.concat(anchors, axis=0)


def train(**kwargs):
    save_checkpoint_dir = Path(kwargs['save_dir'])
    save_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model = efficientdet.models.EfficientDet(
        kwargs['n_classes'],
        D=kwargs['efficientdet'],
        bidirectional=kwargs['bidirectional'],
        freeze_backbone=kwargs['freeze_backbone'],
        weights='imagenet')
    
    if kwargs['checkpoint'] is not None:
        print('Loading checkpoint from {}...'.format(kwargs['checkpoint']))
        model.load_weights(kwargs['checkpoint'])

    ds, class2idx = efficientdet.data.build_ds(
        format=kwargs['format'],
        annots_path=kwargs['train_dataset'],
        images_path=kwargs['images_path'],
        im_size=(model.config.input_size,) * 2,
        class_names=kwargs['classes_names'].split(','),
        batch_size=kwargs['batch_size'])

    val_ds = None
    if kwargs['val_dataset']:
        val_ds, _ = efficientdet.data.build_ds(
            format=kwargs['format'],
            annots_path=kwargs['val_dataset'],
            images_path=kwargs['images_path'],
            class_names=kwargs['classes_names'].split(','),
            im_size=(model.config.input_size,) * 2,
            shuffle=False,
            batch_size=kwargs['batch_size'] // 2)

    anchors = generate_anchors(model.anchors_config,
                               model.config.input_size)
    
    optimizer = tf.optimizers.Adam(
        learning_rate=kwargs['learning_rate'],
        clipnorm=1)

    for epoch in range(kwargs['epochs']):

        engine.train_single_epoch(
            model=model,
            anchors=anchors,
            dataset=ds,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_classes=kwargs['n_classes'],
            epoch=epoch)

        if val_ds is not None:
            engine.evaluate(
                model=model,
                dataset=val_ds,
                class2idx=class2idx)
        
        model_type = 'bifpn' if kwargs['bidirectional'] else 'fpn'
        data_format = kwargs['format']
        fname = f'{model_type}_{data_format}_efficientdet_weights_{epoch}.tf'
        fname = save_checkpoint_dir / fname
        model.save_weights(str(fname))


@click.command()

# Neural network parameters
@click.option('--efficientdet', type=int, default=0,
              help='EfficientDet architecture. '
                   '{0, 1, 2, 3, 4, 5, 6, 7}')
@click.option('--bidirectional/--no-bidirectional', default=True,
              help='If bidirectional is set to false the NN will behave as '
                   'a "normal" retinanet, otherwise as EfficientDet')
@click.option('--freeze-backbone/--no-freeze-backbone', 
              default=False, help='Wether or not freeze EfficientNet backbone')

# Training parameters
@click.option('--epochs', type=int, default=20,
              help='Number of epochs to train the model')
@click.option('--batch-size', type=int, default=16,
              help='Dataset batch size')
@click.option('--learning-rate', type=float, default=1e-3,
              help='Optimizer learning rate. It is recommended to reduce it '
                   'in case backbone is not frozen')

# Data parameters
@click.option('--format', type=click.Choice(['VOC', 'labelme']),
              required=True, help='Dataset to use for training')
@click.option('--train-dataset', type=click.Path(file_okay=False, exists=True),
              required=True, help='Path to annotations and images')
@click.option('--val-dataset', default='', 
              type=click.Path(file_okay=False, exists=True),
              help='Path to validation annotations. If it is '
                   ' not set by the user, validation won\'t be performed')
@click.option('--images-path', type=click.Path(file_okay=False),
              required=True, default='',
              help='Base path to images. '
                   'Required when using labelme format')
@click.option('--n-classes', type=int, required=True,
              help='Number of important classes without '
                   'taking background into account')
@click.option('--classes-names', 
              default='', type=str, 
              help='Only required when format is labelme. '
                   'Name of classes separated using comma. '
                   'class1,class2,class3')

# Checkpointing parameters
@click.option('--checkpoint', help='Path to model checkpoint',
              type=click.Path(), default=None)
@click.option('--save-dir', help='Directory to save model weights',
              required=True, default='models', 
              type=click.Path(file_okay=False))
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    main()