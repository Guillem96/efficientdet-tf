from typing import Tuple
from pathlib import Path

import click

import tensorflow as tf

import efficientdet
import efficientdet.utils as utils
import efficientdet.engine as engine


def loss_fn(y_true_clf: tf.Tensor, 
            y_pred_clf: tf.Tensor, 
            y_true_reg: tf.Tensor, 
            y_pred_reg: tf.Tensor) -> Tuple[float, float]:

    batch, n_anchors, n_classes = y_true_clf.shape

    y_true_clf = tf.reshape(y_true_clf, [-1, n_classes])
    y_pred_clf = tf.reshape(y_pred_clf, [-1, n_classes - 1])
    
    y_true_reg = tf.reshape(y_true_reg, [-1, 5])
    y_pred_reg = tf.reshape(y_pred_reg, [-1, 4])
    
    regress_mask = y_true_clf[:, -1] == 1
    clf_mask = y_true_clf[:, -1] != -1

    reg_loss = efficientdet.losses.huber_loss(y_true_reg[:, :-1], 
                                              y_pred_reg,
                                              reduction='none')
    # No regress non-overlapping boxes
    reg_loss = tf.boolean_mask(reg_loss, regress_mask)
    reg_loss = tf.reduce_mean(reg_loss)

    clf_loss = efficientdet.losses.focal_loss(
        tf.boolean_mask(y_true_clf[:, :-1], clf_mask), 
        tf.boolean_mask(y_pred_clf, clf_mask),
        reduction='mean')
    
    return reg_loss, clf_loss


def generate_anchors(anchors_config: efficientdet.config.AnchorsConfig,
                     im_shape: int) -> tf.Tensor:

    anchors_gen = [utils.anchors.AnchorGenerator(
            size=anchors_config.sizes[i - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.strides[i - 3]) 
            for i in range(3, 8)]
    
    shapes = [im_shape // (2 ** (x - 2)) for x in range(3, 8)]

    anchors = [g((size, size, 3))
               for g, size in zip(anchors_gen, shapes)]

    return tf.concat(anchors, axis=0)


def train(**kwargs):
    save_checkpoint_dir = Path(kwargs['save_dir'])
    save_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model = efficientdet.models.EfficientDet(
        kwargs['n_classes'],
        D=kwargs['efficientdet'],
        freeze_backbone=kwargs['freeze_backbone'],
        weights='imagenet')
    
    if kwargs['checkpoint'] is not None:
        print('Loading checkpoint from {}...'.format(kwargs['checkpoint']))
        model.load_weights(kwargs['checkpoint'])

    if kwargs['format'] == 'VOC':
        ds = efficientdet.data.voc.build_dataset(
            kwargs['train_dataset'],
            batch_size=kwargs['batch_size'],
            im_input_size=(model.config.input_size,) * 2)

    anchors = generate_anchors(model.anchors_config, 
                               model.config.input_size)
    
    optimizer = tf.optimizers.Adam(
        learning_rate=kwargs['learning_rate'],
        clipvalue=1.0)

    for epoch in range(kwargs['epochs']):

        engine.train_single_epoch(
            model=model,
            anchors=anchors,
            dataset=ds,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_classes=kwargs['n_classes'],
            epoch=epoch)

        # TODO: Validate
        fname = save_checkpoint_dir / f'efficientdet_weights_{epoch}.tf'
        model.save_weights(str(fname))
        

@click.command()

# Neural network parameters
@click.option('--efficientdet', type=int, default=0,
              help='EfficientDet architecture. '
                   '{0, 1, 2, 3, 4, 5, 6, 7}')
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
@click.option('--format', type=click.Choice(['VOC']),
              default='VOC', help='Dataset to use for training')
@click.option('--train-dataset', type=click.Path(file_okay=False, exists=True),
              required=True, help='Path to annotations and images')
@click.option('--n-classes', type=int, required=True,
              help='Number of important classes without '
                   'taking background into account')

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