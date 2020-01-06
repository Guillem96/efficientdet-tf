from typing import Tuple

import click

import tensorflow as tf

import efficientdet


def loss_fn(y_true_clf: tf.Tensor, 
            y_pred_clf: tf.Tensor, 
            y_true_reg: tf.Tensor, 
            y_pred_reg: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

    reg_loss = efficientdet.losses.huber_loss(y_true_reg, 
                                              y_pred_reg,
                                              reduction='mean')

    clf_loss = efficientdet.losses.focal_loss(y_true_clf, 
                                              y_pred_clf, 
                                              reduction='mean')

    return reg_loss, clf_loss


def train(**kwargs):

    model = efficientdet.models.EfficientDet(
        kwargs['n_classes'],
        D=kwargs['D'],
        weights='imagenet' if not kwargs['freeze_backbone'] else None)

    if kwargs['train_dataset'] == 'VOC':
        ds = efficientdet.data.voc.build_dataset(
            'test/data/VOC2007',
            batch_size=kwargs['batch_size'],
            im_input_size=model.config.input_size)
    
    optimizer = tf.optimizers.Adam(
        learning_rate=kwargs['learning_rate'])

    for epoch in range(kwargs['epochs']):
        
        efficientdet.engine.train_single_epoch(
            model=model,
            dataset=ds,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=epoch)
        
        # TODO: Save checkpoint & validate


@click.command()

# Neural network parameters
@click.option('--D', type=int, default=0,
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
@click.option('--train-dataset', type=click.Choice(['VOC'],
              default='VOC', help='Dataset to use for training')
@click.option('--n-classes', type=int, required=True,
              help='Number of important classes without '
                   'taking background into account')
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    main()