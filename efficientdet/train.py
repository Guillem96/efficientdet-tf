from typing import Tuple
from pathlib import Path

import click

import tensorflow as tf

import efficientdet
import efficientdet.utils as utils
import efficientdet.engine as engine


huber_loss_fn = tf.losses.Huber(
    reduction=tf.losses.Reduction.SUM)


def ds_len(ds):
    return sum(1 for _ in ds)


def loss_fn(y_true_clf: tf.Tensor, 
            y_pred_clf: tf.Tensor, 
            y_true_reg: tf.Tensor, 
            y_pred_reg: tf.Tensor) -> Tuple[float, float]:

    y_shape = tf.shape(y_true_clf)
    batch = y_shape[0]
    n_anchors = y_shape[1]
    
    anchors_states = y_true_clf[:, :, -1]
    not_ignore_idx = tf.where(tf.not_equal(anchors_states, -1.))
    true_idx = tf.where(tf.equal(anchors_states, 1.))
    
    normalizer = tf.shape(true_idx)[0]
    normalizer = tf.cast(normalizer, tf.float32)
    
    y_true_clf = tf.gather_nd(y_true_clf[:, :, :-1], not_ignore_idx)
    y_pred_clf = tf.gather_nd(y_pred_clf, not_ignore_idx)
    
    y_true_reg = tf.gather_nd(y_true_reg[:, :, :-1], true_idx)
    y_pred_reg = tf.gather_nd(y_pred_reg, true_idx)
    
    reg_loss = huber_loss_fn(y_true_reg, y_pred_reg)

    clf_loss = efficientdet.losses.focal_loss(y_true_clf, 
                                              y_pred_clf,
                                              reduction='sum')

    return tf.divide(reg_loss, normalizer), tf.divide(clf_loss, normalizer)


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

    if kwargs['checkpoint'] is not None:
        print('Loading checkpoint from {}...'.format(kwargs['checkpoint']))
        model = efficientdet.checkpoint.load(kwargs['checkpoint'])
    elif kwargs['from_pretrained'] is not None:
        model = (efficientdet.EfficientDet
                 .from_pretrained(kwargs['from_pretrained'], 
                                  num_classes=kwargs['n_classes']))
        for l in model.layers:
            l.trainable = True
        model.trainable = True
        print('Training from a pretrained model...')
        print('This will override any configuration related to EfficientNet'
              ' using the defined in the pretrained model.')
    else:
        model = efficientdet.models.EfficientDet(
            kwargs['n_classes'],
            D=kwargs['efficientdet'],
            bidirectional=kwargs['bidirectional'],
            freeze_backbone=kwargs['freeze_backbone'],
            weights='imagenet')

    ds, class2idx = efficientdet.data.build_ds(
        format=kwargs['format'],
        annots_path=kwargs['train_dataset'],
        images_path=kwargs['images_path'],
        im_size=(model.config.input_size,) * 2,
        class_names=kwargs['classes_names'].split(','),
        batch_size=kwargs['batch_size'],
        data_augmentation=True)

    val_ds = None
    if kwargs['val_dataset']:
        val_ds, _ = efficientdet.data.build_ds(
            format=kwargs['format'],
            annots_path=kwargs['val_dataset'],
            images_path=kwargs['images_path'],
            class_names=kwargs['classes_names'].split(','),
            im_size=(model.config.input_size,) * 2,
            shuffle=False,
            data_augmentation=False,
            batch_size=max(1, kwargs['batch_size'] // 2))

    anchors = generate_anchors(model.anchors_config,
                               model.config.input_size)
    
    steps_per_epoch = ds_len(ds)
    if val_ds is not None:
        validation_steps = ds_len(val_ds)

    if kwargs['w_scheduler']:
        optim_steps = (steps_per_epoch // kwargs['grad_accum_steps']) + 1
        lr = efficientdet.optim.WarmupCosineDecayLRScheduler(
            kwargs['learning_rate'],
            warmup_steps=optim_steps,
            decay_steps=optim_steps * (kwargs['epochs'] - 1),
            alpha=kwargs['alpha'])
    else:
        lr = kwargs['learning_rate']

    optimizer = tf.optimizers.SGD(
        learning_rate=lr,
        momentum=0.9)

    for epoch in range(kwargs['epochs']):

        engine.train_single_epoch(
            model=model,
            anchors=anchors,
            dataset=ds,
            optimizer=optimizer,
            grad_accum_steps=kwargs['grad_accum_steps'],
            loss_fn=loss_fn,
            num_classes=kwargs['n_classes'],
            epoch=epoch,
            steps=steps_per_epoch,
            print_every=kwargs['print_freq'])

        if val_ds is not None and (epoch + 1) % kwargs['validate_freq'] == 0:
            print('Evaluating COCO mAP...')
            engine.evaluate(
                model=model,
                dataset=val_ds,
                steps=validation_steps,
                print_every=kwargs['print_freq'],
                class2idx=class2idx)
        
        model_type = 'bifpn' if kwargs['bidirectional'] else 'fpn'
        data_format = kwargs['format']
        arch = kwargs['efficientdet']
        save_dir = (save_checkpoint_dir / 
                    f'{arch}_{model_type}_{data_format}_{epoch}')
        efficientdet.checkpoint.save(model, kwargs, save_dir)


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
@click.option('--grad-accum-steps', type=int, default=1,
              help='Gradient accumulation steps. Simulates a larger batch '
                   'size, for example if batch_size=16 and grad_accum_steps=2 '
                   'the simulated batch size is 16 * 2 = 32')
@click.option('--learning-rate', type=float, default=1e-3,
              help='Optimizer learning rate. It is recommended to reduce it '
                   'in case backbone is not frozen')
@click.option('--w-scheduler/--wo-scheduler', default=True,
              help='With learning rate scheduler or not. If left to true, '
                   '--learning-rate option will act as max lr for the scheduler')
@click.option('--alpha', type=float, default=1.,
              help='Proportion to reduce the learning rate during '
                   'the decay period')
# Logging parameters
@click.option('--print-freq', type=int, default=10,
              help='Print training loss every n steps')
@click.option('--validate-freq', type=int, default=3,
              help='Print COCO evaluations every n epochs')

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
@click.option('--from-pretrained', 
              help='Path or reference to pretrained model. For example' 
                   ' if you want to train a model from a VOC pretrained '
                   'checkpoint, use the value --from-pretrained D0-VOC',
              type=str, default=None)
@click.option('--save-dir', help='Directory to save model weights',
              required=True, default='models', 
              type=click.Path(file_okay=False))
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    main()
