import json
from pathlib import Path
from typing import Mapping, Any

import click

import tensorflow as tf
import tensorflow_addons as tfa

import efficientdet
from .callbacks import COCOmAPCallback


def train(config: efficientdet.config.EfficientDetCompudScaling, 
          save_checkpoint_dir: Path, 
          ds: tf.data.Dataset, 
          val_ds: tf.data.Dataset,
          class2idx: Mapping[str, int] ,
          **kwargs: Any) -> None:

    weights_file = str(save_checkpoint_dir / 'model.h5')
    im_size = config.input_size

    steps_per_epoch = sum(1 for _ in ds)
    if val_ds is not None:
        validation_steps = sum(1 for _ in val_ds)
    else:
        validation_steps = 0

    if kwargs['from_pretrained'] is not None:
        model = efficientdet.EfficientDet(
            weights=kwargs['from_pretrained'],
            num_classes=len(class2idx),
            custom_head_classifier=True,
            freeze_backbone=kwargs['freeze_backbone'],
            training_mode=True)

        print('Training from a pretrained model...')
        print('This will override any configuration related to EfficientNet'
              ' using the defined in the pretrained model.')
    else:
        model = efficientdet.EfficientDet(
            len(class2idx),
            D=kwargs['efficientdet'],
            bidirectional=kwargs['bidirectional'],
            freeze_backbone=kwargs['freeze_backbone'],
            weights='imagenet',
            training_mode=True)

    if kwargs['w_scheduler']:
        lr = efficientdet.optim.WarmupCosineDecayLRScheduler(
            kwargs['learning_rate'],
            warmup_steps=steps_per_epoch,
            decay_steps=steps_per_epoch * (kwargs['epochs'] - 1),
            alpha=kwargs['alpha'])
    else:
        lr = kwargs['learning_rate']

    optimizer = tfa.optimizers.SGDW(learning_rate=lr,
                                    momentum=0.9, 
                                    weight_decay=4e-5)

    # Declare loss functions
    regression_loss_fn = efficientdet.losses.EfficientDetHuberLoss()
    clf_loss_fn = efficientdet.losses.EfficientDetFocalLoss()
    
    # Wrap datasets so they return the anchors labels
    wrapped_ds = efficientdet.wrap_detection_dataset(
        ds, im_size=im_size, num_classes=len(class2idx))

    wrapped_val_ds = efficientdet.wrap_detection_dataset(
        val_ds, im_size=im_size, 
        num_classes=len(class2idx))
    
    model.compile(loss=[regression_loss_fn, clf_loss_fn], 
                  optimizer=optimizer)

    # Mock calls to create model specs
    model.build([None, *im_size, 3])
    model.summary()

    if kwargs['checkpoint'] is not None:
        model.load_weights(str(Path(kwargs['checkpoint']) / 'model.h5'))

    model.save_weights(weights_file)
    kwargs.update(n_classes=len(class2idx))
    json.dump(kwargs, (save_checkpoint_dir / 'hp.json').open('w'))

    callbacks = [COCOmAPCallback(val_ds, 
                                 class2idx, 
                                 validate_every=kwargs['validate_freq']),
                 tf.keras.callbacks.ModelCheckpoint(weights_file)]

    model.fit(wrapped_ds.repeat(),
              validation_data=wrapped_val_ds, 
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              epochs=kwargs['epochs'],
              callbacks=callbacks)


@click.group()

# Neural network parameters
@click.option('--efficientdet', type=int, default=0,
              help='EfficientDet architecture. '
                   '{0, 1, 2, 3, 4, 5, 6, 7}')
@click.option('--bidirectional/--no-bidirectional', default=True,
              help='If bidirectional is set to false the NN will behave as '
                   'a "normal" retinanet, otherwise as EfficientDet')
@click.option('--freeze-backbone/--no-freeze-backbone', 
              default=False, help='Wether or not freeze EfficientNet backbone')

# Logging parameters
@click.option('--print-freq', type=int, default=10,
              help='Print training loss every n steps')
@click.option('--validate-freq', type=int, default=3,
              help='Print COCO evaluations every n epochs')

# Training parameters
@click.option('--epochs', type=int, default=20,
              help='Number of epochs to train the model')
@click.option('--batch-size', type=int, default=16,
              help='Dataset batch size')
@click.option('--learning-rate', type=float, default=1e-3,
              help='Optimizer learning rate. It is recommended to reduce it '
                   'in case backbone is not frozen')
@click.option('--w-scheduler/--wo-scheduler', default=True,
              help='With learning rate scheduler or not. If left to true, '
                   '--learning-rate option will act as max lr for the scheduler')
@click.option('--alpha', type=float, default=0.,
              help='Proportion to reduce the learning rate during '
                   'the decay period')

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

@click.pass_context
def main(ctx: click.Context, **kwargs: Any) -> None:
    ctx.ensure_object(dict)

    save_checkpoint_dir = Path(kwargs['save_dir'])
    save_checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    config = efficientdet.config.EfficientDetCompudScaling(
        D=kwargs['efficientdet'])

    ctx.obj['common'] = kwargs
    ctx.obj['config'] = config
    ctx.obj['save_checkpoint_dir'] = save_checkpoint_dir


@main.command(name='VOC')
@click.option('--root-train', type=click.Path(file_okay=False),
              help='Path to VOC formatted train dataset', required=True)
@click.option('--root-valid', type=click.Path(file_okay=False),
              help='Path to VOC formatted validation dataset', default=None)
@click.pass_context
def VOC(ctx: click.Context, **kwargs: Any) -> None:
    kwargs.update(ctx.obj['common'])

    config = ctx.obj['config']
    save_checkpoint_dir = ctx.obj['save_checkpoint_dir']

    class2idx = efficientdet.data.voc.LABEL_2_IDX
    im_size = config.input_size
    train_ds = efficientdet.data.voc.build_dataset(
        kwargs['root_train'],
        im_size,
        shuffle=True,
        data_augmentation=True)
    
    train_ds = train_ds.padded_batch(batch_size=kwargs['batch_size'],
                                     padded_shapes=((*im_size, 3), 
                                                    ((None,), (None, 4))),
                                     padding_values=(0., (-1, -1.)))

    if kwargs['root_valid'] is not None:
        valid_ds = efficientdet.data.voc.build_dataset(
            kwargs['root_valid'],
            im_size,
            shuffle=False, 
            data_augmentation=False)
    
        valid_ds = valid_ds.padded_batch(batch_size=kwargs['batch_size'],
                                        padded_shapes=((*im_size, 3), 
                                                        ((None,), (None, 4))),
                                        padding_values=(0., (-1, -1.)))
    else:
        valid_ds = None

    train(config, save_checkpoint_dir, 
          train_ds, valid_ds, class2idx, **kwargs)


@main.command()

@click.option('--root-train', type=click.Path(file_okay=False, exists=True),
              required=True, help='Path to train  annotations')
@click.option('--root-valid', default=None, 
              type=click.Path(file_okay=False),
              help='Path to validation annotations. If it is '
                   ' not set by the user, validation won\'t be performed')
@click.option('--images-path', type=click.Path(file_okay=False, exists=True),
              required=True, 
              help='Base path to images. '
                   'Required when using labelme format')
@click.option('--classes-file', type=click.Path(dir_okay=False, exists=True), 
              required=True,
              help='path to file containing a class for line')

@click.pass_context
def labelme(ctx: click.Context, **kwargs: Any) -> None:
    kwargs.update(ctx.obj['common'])
    
    config = ctx.obj['config']
    save_checkpoint_dir = ctx.obj['save_checkpoint_dir']

    _, class2idx = efficientdet.utils.io.read_class_names(
        kwargs['classes_file'])

    im_size = config.input_size

    train_ds = efficientdet.data.labelme.build_dataset(
        annotations_path=kwargs['root_train'],
        images_path=kwargs['images_path'],
        class2idx=class2idx,
        im_input_size=im_size,
        shuffle=True,
        data_augmentation=True)
    
    train_ds = train_ds.padded_batch(batch_size=kwargs['batch_size'],
                                     padded_shapes=((*im_size, 3), 
                                                    ((None,), (None, 4))),
                                     padding_values=(0., (-1, -1.)))

    valid_ds = None
    if kwargs['root_valid']:
        valid_ds = efficientdet.data.labelme.build_dataset(
            annotations_path=kwargs['root_valid'],
            images_path=kwargs['images_path'],
            class2idx=class2idx,
            im_input_size=im_size,
            shuffle=False,
            data_augmentation=False)
        
        valid_ds = valid_ds.padded_batch(batch_size=kwargs['batch_size'],
                                         padded_shapes=((*im_size, 3), 
                                                        ((None,), (None, 4))),
                                         padding_values=(0., (-1, -1.)))

    train(config, save_checkpoint_dir, 
          train_ds, valid_ds, class2idx, **kwargs)


if __name__ == "__main__":
    main()
