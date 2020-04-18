from pathlib import Path
from typing import Tuple, Mapping

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


def train(config: efficientdet.config.EfficientDetCompudScaling, 
          save_checkpoint_dir: Path, 
          anchors: tf.Tensor, 
          ds: tf.data.Dataset, 
          val_ds: tf.data.Dataset,
          class2idx: Mapping[str, int] ,
          **kwargs):

    steps_per_epoch = sum(1 for _ in ds)
    if val_ds is not None:
        validation_steps = sum(1 for _ in val_ds)

    if kwargs['checkpoint'] is not None:
        print('Loading checkpoint from {}...'.format(kwargs['checkpoint']))
        (model, optimizer), _ = efficientdet.checkpoint.load(
            kwargs['checkpoint'], load_optimizer=True)
        for l in model.layers:
            l.trainable = True
        model.trainable = True

    elif kwargs['from_pretrained'] is not None:
        model = (efficientdet.EfficientDet
                 .from_pretrained(kwargs['from_pretrained'], 
                                  num_classes=len(class2idx)))
        for l in model.layers:
            l.trainable = True
        model.trainable = True
        print('Training from a pretrained model...')
        print('This will override any configuration related to EfficientNet'
              ' using the defined in the pretrained model.')
    else:
        model = efficientdet.models.EfficientDet(
            len(class2idx),
            D=kwargs['efficientdet'],
            bidirectional=kwargs['bidirectional'],
            freeze_backbone=kwargs['freeze_backbone'],
            weights='imagenet')

    # Only recreate optimizer and scheduler if not loading from checkpoint
    if kwargs['checkpoint'] is None or kwargs['from_pretrained'] is not None:
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
            num_classes=len(class2idx),
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
        arch = kwargs['efficientdet']
        save_dir = (save_checkpoint_dir / 
                    f'{arch}_{model_type}_{epoch}')
        kwargs['n_classes'] = len(class2idx)
        efficientdet.checkpoint.save(
            model, kwargs, save_dir, optimizer=optimizer)


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
def main(ctx, **kwargs):
    ctx.ensure_object(dict)

    save_checkpoint_dir = Path(kwargs['save_dir'])
    save_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    config = efficientdet.config.EfficientDetCompudScaling(
        D=kwargs['efficientdet'])

    anchors = generate_anchors(efficientdet.config.AnchorsConfig(),
                               config.input_size)

    ctx.obj['common'] = kwargs
    ctx.obj['config'] = config
    ctx.obj['anchors'] = anchors
    ctx.obj['save_checkpoint_dir'] = save_checkpoint_dir


@main.command(name='VOC')
@click.option('--train-dataset', type=click.Path(file_okay=False),
              help='Path to save the dataset. Useful to set a google cloud '
                   'bucket to later train with TPUs')
@click.pass_context
def VOC(ctx, **kwargs):
    kwargs.update(ctx.obj['common'])

    config, anchors = ctx.obj['config'], ctx.obj['anchors']
    save_checkpoint_dir = ctx.obj['save_checkpoint_dir']

    class2idx = efficientdet.data.voc.LABEL_2_IDX
    im_size = (config.input_size,) * 2
    train_ds = efficientdet.data.voc.build_dataset(
        kwargs.get('train_dataset', None),
        im_size,
        'train',
        shuffle=True, 
        data_augmentation=True)
    
    valid_ds = efficientdet.data.voc.build_dataset(
        kwargs.get('train_dataset', None),
        im_size,
        'validation',
        shuffle=False, 
        data_augmentation=False)
    
    train_ds = train_ds.padded_batch(batch_size=kwargs['batch_size'],
                                     padded_shapes=((*im_size, 3), 
                                                    ((None,), (None, 4))),
                                     padding_values=(0., (-1, -1.)))

    valid_ds = valid_ds.padded_batch(batch_size=kwargs['batch_size'],
                                     padded_shapes=((*im_size, 3), 
                                                    ((None,), (None, 4))),
                                     padding_values=(0., (-1, -1.)))

    train(config, save_checkpoint_dir, 
          anchors, train_ds, valid_ds, class2idx, **kwargs)


@main.command()

@click.option('--train-dataset', type=click.Path(file_okay=False, exists=True),
              required=True, help='Path to train  annotations')
@click.option('--val-dataset', default=None, 
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
def labelme(ctx, **kwargs):
    kwargs.update(ctx.obj['common'])
    
    config, anchors = ctx.obj['config'], ctx.obj['anchors']
    save_checkpoint_dir = ctx.obj['save_checkpoint_dir']

    classes, class2idx = efficientdet.utils.io.read_class_names(
        kwargs['classes_file'])

    im_size = (config.input_size,) * 2

    train_ds = efficientdet.data.labelme.build_dataset(
        annotations_path=kwargs['train_dataset'],
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
    if kwargs['val_dataset']:
        valid_ds = efficientdet.data.labelme.build_dataset(
            annotations_path=kwargs['val_dataset'],
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
          anchors, train_ds, valid_ds, class2idx, **kwargs)


if __name__ == "__main__":
    main()
