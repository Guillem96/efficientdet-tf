import click

import efficientdet
import efficientdet.engine as engine


@click.group()

@click.option('--batch-size', type=int, default=4)
@click.option('--checkpoint', help='Path to model checkpoint',
              type=click.Path(), required=True)

@click.pass_context
def main(ctx, **kwargs):

    model, params = efficientdet.checkpoint.load(kwargs['checkpoint'])

    ctx.ensure_object(dict)
    ctx.obj['common'] = kwargs
    ctx.obj['model'] = model
    ctx.obj['params'] = params


@main.command()
# Data parameters
@click.option('--test-dataset', type=click.Path(file_okay=False, exists=True),
              required=True, help='Path to annotations and images')
@click.option('--images-path', type=click.Path(file_okay=False),
              required=True, default='',
              help='Base path to images. '
                   'Required when using labelme format')
@click.option('--classes-file', type=click.Path(dir_okay=False, exists=True), 
              required=True,
              help='path to file containing a class for line')
@click.pass_context
def labelme(ctx, **kwargs):
    kwargs.update(ctx.obj['common'])
    model, params = ctx.obj['model'], ctx.obj['params']
    config = model.config
    im_size = (config.input_size,) * 2

    classes, class2idx  = efficientdet.utils.io.read_class_names(
        kwargs['classes_file'])

    ds = efficientdet.data.labelme.build_dataset(
            annotations_path=kwargs['test_dataset'],
            images_path=kwargs['images_path'],
            class2idx=class2idx,
            im_input_size=im_size,
            shuffle=False,
            data_augmentation=False)

    ds = ds.padded_batch(batch_size=kwargs['batch_size'],
                         padded_shapes=((*im_size, 3), 
                                        ((None,), (None, 4))),
                         padding_values=(0., (-1, -1.)))

    engine.evaluate(
        model=model,
        dataset=ds,
        steps=sum(1 for _ in ds),
        class2idx=class2idx)


@main.command(name='VOC')
@click.pass_context
def VOC(ctx, **kwargs):
    kwargs.update(ctx.obj['common'])
    model, params = ctx.obj['model'], ctx.obj['params']
    config = model.config
    im_size = (config.input_size,) * 2

    class2idx = efficientdet.data.voc.LABEL_2_IDX

    im_size = (config.input_size,) * 2
    ds = efficientdet.data.voc.build_dataset(
        im_input_size=im_size,
        split='test',
        shuffle=True, 
        data_augmentation=True)

    ds = ds.padded_batch(batch_size=kwargs['batch_size'],
                         padded_shapes=((*im_size, 3), 
                                        ((None,), (None, 4))),
                         padding_values=(0., (-1, -1.)))

    engine.evaluate(
        model=model,
        dataset=ds,
        steps=sum(1 for _ in ds),
        class2idx=class2idx)

if __name__ == "__main__":
    main()
