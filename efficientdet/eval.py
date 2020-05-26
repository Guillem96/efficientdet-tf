import click
from typing import Any

import efficientdet
from efficientdet import coco


@click.group()

@click.option('--batch-size', type=int, default=4)
@click.option('--checkpoint', help='Path to model checkpoint',
              type=click.Path(), required=True)

@click.pass_context
def main(ctx: click.Context, **kwargs: Any) -> None:

    model, params = efficientdet.checkpoint.load(
        kwargs['checkpoint'])
        
    params: dict = {}

    ctx.ensure_object(dict)
    ctx.obj['common'] = kwargs
    ctx.obj['model'] = model
    ctx.obj['params'] = params


@main.command()
# Data parameters
@click.option('--root-test', type=click.Path(file_okay=False, exists=True),
              required=True, help='Path to annotations and images')
@click.option('--images-path', type=click.Path(file_okay=False),
              required=True, default='',
              help='Base path to images. '
                   'Required when using labelme format')
@click.option('--classes-file', type=click.Path(dir_okay=False, exists=True), 
              required=True,
              help='path to file containing a class for line')
@click.pass_context
def labelme(ctx: click.Context, **kwargs: Any) -> None:
    kwargs.update(ctx.obj['common'])
    model, _ = ctx.obj['model'], ctx.obj['params']
    config = model.config
    im_size = config.input_size

    _, class2idx  = efficientdet.utils.io.read_class_names(
        kwargs['classes_file'])

    ds = efficientdet.labelme.build_dataset(
            annotations_path=kwargs['root_test'],
            images_path=kwargs['images_path'],
            class2idx=class2idx,
            im_input_size=im_size,
            shuffle=False)

    ds = ds.padded_batch(batch_size=kwargs['batch_size'],
                         padded_shapes=((*im_size, 3), 
                                        ((None,), (None, 4))),
                         padding_values=(0., (-1, -1.)))

    gtCOCO = coco.tf_data_to_COCO(ds, class2idx)

    coco.evaluate(
        model=model,
        dataset=ds,
        steps=sum(1 for _ in ds),
        gtCOCO=gtCOCO)


@main.command(name='VOC')
@click.option('--root-test', type=click.Path(file_okay=False, exists=True),
              required=True, help='Path to annotations and images')
@click.pass_context
def VOC(ctx: click.Context, **kwargs: Any) -> None:
    kwargs.update(ctx.obj['common'])
    model, _ = ctx.obj['model'], ctx.obj['params']
    config = model.config
    im_size = config.input_size

    class2idx = efficientdet.data.voc.LABEL_2_IDX

    im_size = config.input_size
    ds = efficientdet.data.voc.build_dataset(
        kwargs['root_test'],
        im_input_size=im_size,
        shuffle=False)

    ds = ds.padded_batch(batch_size=kwargs['batch_size'],
                         padded_shapes=((*im_size, 3), 
                                        ((None,), (None, 4))),
                         padding_values=(0., (-1, -1.)))

    gtCOCO = coco.tf_data_to_COCO(ds, class2idx)
    coco.evaluate(
        model=model,
        dataset=ds,
        steps=sum(1 for _ in ds),
        gtCOCO=gtCOCO)


if __name__ == "__main__":
    main()
