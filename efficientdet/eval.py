import click

import efficientdet
import efficientdet.engine as engine

def evaluate(**kwargs):
    model, params = efficientdet.checkpoint.load(kwargs['checkpoint'])
    
    if kwargs['format'] == 'labelme':
        classes = params['classes_names'].split(',')
        class2idx = {c: i for i, c in enumerate(classes)}
        n_classes = len(classes)

    elif kwargs['format'] == 'VOC':
        class2idx = efficientdet.data.voc.LABEL_2_IDX
        classes = efficientdet.data.voc.IDX_2_LABEL
        n_classes = 20

    ds, class2idx = efficientdet.data.build_ds(
        format=kwargs['format'],
        annots_path=kwargs['test_dataset'],
        images_path=kwargs['images_path'],
        im_size=(model.config.input_size,) * 2,
        class_names=params['classes_names'].split(','),
        data_augmentation=False,
        batch_size=1)
    
    engine.evaluate(
            model=model,
            dataset=ds,
            class2idx=class2idx)


@click.command()

# Data parameters
@click.option('--format', type=click.Choice(['VOC', 'labelme']),
              required=True, help='Dataset to use for training')
@click.option('--test-dataset', type=click.Path(file_okay=False, exists=True),
              required=True, help='Path to annotations and images')
@click.option('--images-path', type=click.Path(file_okay=False),
              required=True, default='',
              help='Base path to images. '
                   'Required when using labelme format')

# Checkpointing parameters
@click.option('--checkpoint', help='Path to model checkpoint',
              type=click.Path(), required=True)
def main(**kwargs):
    evaluate(**kwargs)


if __name__ == "__main__":
    main()