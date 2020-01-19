import click

import efficientdet
import efficientdet.engine as engine

def evaluate(**kwargs):
    model = efficientdet.models.EfficientDet(
        kwargs['n_classes'],
        D=kwargs['efficientdet'],
        bidirectional=kwargs['bidirectional'],
        freeze_backbone=True,
        weights=None)
    
    model.load_weights(kwargs['checkpoint'])

    ds, class2idx = efficientdet.data.build_ds(
        format=kwargs['format'],
        annots_path=kwargs['test_dataset'],
        images_path=kwargs['images_path'],
        im_size=(model.config.input_size,) * 2,
        class_names=kwargs['classes_names'].split(','),
        batch_size=1)
    
    engine.evaluate(
            model=model,
            dataset=ds,
            class2idx=class2idx)


@click.command()

# Neural network parameters
@click.option('--efficientdet', type=int, default=0,
              help='EfficientDet architecture. '
                   '{0, 1, 2, 3, 4, 5, 6, 7}')
@click.option('--bidirectional/--no-bidirectional', default=True,
              help='If bidirectional is set to false the NN will behave as '
                   'a "normal" retinanet, otherwise as EfficientDet')

# Data parameters
@click.option('--format', type=click.Choice(['VOC', 'labelme']),
              required=True, help='Dataset to use for training')
@click.option('--test-dataset', type=click.Path(file_okay=False, exists=True),
              required=True, help='Path to annotations and images')
@click.option('--images-path', type=click.Path(file_okay=False, exists=True),
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
              type=click.Path(), required=True)
def main(**kwargs):
    evaluate(**kwargs)


if __name__ == "__main__":
    main()