import click

import tensorflow as tf

import matplotlib.pyplot as plt

import efficientdet


@click.command()
@click.option('--image', type=click.Path(dir_okay=False, exists=True))
@click.option('--checkpoint', type=click.Path())
@click.option('--score', type=float, default=.4)

@click.option('--format', type=click.Choice(['VOC', 'labelme']),
              required=True, help='Dataset to use for training')

def main(**kwargs):

    # model, params = efficientdet.checkpoint.load(
    #     kwargs['checkpoint'], score_threshold=kwargs['score'])
    model = efficientdet.EfficientDet.from_pretrained(
        kwargs['checkpoint'], score_threshold=kwargs['score'])
    if kwargs['format'] == 'labelme':
        classes = params['classes_names'].split(',')

    elif kwargs['format'] == 'VOC':
        classes = efficientdet.data.voc.IDX_2_LABEL
    
    # load image
    im_size = model.config.input_size
    im = efficientdet.utils.io.load_image(kwargs['image'], (im_size,) * 2)
    norm_image = efficientdet.data.preprocess.normalize_image(im)

    boxes, labels, scores = model(tf.expand_dims(norm_image, axis=0), 
                                  training=False)

    labels = [classes[l] for l in labels[0]]
    im = efficientdet.visualizer.draw_boxes(im, boxes)
    
    plt.imshow(im)
    plt.axis('off')
    plt.show(block=True)


if __name__ == "__main__":
    main()