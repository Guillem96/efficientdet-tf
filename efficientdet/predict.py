import click

import tensorflow as tf

import cv2
import matplotlib.pyplot as plt

import efficientdet


@click.command()
@click.option('--image', type=click.Path(dir_okay=False, exists=True))
@click.option('--checkpoint', type=click.Path())
@click.option('--score', type=float, default=.6)

@click.option('--format', type=click.Choice(['VOC', 'labelme']),
              required=True, help='Dataset to use for training')

@click.option('--classes-names', 
              default='', type=str, 
              help='Only required when format is labelme. '
                   'Name of classes separated using comma. '
                   'class1,class2,class3')
def main(**kwargs):

    model, params = efficientdet.checkpoint.load(
        kwargs['checkpoint'], score_threshold=kwargs['score'])

    if kwargs['format'] == 'labelme':
        classes = params['classes_names'].split(',')
        class2idx = {c: i for i, c in enumerate(classes)}
        n_classes = len(classes)

    elif kwargs['format'] == 'VOC':
        class2idx = efficientdet.data.voc.LABEL_2_IDX
        classes = efficientdet.data.voc.IDX_2_LABEL
        n_classes = 20
    
    # load image
    im_size = model.config.input_size
    im = efficientdet.utils.io.load_image(kwargs['image'], (im_size,) * 2)
    norm_image = efficientdet.data.preprocess.normalize_image(im)

    boxes, labels, scores = model(tf.expand_dims(norm_image, axis=0), 
                                  training=False)

    labels = [classes[l] for l in labels[0]]

    im = im.numpy()
    for l, box, s in zip(labels, boxes[0].numpy(), scores[0]):
        x1, y1, x2, y2 = box.astype('int32')

        cv2.rectangle(im, 
                     (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(im, l + ' {:.2f}'.format(s), 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    2, (0, 255, 0), 2)
                    
    plt.imshow(im)
    plt.axis('off')
    plt.show(block=True)


if __name__ == "__main__":
    main()