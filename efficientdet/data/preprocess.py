import tensorflow as tf


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize the image according imagenet mean and std

    Parameters
    ----------
    image: tf.Tensor of shape [H, W, C]
        Image in [0, 1] range
    
    Returns
    -------
    tf.Tensor
        Normalized image
    """
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return (image - mean) / std


def unnormalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Unnormalize the image according imagenet mean and std

    Parameters
    ----------
    image: tf.Tensor of shape [H, W, C]
        Image in [0, 1] range
    
    Returns
    -------
    tf.Tensor
        Unnormalized image
    """
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return image * std + mean