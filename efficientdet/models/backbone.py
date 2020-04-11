import tensorflow as tf
import efficientnet.tfkeras as efficientnet


def build_efficient_net_backbone(B: int = 0, 
                                 weights: str = 'imagenet') -> tf.keras.Model:
    """
    Creates Efficient Net backbone returning features at
    each level

    Parameters
    ----------
    B: int, default 0
        EfficientNet architecture tu use. Refer to Table 1 from 
        EfficientDet paper
    weights: str, default 'imagenet'
        If set to 'imagenet' then the backbone will
        be pretrained with imagenet dataset. Otherwise weights can be
        set to none for randomly initialized model
    
    Returns
    -------
    tf.keras.Model
        EfficientDet with intermediate outputs so later we can
        create an FPN on top of it
    """
    efficientnet_cls = getattr(efficientnet, f"EfficientNetB{B:d}")
    base_model = efficientnet_cls(weights=weights, include_top=False)

    layers = base_model.layers
    features = []
    for l, nl in zip(layers[:-1], layers[1:]):
        if hasattr(nl, 'strides') and nl.strides[0] == 2:
            features.append(l)
    
    features.append(nl)

    return tf.keras.Model(base_model.input, 
                          outputs=[f.output for f in features[-5:]])
