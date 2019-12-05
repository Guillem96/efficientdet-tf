import tensorflow as tf
import efficientnet.tfkeras as efficientnet


def build_efficient_net_backbone(B: int = 0, weights: str = 'imagenet'):
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
    efficientnet_cls = 'efficientnet.EfficientNetB' + str(B)
    base_model = eval(efficientnet_cls)(weights=weights, 
                                        include_top=False)
    
    # Get level features
    # As stated in paper we get the [3, 4, 5, 6, 7] level features
    wanted_levels = range(3, 8)
    features = []
    for wanted_level in wanted_levels:
        # Get all the layers from an specific level
        layers = _filter_layers(model, wanted_level)
        # Get only the last layer which contains the 'Conv block'
        # features
        features.append(layers[-1].output)

    return tf.keras.Model(base_model.input, 
                          outputs=features)

def _filter_layers(model, level):
    # Get layers starting with 'block' + `level`
    block = 'block' + str(level)
    return [l for l in model.layers if l.name.startswith(block)]
