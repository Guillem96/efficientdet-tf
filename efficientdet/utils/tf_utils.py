from typing import Any, Sequence

import tensorflow as tf


def call_cascade(layers: Sequence[tf.keras.layers.Layer], 
                 inp: Any, training: bool = True) -> Any:
    """
    Calls a set of layers using the output as cascade.

    The equivalent python code would be
        x = inp
        for l in layers:
            x = l(x, **kwargs)
        return x
    """
    x = inp
    for l in layers:
        x = l(x, training=training)
    return x
