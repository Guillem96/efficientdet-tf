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
    step = tf.constant(0)
    x = inp
    for l in layers:
        x = l(x, training=training)
    return x

    # def cond(step, output): 
    #     return step < len(layers)

    # def body(step, output):
    #     return step + 1, layers[int(step)](output, training=training)

    # _, features = tf.while_loop(cond, body, loop_vars=[step, inp])
    # return features
