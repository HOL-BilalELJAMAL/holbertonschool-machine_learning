#!/usr/bin/env python3
"""
2-identity_block.py
Module that defines a function called identity_block
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Function that builds an identity block

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12:
            F11 is the number of filters in the first 1x1 convolution
            F3 is the number of filters in the 3x3 convolution
            F12 is the number of filters in the second 1x1 convolution

    Returns:
        Activated output of the identity block
    """
    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=None)

    my_layer = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               )(A_prev)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    my_layer = K.layers.Conv2D(filters=F3,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    my_layer = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)

    output = K.layers.Add()([my_layer, A_prev])

    output = K.layers.Activation('relu')(output)

    return output
