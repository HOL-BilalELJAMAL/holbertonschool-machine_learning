#!/usr/bin/env python3
"""
5-dense_block.py
Module that defines a function called dense_block
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Function that builds a dense block

    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        Concatenated output of each layer within the Dense Block and
        the number of filters within the concatenated outputs
    """
    initializer = K.initializers.he_normal(seed=None)

    for i in range(layers):
        my_layer = K.layers.BatchNormalization()(X)
        my_layer = K.layers.Activation('relu')(my_layer)

        my_layer = K.layers.Conv2D(filters=4*growth_rate,
                                   kernel_size=1,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)

        my_layer = K.layers.BatchNormalization()(my_layer)
        my_layer = K.layers.Activation('relu')(my_layer)

        my_layer = K.layers.Conv2D(filters=growth_rate,
                                   kernel_size=3,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)

        X = K.layers.concatenate([X, my_layer])
        nb_filters += growth_rate

    return X, nb_filters
