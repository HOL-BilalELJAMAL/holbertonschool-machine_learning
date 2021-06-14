#!/usr/bin/env python3
"""
6-transition_layer.py
Module that defines a function called transition_layer
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Function that builds a transition layer

    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        compression: compression factor for the transition layer

    Returns:
        Output of the transition layer and the number of filters within
        the output
    """
    initializer = K.initializers.he_normal(seed=None)

    my_layer = K.layers.BatchNormalization()(X)
    my_layer = K.layers.Activation('relu')(my_layer)

    nb_filters = int(nb_filters * compression)

    my_layer = K.layers.Conv2D(filters=nb_filters,
                               kernel_size=1,
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    X = K.layers.AveragePooling2D(pool_size=2,
                                  padding='same')(my_layer)

    return X, nb_filters
