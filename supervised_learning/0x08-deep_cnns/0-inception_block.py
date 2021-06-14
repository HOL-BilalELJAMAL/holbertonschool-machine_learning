#!/usr/bin/env python3
"""
0-inception_block.py
Module that defines a function called inception_block
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Function that builds an inception block as described
    in Going Deeper with Convolutions (2014)

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F1, F3R, F3,F5R, F5, FPP:
            F1 is the number of filters in the 1x1 convolution
            F3R is the number of filters in the 1x1 convolution
            before the 3x3 convolution
            F3 is the number of filters in the 3x3 convolution
            F5R is the number of filters in the 1x1 convolution
            before the 5x5 convolution
            F5 is the number of filters in the 5x5 convolution
            FPP is the number of filters in the 1x1 convolution
            after the max pooling

    Returns:
        Concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    initializer = K.initializers.he_normal(seed=None)

    my_layer = K.layers.Conv2D(filters=F1,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(A_prev)

    my_layer1 = K.layers.Conv2D(filters=F3R,
                                kernel_size=(1, 1),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(A_prev)

    my_layer1 = K.layers.Conv2D(filters=F3,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(my_layer1)

    my_layer2 = K.layers.Conv2D(filters=F5R,
                                kernel_size=(1, 1),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(A_prev)

    my_layer2 = K.layers.Conv2D(filters=F5,
                                kernel_size=(5, 5),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(my_layer2)

    my_layer3 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(1, 1))(A_prev)

    my_layer3 = K.layers.Conv2D(filters=FPP,
                                kernel_size=(1, 1),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(my_layer3)

    output = K.layers.concatenate([my_layer, my_layer1, my_layer2, my_layer3])

    return output
