#!/usr/bin/env python3
"""
1-inception_network.py
Module that defines a function called inception_network
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Function that builds the inception network

    Returns:
        The keras model
    """
    initializer = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    my_layer = K.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(X)

    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    my_layer = K.layers.Conv2D(filters=64,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.Conv2D(filters=192,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    my_layer = inception_block(my_layer, [64, 96, 128, 16, 32, 32])

    my_layer = inception_block(my_layer, [128, 128, 192, 32, 96, 64])

    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    my_layer = inception_block(my_layer, [192, 96, 208, 16, 48, 64])
    my_layer = inception_block(my_layer, [160, 112, 224, 24, 64, 64])
    my_layer = inception_block(my_layer, [128, 128, 256, 24, 64, 64])
    my_layer = inception_block(my_layer, [112, 144, 288, 32, 64, 64])
    my_layer = inception_block(my_layer, [256, 160, 320, 32, 128, 128])

    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    my_layer = inception_block(my_layer, [256, 160, 320, 32, 128, 128])
    my_layer = inception_block(my_layer, [384, 192, 384, 48, 128, 128])

    my_layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(my_layer)

    my_layer = K.layers.Dropout(rate=0.4)(my_layer)

    my_layer = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer,
                              )(my_layer)

    model = K.models.Model(inputs=X, outputs=my_layer)

    return model
