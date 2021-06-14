#!/usr/bin/env python3
"""
4-resnet50.py
Module that defines a function called resnet50
"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Function that builds the ResNet-50 architecture

    Returns:
        The keras model
    """
    initializer = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    my_layer = K.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=initializer,
                               )(X)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    my_layer = projection_block(my_layer, [64, 64, 256], 1)
    for i in range(2):
        my_layer = identity_block(my_layer, [64, 64, 256])

    my_layer = projection_block(my_layer, [128, 128, 512])
    for i in range(3):
        my_layer = identity_block(my_layer, [128, 128, 512])

    my_layer = projection_block(my_layer, [256, 256, 1024])
    for i in range(5):
        my_layer = identity_block(my_layer, [256, 256, 1024])

    my_layer = projection_block(my_layer, [512, 512, 2048])
    for i in range(2):
        my_layer = identity_block(my_layer, [512, 512, 2048])

    my_layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(my_layer)

    my_layer = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer,
                              )(my_layer)

    model = K.models.Model(inputs=X, outputs=my_layer)

    return model
