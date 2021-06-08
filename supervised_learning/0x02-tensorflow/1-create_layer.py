#!/usr/bin/env python3
"""
1-create_layer.py
Module that defines a function called create_layer
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Function that returns  tensor output of the layer.

    Args:
        - prev is the tensor output of the previous layer
        - n is the number of nodes in the layer to create
        - activation is the activation function that the layer should use

    Returns:
        the tensor output of the layer
    """
    weight = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=weight, name="layer")
    y = layer(prev)
    return (y)
