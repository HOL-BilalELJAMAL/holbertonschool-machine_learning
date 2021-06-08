#!/usr/bin/env python3
"""
14-batch_norm.py
Module that defines a function called create_batch_norm_layer
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Function that creates a batch normalization layer for a neural network
    in tensorflow

    Args:
        prev: Activated output of the previous layer
        n (int): Number of nodes in the layer to be create
        activation Activation function that should be used on the output of the
        layer

     Returns:
        The normalized Z matrix
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None,
                            kernel_initializer=w, name="layer")
    y = layer(prev)
    mean, variance = tf.nn.moments(y, [0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    y_norm = tf.nn.batch_normalization(y, mean, variance, offset=beta,
                                       scale=gamma, variance_epsilon=1e-8)
    return activation(y_norm)
