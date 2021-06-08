#!/usr/bin/env python3
"""
10-Adam.py
Module that defines a function called create_Adam_op
"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Function that creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm

    Args:
        loss: Loss of the network
        alpha (float): Learning rate
        beta1 (float): Weight used for the first moment
        beta2 (float): Momentum weight
        epsilon (float): Small number to avoid division by zero

    Returns:
        the Adam optimization operation
    """
    train = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return train.minimize(loss)
