#!/usr/bin/env python3
"""
5-create_train_op.py
Module that defines a function called create_train_op
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Function that  creates the training operation for the network

    Args:
        - loss is the loss of the network’s prediction
        - alpha is the learning rate

    Returns:
        Returns: an operation that trains the network using gradient descent
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
