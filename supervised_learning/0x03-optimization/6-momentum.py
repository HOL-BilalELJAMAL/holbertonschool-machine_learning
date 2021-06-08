#!/usr/bin/env python3
"""
6-momentum.py
Module that defines a function called create_momentum_op
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Function that creates the training operation for a neural network in
    tensorflow using the gradient descent with momentum optimization algorithm

    Args:
        loss: Loss of the network
        alpha (float): Learning rate
        beta1 (float): Momentum weight

    Returns:
        The momentum optimization operation
    """
    train = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    return train.minimize(loss)
