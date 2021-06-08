#!/usr/bin/env python3
"""
8-RMSProp.py
Module that defines a function called create_RMSProp_op
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Function that updates a variable using the RMSProp optimization algorithm

    Args:
        loss: Loss of the network
        alpha (float): Learning rate
        beta2 (float): RMSProp weight
        epsilon(float): Small number to avoid division by zero

    Returns:
        the RMSProp optimization operation
    """
    train = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                      epsilon=epsilon)
    return train.minimize(loss)
