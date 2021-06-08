#!/usr/bin/env python3
"""
2-l2_reg_cost.py
Module that defines a function called l2_reg_cost
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Funtion that calculates the cost of a neural network with L2 regularization

    Args:
        cost (tensor): Cost of the network without L2 regularization

    Returns:
        tensor: Cost of the network accounting for L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
