#!/usr/bin/env python3
"""
0-l2_reg_cost.py
Module that defines a function called l2_reg_cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Function that calculates the cost of a neural network with L2
    regularization

    Args:
        cost (float): cost of the network without L2 regularization.
        lambtha (float): regularization parameter.
        weights (dict): the weights and biases (numpy.ndarrays) of the
                        neural network.
        L (int): number of layers in the neural network.
        m (int): number of data points used.

    Returns:
        np.ndarray: the cost of the network accounting for L2 regularization.
    """
    tot = 0
    for ly in range(1, L + 1):
        tot += np.linalg.norm(weights["W{}".format(ly)])
    l2_cost = lambtha * tot / (2 * m)
    return cost + l2_cost
