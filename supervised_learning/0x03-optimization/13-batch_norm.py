#!/usr/bin/env python3
"""
13-batch_norm.py
Module that defines a function called batch_norm
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Function that normalizes an unactivated output of a neural network
    using batch normalization

    Args:
        Z (numpy.ndarray): Array that should be normalized
                - m is the number of data points
                - n is the number of features in Z
        gamma (numpy.ndarray): Scales used for batch normalization
        beta (numpy.ndarray): Offsets used for batch normalization
        epsilon (float): Small number used to avoid division by zero

    Returns:
        The the normalized Z matrix
    """
    z_normalized = (Z - Z.mean(axis=0)) / ((Z.var(axis=0) + epsilon) ** 0.5)
    return gamma * z_normalized + beta
