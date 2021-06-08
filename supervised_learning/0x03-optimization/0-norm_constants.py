#!/usr/bin/env python3
"""
0-norm_constants.py
Module that defines a function called normalization_constants
"""

import numpy as np


def normalization_constants(X):
    """
    Function that calculates the normalization  constants of a matrix

    Args:
        X (numpy.ndarray) of shape (m, nx) to normalize
            - m is the number of data points
            - nx is the number of features

    Returns:
        Mean and standard deviation of each feature
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
