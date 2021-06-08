#!/usr/bin/env python3
"""
2-shuffle_data.py
Module that defines a function called shuffle_data
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Function that shuffles the data points in two matrices the same way

    Args:
        X (numpy.ndarray) of shape (m, nx) to shuffle
            - m is the number of data points
            - nx is the number of features
        Y (numpy.ndarray) of shape (m, ny) to shuffle
            - m is the same number of data points as in X
            - ny is the number of features in Y

    Returns:
        The shuffled X and Y matrices
    """
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return X[p], Y[p]
