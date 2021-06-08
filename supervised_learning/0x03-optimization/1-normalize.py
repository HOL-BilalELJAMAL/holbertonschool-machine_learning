#!/usr/bin/env python3
"""
1-normalize.py
Module that defines a function called normalize
"""


def normalize(X, m, s):
    """
    Function that normalizes a matrix X

    Args:
        X (numpy.ndarray) of shape (m, nx) to normalize
            - m is the number of data points
            - nx is the number of features
        m (numpy.ndarray) of shape (nx,) that contains the mean
        of all features of X
        s (numpy.ndarray) of shape (nx,) that contains the standard
        deviation of all features of X

    Returns:
        The normalized X matrix
    """
    return (X - m) / s
