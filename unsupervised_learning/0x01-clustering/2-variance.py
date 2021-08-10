#!/usr/bin/env python3
"""
2-variance.py
Module that defines a function called variance
"""

import numpy as np


def variance(X, C):
    """
    Function that calculates the total intra-cluster variance
    for a data set

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster

    Returns:
        var, or None on failure
    """
    try:
        a2 = np.sum(C ** 2, axis=1)[:, np.newaxis]
        b2 = np.sum(X ** 2, axis=1)
        ab = np.matmul(C, X.T)
        SED = a2 - 2 * ab + b2
        var = np.sum(np.amin(SED, axis=0))
        return var
    except Exception:
        return None
