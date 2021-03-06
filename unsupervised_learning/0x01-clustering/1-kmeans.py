#!/usr/bin/env python3
"""
1-kmeans.py
Module that defines a function called kmeans
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Function that initializes cluster centroids for K-means

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset that will
            be used for K-means clustering:
            - n is the number of data points
            - d is the number of dimensions for each data point
        k: positive integer containing the number of cluster
        iterations: positive integer containing the maximum number of iteration
        that should be performed

    Returns:
        C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    n, d = X.shape

    min_val = np.amin(X, axis=0)
    max_val = np.amax(X, axis=0)

    C = np.random.uniform(min_val, max_val, (k, d))
    C_prev = np.copy(C)

    X_ = X[:, :, np.newaxis]
    C_ = C.T[np.newaxis, :, :]
    diff = X_ - C_
    D = np.linalg.norm(diff, axis=1)

    clss = np.argmin(D, axis=1)

    for i in range(iterations):

        for j in range(k):
            index = np.where(clss == j)
            if len(index[0]) == 0:
                C[j] = np.random.uniform(min_val, max_val, (1, d))
            else:
                C[j] = np.mean(X[index], axis=0)

        X_ = X[:, :, np.newaxis]
        C_ = C.T[np.newaxis, :, :]
        diff = X_ - C_
        D = np.linalg.norm(diff, axis=1)

        clss = np.argmin(D, axis=1)

        if (C == C_prev).all():
            return C, clss
        C_prev = np.copy(C)

    return C, clss
