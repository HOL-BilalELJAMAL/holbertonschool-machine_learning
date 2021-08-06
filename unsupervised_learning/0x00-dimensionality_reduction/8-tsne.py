#!/usr/bin/env python3
"""
8-tsne.py
Module that defines a function called tsne
"""

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Function that performs a t-SNE transformation
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset to be
            transformed by t-SNE
            n - is the number of data points
            d - is the number of dimensions in each point
        ndims - is the new dimensional representation of X
        idims - is the intermediate dimensional representation of X after PCA
        perplexity - is the perplexity
        iterations - is the number of iterations
        lr - is the learning rate
    Returns:
        Y, a numpy.ndarray of shape (n, ndim) containing the optimized low
        dimensional transformation of X
    """
    n, d = X.shape
    X = pca(X, idims)
    P = P_affinities(X, perplexity=perplexity)
    Y = np.random.randn(n, ndims)
    iY = Y
    P = P * 4.

    for i in range(iterations):
        dY, Q = grads(Y, P)
        if i <= 20:
            momentum = 0.5
        else:
            momentum = 0.8

        iY = momentum * iY - lr * dY
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print('Cost at iteration {}: {}'.format((i+1), C))

        if i == 100:
            P = P / 4.

    return Y
