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
    X = pca(X, idims)
    n, _ = X.shape
    P = P_affinities(X, perplexity=perplexity) * 4
    Y = np.random.randn(n, ndims)
    actualY = Y

    for i in range(0, iterations):
        if i != 0 and i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))

        dY, Q = grads(Y, P)

        if i <= 20:
            alpha = 0.5
        else:
            alpha = 0.8

        auxiliar = Y
        Y = Y - lr * dY + alpha * (Y - actualY)
        actualY = auxiliar
        Y = Y - np.mean(Y, axis=0)

        if i == 100:
            P = P / 4.

    return Y
