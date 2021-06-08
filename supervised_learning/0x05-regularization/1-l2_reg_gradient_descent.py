#!/usr/bin/env python3
"""
1-l2_reg_gradient_descent.py
Module that defines a function called l2_reg_gradient_descent
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function that updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Args:
        Y (np.ndarray): one-hot matrix of shape (classes, m) that contains
                        the correct labels for the data
        weights (dict): the weights and biases of the neural network
        cache (dict): the outputs of each layer of the neural network
        alpha (float):  learning rate
        lambtha (float): L2 regularization parameter
        L (int): number of layers of the network

    Returns:
        dict: weights and biases of the network should be updated in place
    """
    weights_copy = weights.copy()
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        w = "W" + str(i)
        b = "b" + str(i)
        dw = (1 / len(Y[0])) * np.matmul(dz, A.T) + 1/len(
            Y[0]) * lambtha * weights[w]
        db = (1 / len(Y[0])) * np.sum(dz, axis=1, keepdims=True)
        weights[w] = weights[w] - alpha * dw
        weights[b] = weights[b] - alpha * db
        dz = np.matmul(weights_copy["W" + str(i)].T, dz) * (1 - A * A)
