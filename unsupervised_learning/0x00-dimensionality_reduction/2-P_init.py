#!/usr/bin/env python3
"""
2-P_init.py
Module that defines a function called P_init
"""

import numpy as np


def P_init(X, perplexity):
    """
    Function that initializes all variables required to calculate
    the P affinities in t-SNE

    Args:
        X - is a numpy.ndarray of shape (n, d) containing the dataset to
        be transformed by t-SNE
            n - is the number of data points
            d - is the number of dimensions in each point
        perplexity - is the perplexity that all Gaussian distributions
        should have

    Returns: (D, P, betas, H)
        D: a numpy.ndarray of shape (n, n) that calculates the pairwise
            distance between two data points
        P: a numpy.ndarray of shape (n, n) initialized to all 0‘s that will
            contain the P affinities
        betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s that will
            contain all of the beta values
            beta_i = 1/(2*sigma_i^2)
        H is the Shannon entropy for perplexity perplexity
    """
    n = X.shape[0]
    X_sum = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), X_sum).T, X_sum)
    np.fill_diagonal(D, 0.)
    betas = np.ones((n, 1))
    P = np.zeros((n, n))
    P_sum = sum(P)
    H = np.log2(perplexity)
    return (D, P, betas, H)
