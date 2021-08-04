#!/usr/bin/env python3
"""
2-P_init.py
Module that defines a function called P_init
"""

import numpy as np


def P_init(X, perplexity):
    """
    Function that initializes all variables required to calculate the
    P affinities in t-SNE
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
