#!/usr/bin/env python3
"""
4-P_affinities.py
Module that defines a function called P_affinities
"""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Function that calculates the symmetric P affinities of a data set

    Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        to be transformed by t-SNE
            n - is the number of data points
            d - is the number of dimensions in each point

    Returns:
        P, a numpy.ndarray of shape (n, n) containing the symmetric
            P affinities
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        Di = np.append(D[i, :i], D[i, i+1:])

        Hi, Pi = HP(Di, betas[i])
        low, high = None, None
        perp_diff = Hi - H

        while np.abs(perp_diff) > tol:
            if perp_diff > 0:
                low = betas[i, 0]
                if high is None:
                    betas[i] = betas[i] * 2.
                else:
                    betas[i] = (betas[i] + high) / 2.
            else:
                high = betas[i, 0]
                if low is None:
                    betas[i] = betas[i] / 2.
                else:
                    betas[i] = (betas[i] + low) / 2.
            Hi, Pi = HP(Di, betas[i])
            perp_diff = Hi - H

        P[i, :i] = Pi[:i]
        P[i, i+1:] = Pi[i:]
    return (P + P.T) / (2*n)
