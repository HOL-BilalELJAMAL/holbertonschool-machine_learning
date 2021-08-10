#!/usr/bin/env python3
"""
6-expectation.py
Module that defines a function called expectation
"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Function that calculates the expectation step in the EM algorithm
    for a GMM

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        pi: numpy.ndarray of shape (k,) containing the priors for
        each cluster
        m: numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster

    Returns:
        g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
        l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[1] != d or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if S.shape[0] != k:
        return None, None

    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    sum_i = 0
    num = np.zeros((k, n))
    for i in range(k):
        num[i] = pi[i] * pdf(X, m[i], S[i])
        sum_i += num[i]

    sum_i = np.sum(num, axis=0, keepdims=True)

    g = num / sum_i

    log_likelihood = np.sum(np.log(sum_i))

    return g, log_likelihood
