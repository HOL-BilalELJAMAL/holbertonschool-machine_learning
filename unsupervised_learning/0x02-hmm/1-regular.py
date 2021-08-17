#!/usr/bin/env python3
"""
1-regular.py
Module that defines a function called regular
"""

import numpy as np


def regular(P):
    """
    Function that determines the steady state probabilities of a regular
    markov chain

    Args:
        P: square 2D numpy.ndarray of shape (n, n)
        representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain

    Returns:
        numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None

    n = P.shape[0]

    w, v = np.linalg.eig(P.T)

    index = np.where(np.isclose(w, 1))

    if len(index[0]):
        index = index[0][0]
    else:
        return None

    s = v[:, index]

    if any(np.isclose(s, 0)):
        return None

    s = s / np.sum(s)

    s = s[np.newaxis, :]

    return s
