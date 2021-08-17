#!/usr/bin/env python3
"""
0-markov_chain.py
Module that defines a function called markov_chain
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that determines the probability of a markov chain being
    in a particular state after a specified number of iterations

    Args:
        P: square 2D numpy.ndarray of shape (n, n)
        representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
        s: numpy.ndarray of shape (1, n)
        representing the probability of starting in each state
        t: number of iterations that the markov chain has been through

    Returns:
        numpy.ndarray of shape (1, n) representing the probability of
        being in a specific state after t iterations, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    if P.shape[0] != P.shape[1]:
        return None

    if np.sum(P, axis=1).all() != 1:
        return None

    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None

    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None

    if not isinstance(t, int) or t < 0:
        return None

    if t == 0:
        return s

    s_i = np.matmul(s, P)

    for i in range(1, t):
        s_i = np.matmul(s_i, P)

    return s_i
