#!/usr/bin/env python3
"""
policy_gradient.py
Function that defines function to compute Monte Carlo policy gradient
based on state and weight matrices
"""


import numpy as np


def policy(matrix, weight):
    """
    Function that computes policy with a weight of a matrix

    Args:
        matrix [numpy.ndarray]: the matrix to compute policy from
        weight [numpy.ndarray]: the weights applied to the matrix

    Returns:
        the policy
    """
    dot_product = matrix.dot(weight)
    exp = np.exp(dot_product)
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """
    Function that computes the Monte Carlo policy gradient based on
    the policy calculated from the above policy() function

    Args:
        state [numpy.ndarray]:
            matrix representing the current observation of the environment
        weight [numpy.ndarray]:
            matrix of random weight

    Returns:
        the action and the gradient
    """
    my_policy = policy(state, weight)
    action = np.random.choice(len(my_policy[0]), p=my_policy[0])
    s = my_policy.reshape(-1, 1)
    softmax = (np.diagflat(s) - np.dot(s, s.T))[action, :]
    dlog = softmax / my_policy[0, action]
    gradient = state.T.dot(dlog[None, :])
    return action, gradient
