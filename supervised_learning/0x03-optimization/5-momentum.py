#!/usr/bin/env python3
"""
5-momentum.py
Module that defines a function called update_variables_momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Function that updates a variable using the gradient descent with
    momentum optimization

    Args:
        alpha (float): Learning rate
        beta1 (float): Momentum weight
        var (numpy.ndarray): Variable to be updated
        grad (numpy.ndarray): Gradient of the variable to be updated
        v (float): Previous first moment of the variable to be updated

    Returns:
        The updated variable and the new moment
    """
    v_dw = beta1 * v + (1 - beta1) * grad
    w = var - (alpha * v_dw)
    return w, v_dw
