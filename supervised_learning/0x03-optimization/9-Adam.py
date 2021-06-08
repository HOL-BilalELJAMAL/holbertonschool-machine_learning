#!/usr/bin/env python3
"""
9-Adam.py
Module that defines a function called update_variables_Adam
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Function that updates a variable in place using the Adam optimization
    algorithm

    Args:
        alpha (float): Learning rate
        beta1 (float): Momentum weight 1
        beta2 (float): Momentum weight 2
        epsilon (float): Small number to avoid division by zero
        var (numpy.ndarray): Variable to be updated
        grad (numpy.ndarray): Gradient of the variable to be updated
        v (float): Previous first moment of the variable to be updated
        s (float): Previous Second moment of the variable to be updated
        t (float): Time step used for bias correction

    Returns:
        The updated variable, the new first moment, and the new second moment
    """
    Vdw = beta1 * v + (1 - beta1) * grad
    Sdw = beta2 * s + (1 - beta2) * (grad**2)
    Vdwc = Vdw / (1 - beta1**t)
    Sdwc = Sdw / (1 - beta2**t)
    W = var - alpha * Vdwc / (Sdwc ** (1/2) + epsilon)
    return W, Vdw, Sdw
