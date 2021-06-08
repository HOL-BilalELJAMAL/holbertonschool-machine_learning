#!/usr/bin/env python3
"""
7-RMSProp.py
Module that defines a function called update_variables_RMSProp
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Function that updates a variable using the RMSProp optimization algorithm

    Args:
        alpha (float): Learning rate
        beta2 (float): Momentum weight
        epsilon (float): Small number to avoid division by zero
        var (numpy.ndarray): Variable to be updated
        grad (numpy.ndarray): Gradient of the variable to be updated
        s (float): Previous second moment of the variable to be updated

    Returns:
        The updated variable and the new moment
    """
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    w = var - alpha * (grad / ((s_new ** 0.5) + epsilon))
    return w, s_new
