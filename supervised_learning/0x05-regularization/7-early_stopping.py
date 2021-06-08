#!/usr/bin/env python3
"""
7-early_stopping.py
Module that defines a function called early_stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Function that determines if you should stop gradient descent early

    Args:
        cost (float): the current validation cost of the neural network
        opt_cost (float): the lowest recorded validation cost of the neural
                          network
        threshold (float): the threshold used for early stopping
        patience (int): how many steps have to be performed before stop early
        count (int): the count of how long the threshold has not been met

    Returns:
        tuple: (boolean of whether the network should be stopped early,
                the updated count)
    """
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0
    if count == patience:
        return True, count
    else:
        return False, count
