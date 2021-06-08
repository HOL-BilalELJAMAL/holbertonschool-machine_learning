#!/usr/bin/env python3
"""
9-sum_total.py
Module that defines a functiont hat calculates sigma summation
"""


def summation_i_squared(n):
    """
    Function that calculates sum from i=1 to n of i^2:

    Args:
        n (int): is the stopping condition

    Returns:
        sum (int): for success,
        None (): In case n is not valid.
    """
    sum = 0
    if n is None or not isinstance(n, int) or n < 1:
        return None
    else:
        return int(n * (n + 1) * (2 * n + 1) / 6)
