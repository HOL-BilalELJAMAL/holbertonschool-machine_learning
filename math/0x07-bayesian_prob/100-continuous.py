#!/usr/bin/env python3
"""
100-continuous.py
Module that defines a function called posterior
"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Function that calculates the posterior probability that the probability
    of developing severe side effects falls within a specific range
    given the data.

    Args:
        x (int, float): Number of patients that develop severe side effects
        n (int, float): Total number of patients observed
        p1 is the lower bound on the range
        p2 is the upper bound on the range

    Returns:
        The posterior probability that p is within the range [p1, p2]
        given x and n
    """

    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or np.any(p1 > 1) or np.any(p1 < 0):
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or np.any(p2 > 1) or np.any(p2 < 0):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    f1 = x + 1
    f2 = n - x + 1
    ac1 = special.btdtr(f1, f2, p1)
    ac2 = special.btdtr(f1, f2, p2)
    return ac2 - ac1
