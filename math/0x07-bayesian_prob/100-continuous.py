#!/usr/bin/env python3
"""
100-continuous.py
Module that defines a functions called likelihood, intersection, marginal
and posterior
"""
import numpy as np


def likelihood(x, n, P):
    """
    Function that finds the probability that a patient who takes this
    drug will develop severe side effects.

    Args:
        x (int, float): Number of patients that develop severe side effects
        n (int, float): Total number of patients observed
        P (1D numpy.ndarray): Containing the various hypothetical probabilities
        of developing severe side effects

    Returns:
        1D numpy.ndarray with the likelihood of obtaining the data,
        x and n, for each probability in P, respectively
    """
    comb = np.math.factorial(n)/(np.math.factorial(x) * np.math.factorial(n-x))
    likelihood = comb * pow(P, x) * pow(1 - P, n - x)
    return likelihood


def intersection(x, n, P, Pr):
    """
    Function that calculates the intersection of obtaining this data with
    the various hypothetical probabilities.

    Args:
        x (int, float): Number of patients that develop severe side effects
        n (int, float): Total number of patients observed
        P (1D numpy.ndarray): Containing the various hypothetical probabilities
        of developing severe side effects
        Pr (1D numpy.ndarray'): Containing the prior beliefs of P

    Returns:
        1D numpy.ndarray containing the intersection of obtaining x and n
        with each probability in P, respectively
    """
    likelihoods = likelihood(x, n, P)
    intersection = likelihoods * Pr
    return intersection


def marginal(x, n, P, Pr):
    """
    Function that calculates the marginal probability of obtaining the data.

    Args:
        x (int, float): Number of patients that develop severe side effects
        n (int, float): Total number of patients observed
        P (1D numpy.ndarray): Containing the various hypothetical probabilities
        of developing severe side effects
        Pr (1D numpy.ndarray'): Containing the prior beliefs of P

    Returns:
        The marginal probability of obtaining x and n
    """
    intersections = intersection(x, n, P, Pr)
    marginal = np.sum(intersections)
    return marginal


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

    Pr = x / n

    return 1
