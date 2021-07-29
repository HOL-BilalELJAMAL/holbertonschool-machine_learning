#!/usr/bin/env python3
"""
2-marginal.py
Module that defines a functions called likelihood, intersection and marginal
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

    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    intersections = intersection(x, n, P, Pr)
    marginal = np.sum(intersections)

    return marginal
