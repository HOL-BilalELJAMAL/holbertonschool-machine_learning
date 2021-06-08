#!/usr/bin/env python3
"""
1-sensitivity.py
Module that defines a function called sensitivity
"""

import numpy as np


def sensitivity(confusion):
    """
    Function that calculates the sensitivity for each class in a confusion
    matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
        represent the correct labels and column indices represent the predicted
        labels

    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity of each
        class
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=1)
