#!/usr/bin/env python3
"""
2-precision.py
Module that defines a function called precision
"""

import numpy as np


def precision(confusion):
    """
    Function that calculates the precision for each class in a confusion
    matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
        represent the correct labels and column indices represent the predicted
        labels

    Returns:
        numpy.ndarray of shape (classes,) containing the precision of each
        class
    """
    return np.divide(confusion.diagonal(), np.sum(confusion, axis=0))
