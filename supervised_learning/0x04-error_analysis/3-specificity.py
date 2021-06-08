#!/usr/bin/env python3
"""
3-specificity.py
Module that defines a function called specificity
"""

import numpy as np


def specificity(confusion):
    """
    Function that calculates the specificity for each class in a confusion
    matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
        represent the correct labels and column indices represent the predicted
        labels

    Returns:
        numpy.ndarray of shape (classes,) containing the precision of each
        class
    """
    tot = np.sum(confusion)
    tp = np.diagonal(confusion)
    act = np.sum(confusion, axis=1)
    prd = np.sum(confusion, axis=0)
    fp = prd - tp
    an = tot - act
    tn = an - fp
    return tn / an
