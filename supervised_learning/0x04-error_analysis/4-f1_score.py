#!/usr/bin/env python3
"""
4-f1_score.py
Module that defines a function called f1_score
"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Function that calculates the F1 score of a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
        represent the correct labels and column indices represent the predicted
        labels

    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score of each class
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * prec * sens / (prec + sens)
