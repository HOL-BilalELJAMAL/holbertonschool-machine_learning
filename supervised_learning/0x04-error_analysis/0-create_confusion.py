#!/usr/bin/env python3
"""
0-create_confusion.py
Module that defines a function called create_confusion_matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Function that creates a confusion matrix

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
                containing the correct labels for each data point
        logits: one-hot numpy.ndarray of shape (m, classes)
                containing the predicted labels

    Return:
        Confusion numpy.ndarray of shape (classes, classes)
        with row indices representing the correct labels and column indices
        representing the predicted labels
    """
    return np.matmul(labels.T, logits)
