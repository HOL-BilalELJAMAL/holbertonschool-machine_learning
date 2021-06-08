#!/usr/bin/env python3
"""
24-one_hot_encode.py
Module that defines a function called one_hot_encode
"""
import numpy as np


def one_hot_encode(Y, classes):
    """Funcion that converts a numeric label vector into a one-hot matrix"""
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None
    return np.eye(classes)[Y].T
