#!/usr/bin/env python3
"""
24-one_hot_decode.py
Module that defines a function called one_hot_decode
"""
import numpy as np


def one_hot_decode(one_hot):
    """Functio that converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray) or len(one_hot) == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
