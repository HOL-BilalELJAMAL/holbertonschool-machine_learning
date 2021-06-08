#!/usr/bin/env python3
"""
13-cats_got_your_tongue.py
Module that defines a function that concatenates two matrices along a
specific axis using numpy
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis
    using numpy

    Args:
         mat1 (list of list): Matrix 1 of integers/floats
         mat2 (list of list): Matrix 2 of integers/float
         axis (int): Axis concatenation
    """
    return np.concatenate((mat1, mat2), axis=axis)
