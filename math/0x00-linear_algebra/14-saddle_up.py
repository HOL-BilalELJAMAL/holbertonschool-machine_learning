#!/usr/bin/env python3
"""
14-saddle_up.py
Module that defines a function that performs matrix multiplication
using numpy
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Function that performs matrix multiplication using numpy

    Args:
         mat1 (list of list): Matrix 1 of integers/floats
         mat2 (list of list): Matrix 2 of integers/float
    """
    return np.dot(mat1, mat2)
