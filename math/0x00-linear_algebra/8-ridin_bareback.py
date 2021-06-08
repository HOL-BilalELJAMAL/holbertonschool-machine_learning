#!/usr/bin/env python3
"""
8-ridin_bareback.py
Module that defines a function that performs matrices multiplication
"""


def mat_mul(mat1, mat2):
    """
    Function that performs matrices multiplication

    Args:
        mat1 (list of list): Matrix 1 of integers/floats
        mat2 (list of list): Matrix 2 of integers/floats
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(a * b for a, b in zip(X_row, Y_col))
             for Y_col in zip(*mat2)] for X_row in mat1]
