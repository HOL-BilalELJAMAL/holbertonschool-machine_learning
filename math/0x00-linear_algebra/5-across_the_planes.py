#!/usr/bin/env python3
"""
5-across_the_planes.py
Module that defines a function that adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Function that adds two matrices element-wise

    Args:
         mat1 (list of list): Matrix 1 of integers/floats
         mat2 (list of list): Matrix 2 of integers/float
    """
    if len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat2))]
