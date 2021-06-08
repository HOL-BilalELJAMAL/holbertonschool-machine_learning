#!/usr/bin/env python3
"""
7-gettin_cozy.py
Module that defines a function that concatenates two matrices along
a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis

    Args:
        mat1 (list of list): Matrix 1 of integers/floats
        mat2 (list of list): Matrix 2 of integers/floats
        axis (int): Axis concatenation
    """
    if axis == 1 and (len(mat1) == len(mat2)):
        return [mat1[index]+mat2[index] for index, item in enumerate(mat2)]
    if axis == 0 and (len(mat1[0]) == len(mat2[0])):
        return [arr1[:] for arr1 in mat1] + [arr2[:] for arr2 in mat2]
