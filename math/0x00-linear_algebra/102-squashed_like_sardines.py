#!/usr/bin/env python3
"""
102-squashed_like_sardines.py
Module that defines a function that concatenates two matrices along
a specific axis
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis

    Args:
        mat1 (list of list): Matrix 1 of integers/floats
        mat2 (list of list): Matrix 2 of integers/floats
        axis (int): Axis concatenation
    """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    del shape1[axis]
    del shape2[axis]
    return None if shape1 != shape2 else cat_matrices_comp(mat1, mat2, axis)


def cat_matrices_comp(mat1, mat2, axis):
    """
    Recursive function that concatenates two matrices along a specific axis

    Args:
        mat1 (list of list): Matrix 1 of integers/floats
        mat2 (list of list): Matrix 2 of integers/floats
        axis (int): Axis concatenation
    """
    if axis != 0:
        return [cat_matrices_comp(u, v, axis - 1) for u, v in zip(mat1, mat2)]
    return matrix_copy(mat1 + mat2)


def matrix_copy(matrix):
    """
    Function that returns a deep copy of the matrix

    Args:
        matrix (list of list): Matrix of integers/floats
    """
    if len(matrix) != 0 and isinstance(matrix[0], list):
        return list(map(matrix_copy, matrix))
    return matrix[:]


def matrix_shape(matrix):
    """
    Function that returns the shape of a matrix

    Args:
        matrix (list of list): Matrix of integers/floats
    """
    shape = [len(matrix)]
    if len(matrix) != 0 and isinstance(matrix[0], list):
        shape += matrix_shape(matrix[0])
    return shape
