#!/usr/bin/env python3
"""
101-the_whole_barn.py
Module that defines a function that adds two matrices
"""


def get_length(rows):
    """
    Recursive function that calculates the length of matrix rows

    Args:
        rows (list or tuple): Matrix rows
    """
    if rows and (isinstance(rows, list) or isinstance(rows, tuple)):
        return [len(rows), *get_length(rows[0])]
    return []


def matrix_shape(matrix):
    """
    Function that calculates the matrix shape

    Args:
        matrix (list or list): Matrix of integers/floats
    """
    return [*get_length(matrix)]


def add_rows(row1, row2):
    """
    Function that add numbers on the same row level recursively

    Args:
        row1 (list): Matrix row 1
        row2 (list): Matrix row 2
    """
    if isinstance(row1[0], list):
        return [add_rows(row1[i], row2[i]) for i in range(len(row1))]
    return [row1[i] + row2[i] for i in range(len(row1))]


def add_matrices(mat1, mat2):
    """
    Function that adds two matrices

    Args:
         mat1 (list of list): Matrix 1 of integers/floats
         mat2 (list of list): Matrix 2 of integers/float
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    return add_rows(mat1, mat2)
