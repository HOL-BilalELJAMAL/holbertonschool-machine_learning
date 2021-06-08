#!/usr/bin/env python3
"""
3-flip_me_over.py
Module that defines a function that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Function that returns the transpose of a 2D matrix

    Args:
         matrix (list of list): Matrix
    """
    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
