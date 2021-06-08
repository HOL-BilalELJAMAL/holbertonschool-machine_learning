#!/usr/bin/env python3
"""
2-size_me_please.py
Module that defines a function that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Function that calculates the shape of a matrix

    Args:
         matrix (list of list): Matrix
    """
    my_list = [len(matrix)]
    while not isinstance(matrix[0], int):
        my_list.append(len(matrix[0]))
        matrix = matrix[0]
    return my_list
