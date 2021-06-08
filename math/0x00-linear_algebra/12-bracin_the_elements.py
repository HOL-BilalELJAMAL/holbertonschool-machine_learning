#!/usr/bin/env python3
"""
12-bracin_the_elements.py
Module that defines a function that performs element-wise addition,
subtraction, multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """
    Function that performs element-wise addition, subtraction,
    multiplication, and division

    Args:
         mat1 (list of list): Matrix 1 of integers/floats
         mat2 (list of list): Matrix 2 of integers/float
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
