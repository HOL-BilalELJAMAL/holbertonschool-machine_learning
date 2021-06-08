#!/usr/bin/env python3
"""
4-line_up.py
Module that defines a function that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Function that adds two arrays element-wise

    Args:
         arr1 (list): List 1 of integers/floats
         arr2 (list): List 2 of integers/float
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
