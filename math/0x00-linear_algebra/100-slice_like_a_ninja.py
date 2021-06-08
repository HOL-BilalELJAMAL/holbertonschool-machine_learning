#!/usr/bin/env python3
"""
100-slice_like_a_ninja.py
Module that defines a function that slices a matrix along a specific axes
"""


def np_slice(matrix, axes={}):
    """
    Function that slices a matrix along a specific axes

    Args:
        matrix (list of list): Matrix
        axes (dict): Axis key and Slice value
    """
    h_axis = max(axes, key=int) + 1
    slice_obj = tuple([slice(*axes.get(i) or (None,)) for i in range(h_axis)])
    return matrix[slice_obj]
