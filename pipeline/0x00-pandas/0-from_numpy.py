#!/usr/bin/env python3
"""
0-from_numpy.py
Module that defines a function called from_numpy
"""

import pandas as pd


def from_numpy(array):
    """
    Function that creates a Pandas DataFrame from a numpy.ndarray

    Args:
        array [numpy.ndarray]: array to create pd.DataFrame from
        columns of the DataFrame should be labeled in alphabetical
        order and capitalized (there will not be more than 26 columns)

    Returns:
        The newly created pd.DataFrame
    """
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
                "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                "S", "T", "U", "V", "W", "X", "Y", "Z"]
    column_labels = []
    for i in range(len(array[0])):
        column_labels.append(alphabet[i])
    df = pd.DataFrame(array, columns=column_labels)
    return df
