#!/usr/bin/env python3
"""
2-from_file.py
Module that defines a function called from_file
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Function that loads data from a file as a Pandas DataFrame

    Args:
        filename [str]: file to load the data from
        delimiter [str]: the column separator

    Returns:
        The newly created pd.DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
