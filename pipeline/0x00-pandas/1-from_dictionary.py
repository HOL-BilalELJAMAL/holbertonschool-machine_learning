#!/usr/bin/env python3
"""
1-from_dictionary.py
Module that defines a function called from_dictionary
"""

import pandas as pd


def from_dictionary():
    """
    Function that creates a Pandas DataFrame from a dictionary

    Returns:
        the newly created pd.DataFrame
    """
    df = pd.DataFrame(
        {
            "First": [0.0, 0.5, 1.0, 1.5],
            "Second": ["one", "two", "three", "four"]
        },
        index=list("ABCD"))
    return df


df = from_dictionary()
