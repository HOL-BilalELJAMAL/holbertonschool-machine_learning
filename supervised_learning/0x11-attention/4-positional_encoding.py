#!/usr/bin/env python3
"""
4-positional_encoding.py
Module that defines a function called positional_encoding
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Function that computes the positional encoding vector"""
    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :],
                            dm)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding[0]


def get_angles(position, i, dm):
    """Function that computes the angles for the positional encoding"""
    angle_rates = 1 / np.power(10000, (2 * np.floor(i / 2)) / np.float32(dm))
    return position * angle_rates
