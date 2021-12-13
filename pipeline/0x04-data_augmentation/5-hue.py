#!/usr/bin/env python3
"""
5-hue.py
Module that defines a function called change_hue
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Function that changes the hue of an image

    Args:
        image [3D td.Tensor]:
            contains the image to change
        delta [float]:
            the amount the hue should change

    Returns:
        The altered image
    """
    return tf.image.adjust_hue(image, delta)
