#!/usr/bin/env python3
"""
4-brightness.py
Module that defines a function called change_brightness
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Function that randomly changes the brightness of an image

    Args:
        image [3D td.Tensor]:
            contains the image to change
        max_delta [float]:
            maximum amount the image should be brightened (or darkened)

    Returns:
        the altered image
    """
    return tf.image.random_brightness(image, max_delta)
