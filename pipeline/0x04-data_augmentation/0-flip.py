#!/usr/bin/env python3
"""
0-flip.py
Module that defines a function called flip_image
"""

import tensorflow as tf


def flip_image(image):
    """
    Function that flips an image horizontally

    Args:
        image [3D td.Tensor]:
            contains the image to flip

    Returns:
        The flipped image
    """
    return tf.image.flip_left_right(image)
