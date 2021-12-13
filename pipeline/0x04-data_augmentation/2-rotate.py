#!/usr/bin/env python3
"""
2-rotate.py
Module that defines a function called rotate_image
"""

import tensorflow as tf


def rotate_image(image):
    """
    Function that rotates an image 90 degrees counter-clockwise

    Args:
        image [3D td.Tensor]:
            contains the image to rotate

    Returns:
        the rotated image
    """
    return tf.image.rot90(image)
