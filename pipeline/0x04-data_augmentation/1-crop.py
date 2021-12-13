#!/usr/bin/env python3
"""
1-crop.py
Module that defines a function called crop_image
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Function that performs a random crop of an image

    Args:
        image [3D td.Tensor]:
            contains the image to crop
        size [tuple]:
            contains the size of the crop

    Returns:
        the cropped image
    """
    return tf.image.random_crop(image, crop_size=size)
