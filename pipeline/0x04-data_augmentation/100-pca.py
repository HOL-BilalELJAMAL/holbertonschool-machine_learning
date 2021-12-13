#!/usr/bin/env python3
"""
100-pca.py
Module that defines a function called pca_color
"""

import tensorflow as tf


def pca_color(image, alphas):
    """
    Function that performs PCA color augmentation on an image

    Args:
        image [3D td.Tensor]:
            contains the image to change
        alphas [tuple of length 3]:
            contains the amount that each channel should change

    Returns:
        The augmented image
    """
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    img = image_array.reshape(-1, 3).astype(float)
    return img
