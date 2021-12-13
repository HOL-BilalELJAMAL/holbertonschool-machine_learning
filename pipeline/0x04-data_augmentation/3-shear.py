#!/usr/bin/env python3
"""
3-shear.py
Module that defines a function called shear_image
"""

import tensorflow as tf


def shear_image(image, intensity):
    """
    Function that shears an image

    Args:
        image [3D td.Tensor]:
            contains the image to shear
        intensity [int]:
            intensity with which the image should be sheared

    Returns:
        The sheared image
    """
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    shear_array = tf.keras.preprocessing.image.random_shear(image_array,
                                                            intensity)
    image_result = tf.keras.preprocessing.image.array_to_img(shear_array)
    return image_result
