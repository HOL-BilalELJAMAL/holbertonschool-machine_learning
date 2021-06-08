#!/usr/bin/env python3
"""
2-convolve_grayscale_padding.py
Module that defines a function called convolve_grayscale_padding
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a convolution on grayscale images with padding

    Args:
        images (numpy.ndarray) with shape (m, h, w) containing multiple
        grayscale images:
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel (numpy.ndarray) with shape (kh, kw) containing the kernel for
        the convolution:
            kh is the height of the kernel
            kw is the width of the kernel
        padding (tuple): tuple of (ph, pw):
            ph is the padding for the height of the image
            pw is the padding for the width of the image

    Returns:
        numpy.ndarray containing the convolved images
    """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]
    ph = padding[0]
    pw = padding[1]
    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)
    new_h = int(images_padded.shape[1] - kh + 1)
    new_w = int(images_padded.shape[2] - kw + 1)
    output = np.zeros((m, new_h, new_w))
    for x in range(new_w):
        for y in range(new_h):
            output[:, y, x] = \
                (kernel * images_padded[:,
                                        y: y + kh,
                                        x: x + kw]).sum(axis=(1, 2))
    return output
