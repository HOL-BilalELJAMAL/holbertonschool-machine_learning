#!/usr/bin/env python3
"""
1-convolve_grayscale_same.py
Module that defines a function called convolve_grayscale_same
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function that performs a same convolution on grayscale images

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

    Returns:
        numpy.ndarray containing the convolved images
    """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]
    ph = max(int((kh - 1) / 2), int(kh / 2))
    pw = max(int((kw - 1) / 2), int(kw / 2))
    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)
    output = np.zeros((m, h, w))
    for y in range(h):
        for x in range(w):
            output[:, y, x] =\
                (kernel * images_padded[:,
                                        y: y + kh,
                                        x: x + kw]).sum(axis=(1, 2))
    return output
