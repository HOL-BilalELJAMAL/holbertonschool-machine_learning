#!/usr/bin/env python3
"""
5-convolve.py
Module that defines a function called convolve
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images with channels

    Args:
        images (numpy.ndarray) with shape (m, h, w, c) containing multiple
        grayscale images:
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernels (numpy.ndarray) with shape (kh, kw, c, nc) containing the
        kernels for the convolution:
            kh is the height of the kernel
            kw is the width of the kernel
            c is the number of channels in the image
            nc is the number of kernels
        padding: either a tuple of (ph, pw), 'same', or 'valid'
            if 'same', performs a same convolution
            if 'valid', performs a valid convolution
            if a tuple: tuple of (ph, pw):
                ph is the padding for the height of the image
                pw is the padding for the width of the image
        stride (tuple): tuple of (sh, sw):
            sh is the stride for the height of the image
            sw is the stride for the width of the image

    Returns:
        numpy.ndarray containing the convolved images
    """
    c, w, = images.shape[3], images.shape[2]
    h, m = images.shape[1], images.shape[0]
    nc, kw, kh = kernels.shape[3], kernels.shape[1], kernels.shape[0]
    sw, sh = stride[1], stride[0]
    pw, ph = 0, 0
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]
    images = np.pad(images,
                    pad_width=((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                    mode='constant', constant_values=0)
    new_h = int(((h + 2 * ph - kh) / sh) + 1)
    new_w = int(((w + 2 * pw - kw) / sw) + 1)
    output = np.zeros((m, new_h, new_w, nc))
    for y in range(new_h):
        for x in range(new_w):
            for v in range(nc):
                output[:, y, x, v] = \
                    (kernels[:, :, :, v] *
                     images[:,
                     y * sh: y * sh + kh,
                     x * sw: x * sw + kw,
                     :]).sum(axis=(1, 2, 3))
    return output
