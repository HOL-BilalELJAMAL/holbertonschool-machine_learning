#!/usr/bin/env python3
"""
2-conv_backward.py
Module that defines a function called conv_backward
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Function that performs back propagation over a convolutional layer
    of a neural network
    Args:
        dZ (np.ndarray): matrix of shape (m, h_new, w_new, c_new) containing
                         the partial derivatives with respect to the
                         unactivated output of the convolutional layer
        A_prev (np.ndarray): matrix of shape (m, h_prev, w_prev, c_prev)
                             containing the output of the previous layer
        W (np.ndarray): matrix of shape (kh, kw, c_prev, c_new) containing the
                        kernels for the convolution
        b (np.ndarray): matrix of shape (1, 1, 1, c_new) containing the biases
                        applied to the convolution
        padding (str): same or valid, indicating the type of padding used.
        stride (tuple): (sh, sw) containing the strides for the convolution
    Returns:
        The partial derivatives with respect to the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    pw, ph = 0, 0

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = int(np.ceil(((h_prev-1)*sh+kh-h_prev)/2))
        pw = int(np.ceil(((w_prev-1)*sw+kw-w_prev)/2))

    A_prev = np.pad(A_prev,
                    pad_width=((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                    mode='constant', constant_values=0)

    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)

    for z in range(m):
        for y in range(h_new):
            for x in range(w_new):
                for k in range(c_new):
                    aux_W = W[:, :, :, k]
                    aux_dz = dZ[z, y, x, k]
                    dA[z, y*sh: y*sh+kh, x*sw: x*sw+kw, :] += aux_dz * aux_W
                    aux_A_prev = A_prev[z, y*sh: y*sh+kh, x*sw: x*sw+kw, :]
                    dW[:, :, :, k] += aux_A_prev * aux_dz

    dA = dA[:, ph:dA.shape[1] - ph, pw:dA.shape[2] - pw, :]

    return dA, dW, db

