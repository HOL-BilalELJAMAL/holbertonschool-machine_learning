#!/usr/bin/env python3
"""
4-deep_rnn.py
Module that defines a function called deep_rnn
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Function that performs forward propagation for a deep RNN

    Args:
        rnn_cells is a list of RNNCell instances of length l that
        will be used for the forward propagation
        l is the number of layers
        X is the data to be used, given as a numpy.ndarray of shape
        (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
        h_0 is the initial hidden state, given as a numpy.ndarray
        of shape (l, m, h)
        h is the dimensionality of the hidden state

    Returns:
        H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, m, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    for i in range(t):
        h_aux = X[i]
        for j in range(len(rnn_cells)):
            r_cell = rnn_cells[j]
            x_t = h_aux
            h_prev = H[i][j]
            h_next, y_next = r_cell.forward(h_prev, x_t)
            h_aux = h_next
            H[i + 1][j] = h_aux
        Y[i] = y_next
    return H, Y
