#!/usr/bin/env python3
"""
8-bi_rnn.py
Module that defines a function called bi_rnn
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Function that performs forward propagation for a bidirectional RNN

    Args:
        bi_cell is an instance of BidirectinalCell that will be used for the
        forward propagation
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
        h_0 is the initial hidden state in the forward direction, given as a
        numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
        h_t is the initial hidden state in the backward direction, given as a
        numpy.ndarray of shape (m, h)

    Returns:
        H, Y
        H is a numpy.ndarray containing all of the concatenated hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_t.shape[1]
    h_pev, h_ant = np.zeros((t, m, h)), np.zeros((t, m, h))
    H_p, H_a = h_0, h_t
    for i in range(t):
        x_pev = X[i]
        x_ant = X[-(i + 1)]
        H_p = bi_cell.forward(H_p, x_pev)
        H_a = bi_cell.backward(H_a, x_ant)
        h_pev[i] = H_p
        h_ant[-(i + 1)] = H_a
    H = np.concatenate((h_pev, h_ant), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
