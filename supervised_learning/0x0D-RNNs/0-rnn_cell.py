#!/usr/bin/env python3
"""
0-rnn_cell.py
Module that defines a class called RNNCell
"""

import numpy as np


class RNNCell:
    """
    RNNCell Class
    """
    def __init__(self, i, h, o):
        """
        Class Constructor

        Args:
            i: Dimensionality of the data
            h: Dimensionality of the hidden state
            o: Dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Function that performs forward propagation for one time step

        Args:
            x_t: numpy.ndarray of shape (m, i) that contains the data
            input for the cell where m is the batch size for the data
            h_prev: a numpy.ndarray of shape (m, h) containing the
            previous hidden state

        Returns:
            h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(x, self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / (np.sum(np.exp(y), axis=1, keepdims=True))
        return h_next, y
