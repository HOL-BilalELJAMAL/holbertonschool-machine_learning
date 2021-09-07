#!/usr/bin/env python3
"""
2-gru_cell.py
Module that defines a class called GRUCell
"""

import numpy as np


class GRUCell:
    """
    Class GRUCell
    """
    def __init__(self, i, h, o):
        """
        Class Constructor

        Args:
            i: Dimensionality of the data
            h: Dimensionality of the hidden state
            o: Dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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
        z = np.matmul(x, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))
        r = np.matmul(x, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))
        x = np.concatenate((r * h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(x, self.Wh) + self.bh)
        h = z * h_next + (1 - z) * h_prev
        y = np.matmul(h, self.Wy) + self.by
        y = (np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True))
        return h, y
