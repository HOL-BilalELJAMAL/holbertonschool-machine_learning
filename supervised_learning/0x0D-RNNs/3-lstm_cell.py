#!/usr/bin/env python3
"""
3-lstm_cell.py
Module that defines a class called LSTMCell
"""

import numpy as np


class LSTMCell:
    """
    Class LSTMCell
    """
    def __init__(self, i, h, o):
        """
        Class Constructor

        Args:
            i: Dimensionality of the data
            h: Dimensionality of the hidden state
            o: Dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Function that performs forward propagation for one time step

        Args:
            x_t: numpy.ndarray of shape (m, i) that contains the data
            input for the cell where m is the batch size for the data
            h_prev: a numpy.ndarray of shape (m, h) containing the
            previous hidden state
            c_prev: a numpy.ndarray of shape (m, h) containing the previous
            cell state

        Returns:
            h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        x = np.concatenate((h_prev, x_t), axis=1)
        f = np.matmul(x, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f))
        u = np.matmul(x, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))
        c = np.matmul(x, self.Wc) + self.bc
        c = np.tanh(c)
        c_next = (u * c) + (f * c_prev)
        o = np.matmul(x, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))
        h_next = o * np.tanh(c_next)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / (np.sum(np.exp(y), axis=1, keepdims=True))
        return h_next, c_next, y
