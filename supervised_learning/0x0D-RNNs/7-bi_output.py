#!/usr/bin/env python3
"""
7-bi_output.py
Module that defines a class called BidirectionalCell
"""

import numpy as np


class BidirectionalCell:
    """
    Class BidirectionalCell
    """
    def __init__(self, i, h, o):
        """
        Class Constructor

        Args:
            i: Dimensionality of the data
            h: Dimensionality of the hidden state
            o: Dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
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
            h_next the next hidden state
        """
        x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(x, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Function that performs backward propagation for one time step

        Args:
            x_t: numpy.ndarray of shape (m, i) that contains the data
            input for the cell where m is the batch size for the data
            h_next: a numpy.ndarray of shape (m, h) containing the
            next hidden state

        Returns:
            h_pev the previous hidden state
        """
        x = np.concatenate((h_next, x_t), axis=1)
        h_pev = np.tanh(np.matmul(x, self.Whb) + self.bhb)
        return h_pev

    def output(self, H):
        """
        Function that calculates all outputs for the RNN

        Args:
            H: numpy.ndarray of shape (t, m, 2 * h) that contains the
            concatenated hidden states from both directions, excluding their
            initialized states
            t is the number of time steps
            m is the batch size for the data
            h is the dimensionality of the hidden states

        Returns:
            Y the outputs
        """
        t, m, h = H.shape
        o = self.by.shape[-1]
        Y = np.zeros((t, m, o))
        for i in range(t):
            Y[i] = np.matmul(H[i], self.Wy) + self.by
            Y[i] = np.exp(Y[i]) / (np.sum(np.exp(Y[i]), axis=1, keepdims=True))
        return Y
