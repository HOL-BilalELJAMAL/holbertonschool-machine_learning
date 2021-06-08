#!/usr/bin/env python3
"""
0-neuron.py
Module that defines a class called Neuron
"""
import numpy as np


class Neuron:
    """Neuron Class"""
    def __init__(self, nx):
        """Function init of the Neuron Class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
