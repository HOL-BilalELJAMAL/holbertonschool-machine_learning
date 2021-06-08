#!/usr/bin/env python3
"""
1-neuron.py
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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter property to get the weights of the neuron"""
        return self.__W

    @property
    def b(self):
        """Getter property to get the biases of the neuron"""
        return self.__b

    @property
    def A(self):
        """Getter property to get the output of the neuron"""
        return self.__A
