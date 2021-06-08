#!/usr/bin/env python3
"""
9-neural_network
Module that defines a class called NeuralNetwork
"""
import numpy as np


class NeuralNetwork:
    """NeuralNetwork Class"""
    def __init__(self, nx, nodes):
        """Function init of the NeuronNetwork Class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter property to get the weight 1 of the neural network"""
        return self.__W1

    @property
    def b1(self):
        """Getter property to get the bias 1 of the neural network"""
        return self.__b1

    @property
    def A1(self):
        """Getter property to get the output 1 of the neuron network"""
        return self.__A1

    @property
    def W2(self):
        """Getter property to get the weight 2 of the neural network"""
        return self.__W2

    @property
    def b2(self):
        """Getter property to get the bias 2 of the neural network"""
        return self.__b2

    @property
    def A2(self):
        """Getter property to get the output 2 of the neuron network"""
        return self.__A2
