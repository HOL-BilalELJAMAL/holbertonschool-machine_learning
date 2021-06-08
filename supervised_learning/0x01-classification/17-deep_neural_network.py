#!/usr/bin/env python3
"""
17-deep_neural_network.py
Module that defines a class called DeepNeuralNetwork
"""
import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork Class"""
    def __init__(self, nx, layers):
        """Function init of the DeepNeuralNetwork Class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layer = 1
        layer_size = nx
        for i in layers:
            if not isinstance(i, int) or i <= 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W" + str(layer)
            b = "b" + str(layer)
            self.__weights[w] = np.random.randn(
                i, layer_size) * np.sqrt(2/layer_size)
            self.__weights[b] = np.zeros((i, 1))
            layer += 1
            layer_size = i

    @property
    def L(self):
        """
        Getter property to get the number of layers of the deep neural network
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter property to get intermediary values of the deep neural network
        """
        return self.__cache

    @property
    def weights(self):
        """Getter property to get the weights of the deep neural network"""
        return self.__weights
