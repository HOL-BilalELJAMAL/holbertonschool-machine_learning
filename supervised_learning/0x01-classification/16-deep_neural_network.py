#!/usr/bin/env python3
"""
16-deep_neural_network.py
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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        layer = 1
        layer_size = nx
        for i in layers:
            if not isinstance(i, int) or i <= 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W" + str(layer)
            b = "b" + str(layer)
            self.weights[w] = np.random.randn(
                i, layer_size) * np.sqrt(2/layer_size)
            self.weights[b] = np.zeros((i, 1))
            layer += 1
            layer_size = i
