#!/usr/bin/env python3
"""
21-deep_neural_network.py
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

    def forward_prop(self, X):
        """
        Function that calculates the forward propagation of the deep neural
        network using sigmoid as activation function
        """
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            w = "W" + str(i)
            b = "b" + str(i)
            a = "A" + str(i - 1)
            Z = np.matmul(self.__weights[w],
                          self.__cache[a]) + self.__weights[b]
            a_new = "A" + str(i)
            self.__cache[a_new] = 1 / (1 + np.exp(-Z))
        act_fnc = "A" + str(self.__L)
        return self.__cache[act_fnc], self.__cache

    def cost(self, Y, A):
        """
        Function that calculates the cost function of the deep neural network
        using logistic regression
        """
        cost = np.multiply(np.log(A), Y) + np.multiply((
            1 - Y), np.log(1.0000001 - A))
        return -np.sum(cost) / len(A[0])

    def evaluate(self, X, Y):
        """Function that evaluates the predictions of the neural network"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A > 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Function that calculates one pass of gradient descent"""
        n = len(Y[0])
        weights_copy = self.__weights.copy()
        dz = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A = cache["A" + str(i - 1)]
            dw = np.matmul(dz, A.T) / n
            db = np.sum(dz, axis=1, keepdims=True) / n
            w = "W" + str(i)
            b = "b" + str(i)
            self.__weights[w] = self.__weights[w] - alpha * dw
            self.__weights[b] = self.__weights[b] - alpha * db
            dz = np.matmul(weights_copy["W" + str(i)].T, dz) * A * (1 - A)
