#!/usr/bin/env python3
"""
28-deep_neural_network.py
Module that defines a class called DeepNeuralNetwork
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """DeepNeuralNetwork Class"""
    def __init__(self, nx, layers, activation='sig'):
        """Function init of the DeepNeuralNetwork Class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation != "sig" and activation != "tanh":
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
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

    @property
    def activation(self):
        """Getter property to get the activation of the deep neural network"""
        return self.__activation

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
            if i != self.__L:
                if self.__activation == "sig":
                    self.__cache[a_new] = 1 / (1 + np.exp(-Z))
                else:
                    self.__cache[a_new] = (
                        np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            else:
                t = np.exp(Z)
                a_new = "A" + str(i)
                self.__cache[a_new] = t / t.sum(axis=0, keepdims=True)
        act_func = "A" + str(self.__L)
        return (self.__cache[act_func], self.__cache)

    def cost(self, Y, A):
        """
        Function that calculates the cost function of the deep neural network
        using logistic regression
        """
        cost_array = np.log(A) * Y
        cost = -np.sum(cost_array)/len(A[0])
        return cost

    def evaluate(self, X, Y):
        """Function that evaluates the predictions of the neural network"""
        A, _ = self.forward_prop(X)
        cost_r = self.cost(Y, A)
        decode = np.amax(A, axis=0)
        return (np.where(A == decode, 1, 0), cost_r)

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
            if self.__activation == "sig":
                dz = np.matmul(weights_copy["W" + str(i)].T, dz) * (
                    A * (1 - A))
            else:
                dz = np.matmul(weights_copy["W" + str(i)].T, dz) * (1 - A * A)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Function that performs iterations training process"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        _, cache = self.forward_prop(X)
        cost_list = []
        iter_x = []
        for i in range(iterations + 1):
            A, cost = self.evaluate(X, Y)
            if verbose is True and (
                    i % step == 0 or i == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))
                cost_list.append(cost)
                iter_x.append(i)
            if i != iterations:
                self.gradient_descent(Y, cache, alpha)
                _, cache = self.forward_prop(X)
        if graph is True:
            plt.plot(iter_x, cost_list)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return A, cost

    def save(self, filename):
        """
        Function that saves the DeepNeuralNetwork instance object to a file
        in pickle format
        """
        if filename.split(".")[-1] != "pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as fileObject:
            pickle.dump(self, fileObject)

    def load(filename):
        """Function that loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as file_object:
                b = pickle.load(file_object)
                return b
        except (OSError, IOError) as e:
            return None
