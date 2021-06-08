#!/usr/bin/env python3
"""
7-neuron.py
Module that defines a class called Neuron
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """
        Function that calculates the forward propagation of the neuron
        using sigmoid as activation function
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Function that calculates the cost function of the neuron
        model using logistic regression
        """
        cost = np.multiply(np.log(A), Y) + np.multiply((
            1 - Y), np.log(1.0000001 - A))
        return -np.sum(cost) / len(A[0])

    def evaluate(self, X, Y):
        """Function that evaluates the predictions of the neurons"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return np.where(self.__A > 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Function that calculates one pass of gradient descent"""
        n = len(Y[0])
        dz = A - Y
        dw = np.matmul(dz, X.T) / n
        db = np.sum(dz) / n
        self.__W += - alpha * dw
        self.__b += - alpha * db

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """Function that performs iterations training process"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        iters = []
        for i in range(iterations + 1):
            A, cost = self.evaluate(X, Y)
            if i != iterations:
                self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A, alpha)
            if (i % step == 0 or i == 0 or i == iterations) and verbose:
                print("Cost after {} iterations: {}".format(i, cost))
                costs.append(cost)
                iters.append(i)
        if graph is True:
            plt.plot(iters, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.xlim(0, iterations)
            plt.show()
        return A, cost
