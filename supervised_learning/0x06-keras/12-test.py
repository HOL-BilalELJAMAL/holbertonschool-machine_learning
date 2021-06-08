#!/usr/bin/env python3
"""
12-test.py
Module that defines a function called test_model
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Function that that a neural network

    Args:
        network (Keras model): the network model to test
        data (np.ndarray): input data to test the model with
        labels (np.ndarray): correct one-hot labels of data
        verbose (bool): determines if output should be printed during the
                        testing process

    Returns:
        The loss and accuracy of the model with the testing data respectively
    """
    return network.evaluate(data, labels, verbose=verbose)
