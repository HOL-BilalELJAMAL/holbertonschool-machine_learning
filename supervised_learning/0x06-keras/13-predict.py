#!/usr/bin/env python3
"""
13-predict.py
Module that defines a function called predict
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Function that makes a prediction using a neural network

    Args:
        network (Keras model): network model to make the prediction with
        data (np.ndarray): input data to make the prediction with
        verbose (bool): determines if output should be printed during the
                        prediction process

    Returns:
        The prediction for the data
    """
    return network.predict(data, verbose=verbose)
