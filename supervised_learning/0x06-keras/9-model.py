#!/usr/bin/env python3
"""
9-model.py
Module that defines 2 functions called save_model, load_model
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Function that saves an entire model

    Args:
        network: model to save
        filename: the path of the file that the model should be saved to

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Function that loads an entire model

    Args:
        filename: the path of the file that the model should be loaded from

    Returns:
        the loaded model
    """
    return K.models.load_model(filename)
