#!/usr/bin/env python3
"""
4-train.py
Module that defines a function called train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """
    Function that trains a model using mini-batch gradient descent

    Args:
        network (keras model): model to train
        data (np.ndarray): matrix of shape (m, nx) containing the input data.
        labels (np.ndarray): one hot matrix of shape (m, classes) containing
                             the labels of data
        batch_size (int): size of the batch used for mini-batch gradient
                          descent
        epochs (int): number of passes through data for mini-batch gradient
                      descent
        verbose (bool): determines if output should be printed during training
        shuffle (bool): determines whether to shuffle the batches every epoch

    Returns:
        The History object generated after training the model
    """
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle)
