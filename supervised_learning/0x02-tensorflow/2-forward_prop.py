#!/usr/bin/env python3
"""
2-forward_prop.py
Module that defines a function called create_layer
"""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that creates the forward propagation graph for the neural
    network:

    Args:
        - x is the placeholder for the input data
        - layer_sizes is a list containing the number of nodes in each
          layer of the network
        - activations is a list containing the activation functions for
          each layer of the network

    Returns:
        the prediction of the network in tensor form
    """
    y = x
    n = len(layer_sizes)
    for i in range(n):
        y = create_layer(y, layer_sizes[i], activations[i])
    return y
