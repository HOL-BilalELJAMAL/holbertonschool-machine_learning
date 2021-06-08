#!/usr/bin/env python3
"""
11-learning_rate_decay.py
Module that defines a function called learning_rate_decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that updates the learning rate using inverse time decay in numpy

    Args:
        alpha (float): Original learning rate
        decay_rate (float): Weight used to find rate at which alpha will decay
        global_step (float): No of passes of gradient descent that have elapsed
        decay_step (float): No of passes of gradient descent that should occur
                            before alpha is decayed further
    Notes
        The learning rate decay should occur in a stepwise fashion

    Returns
        The updated value for alpha
    """
    return alpha / (1 + decay_rate * int(global_step / decay_step))
