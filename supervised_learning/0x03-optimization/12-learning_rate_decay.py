#!/usr/bin/env python3
"""
12-learning_rate_decay.py
Module that defines a function called learning_rate_decay
"""

import tensorflow as tf


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
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        step = sess.run(global_step)
        if step % decay_rate == 0:
            return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                               decay_rate, staircase=True)
        return alpha
