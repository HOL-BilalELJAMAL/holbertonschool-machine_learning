#!/usr/bin/env python3
"""
2-epsilon_greedy.py
Module that defines a function called epsilon_greedy
"""

import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action
    """
    exploration_rate_threshold = np.random.uniform(0, 1)
    if exploration_rate_threshold > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(Q.shape[1])
    return action
