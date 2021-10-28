#!/usr/bin/env python3
"""
1-q_init.py
Module that defines a function called q_init
"""

import numpy as np
import gym


def q_init(env):
    """
    Function that initializes the Q-table
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros(shape=(state_space_size, action_space_size))
    return q_table
