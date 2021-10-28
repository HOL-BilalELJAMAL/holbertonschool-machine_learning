#!/usr/bin/env python3
"""
0-load_env.py
Module that defines a function called load_frozen_lake
"""

import numpy as np
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Function that loads a pre-made FrozenLakeEnv environment from OpenAI’s gym
    """
    env = gym.make("FrozenLake-v0",
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    return env
