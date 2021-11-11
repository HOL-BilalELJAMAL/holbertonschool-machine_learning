#!/usr/bin/env python3
"""
0-monte_carlo.py
Module that defines function to perform the Monte Carlo algorithm
"""

import gym
import numpy as np


def generate_episode(env, policy, max_steps):
    """
    Function that generates an episode using policy

    Args:
        env: the openAI environment instance
        policy: function that takes in state & returns the next action to take
        max_steps: the maximum number of steps per episode

    Returns:
        returns the episode
    """
    episode = [[], []]
    state = env.reset()
    for step in range(max_steps):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        episode[0].append(state)

        if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
            episode[1].append(-1)
            return episode

        if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
            episode[1].append(1)
            return episode

        episode[1].append(0)
        state = next_state
    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Function that performs the Monte Carlo algorithm

    Args:
        env: the openAI environment instance
        V [numpy.ndarray of shape(s,)]: contains the value estimate
        policy: function that takes in state & returns the next action to take
        episodes [int]: total number of episodes to train over
        max_steps [int]: the maximum number of steps per episode
        alpha [float]: the learning rate
        gamma [float]: the discount rate

    Returns:
        V: the updated value estimate
    """
    discounts = np.array([gamma ** i for i in range(max_steps)])
    for ep in range(episodes):
        episode = generate_episode(env, policy, max_steps)

        for i in range(len(episode[0])):
            Gt = np.sum(np.array(episode[1][i:]) *
                        np.array(discounts[:len(episode[1][i:])]))
            V[episode[0][i]] = (V[episode[0][i]] +
                                alpha * (Gt - V[episode[0][i]]))
    return V
