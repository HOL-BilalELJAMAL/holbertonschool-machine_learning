#!/usr/bin/env python3
"""
2-sarsa_lambtha.py
Module that defines function to perform the SARSA(λ) algorithm
"""

import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine if the reinforcement
    learning is exploring or exploiting and uses to get action

    Args:
        Q [numpy.ndarray of shape (s, a)]: contains the Q table
        state: the current state
        epsilon: the threshold for epsilon-greedy

    Returns:
        the action to take
    """
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs the SARSA(λ) algorithm

    Args:
        env: the openAI environment instance
        Q [numpy.ndarray of shape(s, a)]: contains the Q table
        lambtha: the eligibility trace factor
        episodes [int]: total number of episodes to train over
        max_steps [int]: the maximum number of steps per episode
        alpha [float]: the learning rate
        gamma [float]: the discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon [float]: the minimum value that epsilon should decay to
        epsilon_decay [float]: decay rate for updating epsilon between episodes

    Returns:
        Q: the updated Q table
    """
    max_epsilon = epsilon
    Et = np.zeros((Q.shape))
    for ep in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        for step in range(max_steps):
            Et = Et * lambtha * gamma
            Et[state, action] += 1

            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q, state, epsilon)

            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1
            if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1

            delta_t = reward + (
                gamma * Q[next_state, next_action]) - Q[state, action]
            Q[state, action] = Q[state, action] + (
                alpha * delta_t * Et[state, action])
            if done:
                break
            state = next_state
            action = next_action
        epsilon = min_epsilon + (
            (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * ep))
    return Q
