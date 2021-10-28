#!/usr/bin/env python3
"""
3-q_learning.py
Module that defines a function called train
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs Q-learning
    """
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            map_size = env.desc.shape[0]
            new_state_on_map = env.desc[int(np.floor(new_state / map_size)),
                                        new_state % map_size]
            if new_state_on_map == b'H':
                reward = -1.0
            Q[state, action] = ((1 - alpha) * Q[state, action] + alpha *
                                (reward + gamma * np.max(Q[new_state, :])))
            state = new_state
            if done is True:
                break
        max_epsilon = 1
        epsilon = (min_epsilon + (max_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))
        total_rewards.append(reward)

    return Q, total_rewards
