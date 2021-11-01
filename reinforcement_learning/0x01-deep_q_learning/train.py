#!/usr/bin/env python3
"""
train.py
Module that defines a class called AtariProcessor
"""

import gym
import numpy as np
import tensorflow.keras as K
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor


class AtariProcessor(Processor):
    """
    Class that defines atari environment to play breakout
    """
    def process_observation(self, observation):
        """
        Function that resizes images and makes grayscale to conserve memory
        """
        assert observation.ndim == 3
        image = Image.fromarray(observation)
        image = image.resize((84, 84), Image.ANTIALIAS).convert('L')
        processed_observation = np.array(image)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Function that converts a batch of images to float32
        """
        processed_batch = batch.astype('float32') / 255.0
        return processed_batch

    def process_reward(self, reward):
        """
        Function that processes reward between -1 and 1
        """
        return np.clip(reward, -1., 1.)


def create_CNN_model(number_actions, frames=4, input_shape=(84, 84)):
    """
    Function that creates a CNN model with Keras as defined by the
    DeepMind resource
    """
    full_input_shape = (window,) + input_shape
    inputs = K.layers.Input(shape=full_input_shape)
    layer_0 = K.layers.Permute((2, 3, 1))(inputs)

    layer_1 = K.layers.Conv2D(filters=32,
                              kernel_size=8,
                              strides=4,
                              activation='relu',
                              data_format='channels_last')(layer_0)

    layer_2 = K.layers.Conv2D(filters=64,
                              kernel_size=4,
                              strides=2,
                              activation='relu',
                              data_format='channels_last')(layer_1)

    layer_3 = K.layers.Conv2D(filters=64,
                              kernel_size=3,
                              strides=1,
                              activation='relu',
                              data_format='channels_last')(layer_2)

    layer_4 = layers.Flatten()(layer_3)

    layer_5 = layers.Dense(units=512,
                           activation='relu')(layer_4)

    outputs = layers.Dense(units=number_actions,
                           activation='linear')(layer_5)

    return K.Model(inputs=inputs, outputs=outputs), frames


def training():
    """
    Function that trains an agent to play Atari's Breakout
    """
    env = gym.make("Breakout-v0")
    env.reset()
    nb_actions = env.action_space.n
    model, frames = create_CNN_model(nb_actions)
    model.summary()
    memory = SequentialMemory(limit=1000000, window_length=frames)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.0,
                                  value_min=0.1,
                                  value_test=0.05,
                                  nb_steps=1000000)
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.0)
    dqn.compile(K.optimizers.Adam(lr=0.00025),
                metrics=['mae'])
    dqn.fit(env,
            nb_steps17500,
            log_interval=10000,
            visualize=False,
            verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)


if __name__ == '__main__':
    training()
