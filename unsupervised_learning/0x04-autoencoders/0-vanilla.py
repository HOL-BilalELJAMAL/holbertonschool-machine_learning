#!/usr/bin/env python3
"""
0-vanilla.py
Module that defines a function called autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates an auto encoder

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each
        hidden layer in the encoder, respectively
        latent_dims:  integer containing the dimensions of the latent
        space representation

    Returns:
        encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    model_input = keras.layers.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(model_input)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    decoded = keras.layers.Input(shape=(latent_dims,))
    input_d = decoded
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    encoder = keras.models.Model(model_input, encoded)
    decoder = keras.models.Model(input_d, decoded)
    auto = keras.models.Model(model_input, decoder(encoder(model_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
