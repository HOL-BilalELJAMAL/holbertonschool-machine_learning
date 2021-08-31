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
    inputs = keras.Input(shape=(input_dims,))

    my_layer = keras.layers.Dense(units=hidden_layers[0],
                                  activation='relu',
                                  input_shape=(input_dims,))(inputs)

    for i in range(1, len(hidden_layers)):
        my_layer = keras.layers.Dense(units=hidden_layers[i],
                                      activation='relu'
                                      )(my_layer)

    my_layer = keras.layers.Dense(units=latent_dims,
                                  activation='relu'
                                  )(my_layer)

    encoder = keras.Model(inputs=inputs, outputs=my_layer)

    inputs_dec = keras.Input(shape=(latent_dims,))

    my_layer_dec = keras.layers.Dense(units=hidden_layers[-1],
                                      activation='relu',
                                      input_shape=(latent_dims,))(inputs_dec)

    for i in range(len(hidden_layers) - 2, -1, -1):
        my_layer_dec = keras.layers.Dense(units=hidden_layers[i],
                                          activation='relu'
                                          )(my_layer_dec)

    my_layer_dec = keras.layers.Dense(units=input_dims,
                                      activation='sigmoid'
                                      )(my_layer_dec)

    decoder = keras.Model(inputs=inputs_dec, outputs=my_layer_dec)

    auto_bottleneck = encoder.layers[-1].output
    auto_output = decoder(auto_bottleneck)

    auto = keras.Model(inputs=inputs, outputs=auto_output)

    auto.compile(optimizer=keras.optimizers.Adam(),
                 loss='binary_crossentropy')

    return encoder, decoder, auto
