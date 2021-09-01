#!/usr/bin/env python3
"""
2-convolutional.py
Module that defines a function called autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Function that creates a convolutional auto encoder

    Args:
        input_dims: tuple of integers containing the dimensions
        of the model input
        filters: list containing the number of filters for
        each convolutional layer in the encoder, respectively
        latent_dims: tuple of integers containing
        the dimensions of the latent space representation

    Returns:
        encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    model_input = keras.layers.Input(shape=input_dims)
    encoded = model_input
    for i in range(len(filters)):
        encoded = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                      padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    decoded = keras.layers.Input(shape=latent_dims)
    input_d = decoded
    padding = 'same'
    for i in range(len(filters) - 1, -1, -1):
        if i == 0:
            padding = 'valid'
        decoded = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                      padding=padding)(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                                  padding='same')(decoded)
    encoder = keras.models.Model(model_input, encoded)
    decoder = keras.models.Model(input_d, decoded)
    auto = keras.models.Model(model_input, decoder(encoder(model_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
