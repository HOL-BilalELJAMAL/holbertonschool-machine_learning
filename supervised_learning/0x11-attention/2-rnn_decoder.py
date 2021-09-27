#!/usr/bin/env python3
"""
2-rnn_decoder.py
Module that defines a class called RNNDecoder
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder Class"""

    def __init__(self, vocab, embedding, units, batch):
        """Class Constructor"""
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Function that builds the decoder"""
        self_attention = SelfAttention(self.units)
        context_vector, attention_weights = self_attention(s_prev,
                                                           hidden_states)
        embeddings = self.embedding(x)
        context_vector = tf.expand_dims(context_vector, 1)
        inputs = tf.concat([context_vector, embeddings], axis=-1)
        decoder_outputs, last_hidden_state = self.gru(inputs)
        y = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))
        y = self.F(y)
        return y, last_hidden_state
