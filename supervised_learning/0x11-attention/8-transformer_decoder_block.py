#!/usr/bin/env python3
"""
8-transformer_decoder_block.py
Module that defines a class called DecoderBlock
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock Class"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class Constructor"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Function that builds the Decoder block"""
        attention_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        attention_output1 = self.layernorm1(attention_output1 + x)
        attention_output2, _ = self.mha2(attention_output1, encoder_output,
                                         encoder_output, padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        attention_output2 = self.layernorm2(attention_output2 +
                                            attention_output1)
        ffn_output = self.dense_hidden(attention_output2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        decoder_output = self.layernorm3(ffn_output + attention_output2)
        return decoder_output
