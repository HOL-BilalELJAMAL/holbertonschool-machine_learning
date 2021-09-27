#!/usr/bin/env python3
"""
10-transformer_decoder.py
Module that defines a class called Decoder
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder Class"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Class Constructor"""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Function that builds a Decoder"""
        seq_len = x.shape[1]
        embeddings = self.embedding(x)
        embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embeddings += self.positional_encoding[:seq_len, :]
        output = self.dropout(embeddings, training=training)
        for i in range(self.N):
            output = self.blocks[i](output, encoder_output, training,
                                    look_ahead_mask, padding_mask)
        return output
