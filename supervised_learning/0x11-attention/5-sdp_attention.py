#!/usr/bin/env python3
"""
5-sdp_attention.py
Module that defines a function called sdp_attention
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Function that computes the scaled dot product (sdp) attention"""
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)
    return output, attention_weights
