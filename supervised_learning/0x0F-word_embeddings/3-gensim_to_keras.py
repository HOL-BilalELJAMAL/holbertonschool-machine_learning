#!/usr/bin/env python3
"""
3-gensim_to_keras.py
"""

from gensim.models import Word2Vec


def gensim_to_keras(model):
    """
    Function that converts a gensim word2vec model to a trainable keras layer
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
