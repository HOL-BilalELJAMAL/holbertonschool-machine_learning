#!/usr/bin/env python3
"""
1-dataset.py
Module that defines a class called Dataset
"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Class Dataset
    """
    def __init__(self):
        """
        Class Constructor
        """
        self.data_train = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="train",
                                    as_supervised=True)
        self.data_valid = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="validation",
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Function that creates sub_word tokenizers for our dataset

        Args:
            data [tf.data.Dataset]:
                dataset to use whose examples are formatted as tuple (pt, en)
                pt [tf.Tensor]:
                    contains the Portuguese sentence
                en [tf.Tensor]:
                    contains the corresponding English sentence

        Returns:
            tokenizer_pt, tokenizer_en:
                tokenizer_pt: the Portuguese tokenizer
                tokenizer_en: the English tokenizer
        """
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=(2 ** 15))
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=(2 ** 15))
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Function that encodes a translation into tokens

        Args:
            pt [tf.Tensor]:
                contains the Portuguese sentence
            en [tf.Tensor]:
                contains the corresponding English sentence

        Returns:
            pt_tokens, en_tokens:
                pt_tokens [np.ndarray]: the Portuguese tokens
                en_tokens [np.ndarray]: the English tokens
        """
        pt_start_index = self.tokenizer_pt.vocab_size
        pt_end_index = pt_start_index + 1
        en_start_index = self.tokenizer_en.vocab_size
        en_end_index = en_start_index + 1
        pt_tokens = [pt_start_index] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_end_index]
        en_tokens = [en_start_index] + self.tokenizer_en.encode(
            en.numpy()) + [en_end_index]
        return pt_tokens, en_tokens