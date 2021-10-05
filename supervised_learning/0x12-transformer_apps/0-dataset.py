#!/usr/bin/env python3
"""
0-dataset.py
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
        Class constructor
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
