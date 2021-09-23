#!/usr/bin/env python3
"""
0-uni_bleu.py
Module that defines a function called uni_bleu
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Function that calculates the unigram BLEU score
    in a machine translation paradigm
    """

    if not isinstance(sentence, list):
        sentence = sentence.lower()
        sentence = np.array(sentence.split())
    else:
        sentence = np.array([word.lower() for word in sentence])

    sentence_len = sentence.shape[0]

    if not np.all([isinstance(reference, list) for reference in references]):
        references = [reference.split() for reference in references]
        references = [np.array(reference.split()) for reference in references]
    else:
        references = [np.array([word.lower() for word in reference])
                      for reference in references]

    references_len = [reference.shape[0] for reference in references]

    clipped_precision_score = 0

    sentence_n_grams = sentence
    unique, counts = np.unique(sentence_n_grams, return_counts=True)
    sentence_n_grams_dict = dict(zip(unique, counts))

    references_n_grams_dicts = []
    for reference in references:
        reference_n_grams = reference
        unique, counts = np.unique(reference_n_grams, return_counts=True)
        reference_n_grams_dict = dict(zip(unique, counts))
        references_n_grams_dicts.append(reference_n_grams_dict)

    count = sum(sentence_n_grams_dict.values())
    for n_gram in sentence_n_grams_dict.keys():
        references_n_gram_count = [reference_n_grams_dict[n_gram]
                                   for reference_n_grams_dict
                                   in references_n_grams_dicts
                                   if n_gram in reference_n_grams_dict]
        if not len(references_n_gram_count):
            keep_max = 0
        else:
            keep_max = max(references_n_gram_count)
        if sentence_n_grams_dict[n_gram] > keep_max:
            sentence_n_grams_dict[n_gram] = keep_max

    clipped_count = sum(sentence_n_grams_dict.values())
    clipped_precision_score += (clipped_count / count)

    if sentence_len > np.min(references_len):
        BP = 1
    else:
        bp = 1 - (np.min(references_len) / sentence_len)
        BP = np.exp(bp)

    BLEU = BP * clipped_precision_score
    return BLEU
