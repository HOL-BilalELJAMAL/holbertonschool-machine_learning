#!/usr/bin/env python3
"""
2-cumulative_bleu.py
Module that defines a function called cumulative_bleu
"""

import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Function that calculates the cumulative BLEU score
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

    clipped_precision_scores = []

    for i in range(1, n + 1):
        sentence_n_grams = n_gram_generator(sentence, i)
        unique, counts = np.unique(sentence_n_grams, return_counts=True)
        sentence_n_grams_dict = dict(zip(unique, counts))

        references_n_grams_dicts = []
        for reference in references:
            reference_n_grams = n_gram_generator(reference, i)
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
        clipped_precision_scores.append(clipped_count / count)

    weights = [1 / n] * n

    if sentence_len > np.min(references_len):
        BP = 1
    else:
        bp = 1 - (np.min(references_len) / sentence_len)
        BP = np.exp(bp)

    BLEU = BP * np.exp(np.sum(weights * np.log(clipped_precision_scores)))
    return BLEU


def n_gram_generator(sentence, n=2):
    """
    Function that generates a list of n_grams from a sentence
    """
    word_num = sentence.shape[0]
    n_gram_list = np.empty((word_num + 1 - n,), dtype=object)
    for i in range(word_num + 1):
        if i < n:
            continue
        indices = list(range(i - n, i))
        n_gram = sentence[indices]
        n_gram = ' '.join(n_gram)
        n_gram_list[i - n] = n_gram
    return n_gram_list
