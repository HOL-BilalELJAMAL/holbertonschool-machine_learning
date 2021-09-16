#!/usr/bin/env python3
"""
preprocess_data.py
Module that defines the data pre-process of the Time Series Forecasting
"""

import numpy as np
import pandas as pd
import math
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def preprocess_data(file_path):
    """Function that preprocesses the data from the dataset"""

    df = pd.read_csv(file_path)
    df = df.dropna()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'],
                                     infer_datetime_format=True,
                                     unit='s')
    df = df.set_index('Timestamp')
    df = df.drop(['Low', 'High', 'Volume_(BTC)', 'Weighted_Price'], axis=1)
    df = df.reindex(columns=['Open', 'Close', 'Volume_(Currency)'])
    df['Open'] = df['Open'].resample('1H').first()
    df['Close'] = df['Close'].resample('1H').last()
    df['Volume_(Currency)'] = df['Volume_(Currency)'].resample('1H').sum()
    df = df.dropna()
    df = df.iloc[-int((df.shape[0]/2)):]

    print(df.head(10))
    print('=================')

    dataset = df.values
    print(dataset[:10])
    print(dataset.shape)
    print('=======================')

    mean = np.mean(dataset, axis=0)
    stddev = np.std(dataset, axis=0)
    dataset = (dataset - mean) / stddev
    print(dataset[10:])
    print(dataset.shape)
    print('=======================')

    def split_sequence(sequence, n_steps):
        """
        Function that splits a dataset sequence into input data and labels
        """

        X, Y = [], []
        for i in range(sequence.shape[0]):
            if (i + n_steps) >= sequence.shape[0]:
                break
            seq_X, seq_Y = sequence[i: i + n_steps], sequence[i + n_steps, -2]
            X.append(seq_X)
            Y.append(seq_Y)

        return np.array(X), np.array(Y)

    dataset_size = dataset.shape[0]
    x_train, y_train = split_sequence(
        dataset[0: math.ceil(0.7 * dataset_size)], 24)
    x_val, y_val = split_sequence(
        dataset[math.floor(0.7 * dataset_size):], 24)

    return dataset, df, x_train, y_train, x_val, y_val
