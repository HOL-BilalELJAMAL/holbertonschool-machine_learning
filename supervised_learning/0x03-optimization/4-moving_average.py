#!/usr/bin/env python3
"""
4-moving_average.py
Module that defines a function called moving_average
"""


def moving_average(data, beta):
    """
    Function that calculates the weighted moving average of a data set

    Args:
        data (list): List of data to calculate the moving average of
        beta (float): Weight used for the moving average

    Returns:
        List containing the moving averages of data
    """
    avg_list = []
    avg = 0
    for i in range(len(data)):
        avg = ((avg * beta) + ((1 - beta) * data[i]))
        bias_correction = 1 - (beta ** (i + 1))
        avg_list.append(avg / bias_correction)
    return avg_list
