#!/usr/bin/env python3
"""
exponential.py
Module that defines the exponential distribution
"""


class Exponential:
    """Class Exponential"""

    def __init__(self, data=None, lambtha=1.):
        """
        Function init of the Exponential Class

        Args:
            data (list): List of the data to be used to estimate the dist
            lambtha (float): Expected number of events in a given time frame

        Returns:
            λ
        """
        self.lambtha = float(lambtha)
        self.π = 3.1415926536
        self.e = 2.7182818285
        λ = self.lambtha
        if data is None:
            if λ <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            λ = float(1 / (sum(data) / len(data)))
            self.lambtha = λ

    def pdf(self, x):
        """
        Function that calculates the value of the PDF for a given number
        of successes

        Args:
            x (float):  is the time period

        Returns:
               pdf (float): The PDF value for x.
               0 if x is out of range
        """
        k = x
        λ = self.lambtha
        e = self.e
        if k < 0:
            return 0
        return λ * e ** (-1 * λ * k)

    def factorial(self, k):
        """
        Recursive function that returns the factorial of a number

        Args:
            k (int): Value

        Returns:
            num! (int): An error aproximation using the Maclaurin series.
        """
        n = int(k)
        fact = 1
        for n in range(1, n + 1):
            fact = fact * n
        return fact

    def cdf(self, x):
        """
        Function that calculates the value of the CDF for a given number
        of successes

        Args:
            x (float): Number of successes

        Returns:
            cdf (float): The PMF value for k.
        """
        k = x
        λ = self.lambtha
        e = self.e
        if k < 0:
            return 0
        return 1 - e ** (-1 * λ * k)
