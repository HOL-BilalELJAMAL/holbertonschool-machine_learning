#!/usr/bin/env python3
"""
poisson.py
Module that defines the poisson distribution
"""


class Poisson:
    """Class Poisson"""

    def __init__(self, data=None, lambtha=1.):
        """
        Function init of the Poisson Class

        Args:
            data (list): List of the data to be used to estimate the dist
            lambtha (float): Expected number of events in a given time frame

        Returns:
            λ
        """
        self.lambtha = lambtha
        self.π = 3.1415926536
        self.e = 2.7182818285
        λ = float(lambtha)
        if data is None:
            if λ <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            λ = float(sum(data) / len(data))
            self.lambtha = λ

    def pmf(self, k):
        """
        Function that calculates the value of the PMF for a given number
        of successes

        Args:
            k (float): Number of successes

        Returns:
            PMF (float): The PMF value for k
        """
        if k <= 0:
            return 0
        k = int(k)
        λ = self.lambtha
        k_f = self.factorial(k)
        e = self.e
        return (λ ** k) * (e ** -λ) / k_f

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

    def cdf(self, k):
        """
        Function that calculates the value of the CDF for a given number
        of successes

        Args:
            k (float): Number of successes

        Returns:
               CMF (float): The PMF value for k
        """
        if k <= 0:
            return 0
        k = int(k)
        cdf = 0
        while (k > 0):
            cdf += self.pmf(k)
            k = k - 0.9999999999
        return cdf
