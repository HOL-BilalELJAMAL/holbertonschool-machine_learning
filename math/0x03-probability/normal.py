#!/usr/bin/env python3
"""
normal.py
Module that defines the normal distribution
"""


class Normal:
    """Class Normal"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Function init of the Normal Class

        Args:
            data (lst): List of the data to be used to estimate the dist
            mean (float): the mean of the distribution
            stddev (float): Standard deviation of the distribution

        Returns:
            Nothing.
        """
        self.mean = float(mean)
        μ = self.mean
        self.stddev = float(stddev)
        σ = self.stddev
        π = 3.1415926536
        e = 2.7182818285
        if data is None:
            if σ <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            μ = sum(data) / n
            self.mean = μ
            variance = sum((x - μ)**2 for x in data) / n
            σ = variance ** 0.5
            self.stddev = σ

    def z_score(self, x):
        """
        Function that calculates the z-score of a given x-value

        Args:
            x (float):  is the time period

        Returns:
            z-score
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Function that calculates the x-value of a given z-score

        Args:
            z (float):  is the time period

        Returns:
            z-score
        """
        return (z * self.stddev) + self.mean

    def erf(self, x):
        """
        Function that calculates the x-value of a given z-score

        Args:
            x (float):  is the time period

        Returns:
            z-score
        """
        π = 3.1415926536
        a = ((4 / π)**0.5)
        b = (x - (x**3) / 3 + (x**5) / 10 - x**7 / 42 + (x**9) / 216)
        return a * b

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
        π = 3.1415926536
        e = 2.7182818285
        return (e**((-1/2) * (((
            x - self.mean) / self.stddev)**2))) * (
            1 / (self.stddev * (2 * π) ** (1/2)))

    def cdf(self, x):
        """
        Function that calculates the value of the CDF for a given number
        of successes

        Args:
            x (float): Number of “successes”

        Returns:
               cdf (float): The PMF value for k.
        """
        a = (x - self.mean) / (self.stddev * (2**0.5))
        erf = self.erf(a)
        return (1 + erf) / 2
