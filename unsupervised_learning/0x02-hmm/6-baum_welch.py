#!/usr/bin/env python3
"""
6-baum_welch.py
Module that defines a function called baum_welch
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Function that performs the forward algorithm for a hidden
    markov model

    Args:
        Observation: numpy.ndarray of shape (T,)
        that contains the index of the observation
        T is the number of observations
        Emission: numpy.ndarray of shape (N, M)
        containing the emission probability of a specific observation
        given a hidden state
        Emission[i, j] is the probability of observing j given the
        hidden state i
        N is the number of hidden states
        M is the number of all possible observations
        Transition: 2D numpy.ndarray of shape (N, N)
        containing the transition probabilities
        Transition[i, j] is the probability of transitioning from the
        hidden state i to j
        Initial: numpy.ndarray of shape (N, 1) containing the probability
        of starting in a particular hidden state

    Returns:
        P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T)
        containing the forward path probabilities
        F[i, j] is the probability of being in hidden state i at time j
        given the previous observations
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]

    N, M = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    F = np.zeros((N, T))

    Obs_i = Observation[0]
    prob = np.multiply(Initial[:, 0], Emission[:, Obs_i])
    F[:, 0] = prob

    for i in range(1, T):
        Obs_i = Observation[i]
        state = np.matmul(F[:, i - 1], Transition)
        prob = np.multiply(state, Emission[:, Obs_i])
        F[:, i] = prob

    P = np.sum(F[:, T - 1])

    return P, F


def backward(Observation, Emission, Transition, Initial):
    """
    Function that performs the backward algorithm for a hidden
    markov model

    Args:
        Observation: numpy.ndarray of shape (T,) that contains the index
        of the observation T is the number of observations
        Emission: numpy.ndarray of shape (N, M)
        containing the emission probability of a specific observation
        given a hidden state
        Emission[i, j] is the probability of observing j given the
        hidden state i
        N is the number of hidden states
        M is the number of all possible observations
        Transition: 2D numpy.ndarray of shape (N, N)
        containing the transition probabilities
        Transition[i, j] is the probability of transitioning from the
        hidden state i to j
        Initial: numpy.ndarray of shape (N, 1) containing the probability
        of starting in a particular hidden state

    Returns:
        P, B, or None, None on failure
        P is the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the
        backward path probabilities
        B[i, j] is the probability of generating the future observations
        from hidden state i at time j
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]

    N, M = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    Beta = np.zeros((N, T))
    Beta[:, T - 1] = np.ones(N)

    for t in range(T - 2, -1, -1):
        a = Transition
        b = Emission[:, Observation[t + 1]]
        c = Beta[:, t + 1]

        abc = a * b * c
        prob = np.sum(abc, axis=1)
        Beta[:, t] = prob

    P_first = Initial[:, 0] * Emission[:, Observation[0]] * Beta[:, 0]
    P = np.sum(P_first)

    return P, Beta


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Function that performs the Baum-Welch algorithm for a hidden
    markov model

    Args:
        Observations: numpy.ndarray of shape (T,)
        that contains the index of the observation
        T is the number of observations
        Transition: numpy.ndarray of shape (M, M)
        that contains the initialized transition probabilities
        M is the number of hidden states
        Emission: numpy.ndarray of shape (M, N)
        that contains the initialized emission probabilities
        N is the number of output states
        Initial: numpy.ndarray of shape (M, 1)
        that contains the initialized starting probabilities
        iterations: number of times expectation-maximization
        should be performed

    Returns:
        the converged Transition, Emission, or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) \
            or len(Observations.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observations.shape[0]
    N, M = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    for n in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            a = np.matmul(alpha[:, t].T, Transition)
            b = Emission[:, Observations[t + 1]].T
            c = beta[:, t + 1]
            denominator = np.matmul(a * b, c)

            for i in range(N):
                a = alpha[i, t]
                b = Transition[i]
                c = Emission[:, Observations[t + 1]].T
                d = beta[:, t + 1].T
                numerator = a * b * c * d
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)

        num = np.sum(xi, 2)
        den = np.sum(gamma, axis=1).reshape((-1, 1))
        Transition = num / den

        xi_sum = np.sum(xi[:, :, T - 2], axis=0)
        xi_sum = xi_sum.reshape((-1, 1))
        gamma = np.hstack((gamma, xi_sum))

        denominator = np.sum(gamma, axis=1)
        denominator = denominator.reshape((-1, 1))

        for i in range(M):
            gamma_i = gamma[:, Observations == i]
            Emission[:, i] = np.sum(gamma_i, axis=1)

        Emission = Emission / denominator

    return Transition, Emission
