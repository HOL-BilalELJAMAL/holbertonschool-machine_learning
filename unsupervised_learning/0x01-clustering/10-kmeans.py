#!/usr/bin/env python3
"""
10-kmeans.py
Module that defines a function called kmeans
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Function that performs K-means on a dataset

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters

    Returns:
        C, clss
        C is a numpy.ndarray of shape (k, d)
            containing the centroid means for each cluster
        clss is a numpy.ndarray of shape (n,)
            containing the index of the cluster in C that
            each data point belongs to
    """
    Kmean = sklearn.cluster.KMeans(n_clusters=k)
    Kmean.fit(X)
    C = Kmean.cluster_centers_
    clss = Kmean.labels_
    return C, clss
