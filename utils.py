#!/usr/bin/env python
# -*- coding:utf-8 -*-

import warnings
import numpy as np
from distribution.distance import _l2_distance_sq

def soft_nearest_neighbor_loss(mat_x: np.ndarray, vec_y: np.ndarray, temperature: float = 1.0, normalize: bool = False):
    """

    :param mat_x: 2-dimensional feature matrix, (n_sample, n_feature)
    :param vec_y: 1-dimensional category vector, (n_sample,), vec_y[i] \in category
    :param templerature:
    """
    n_sample = mat_x.shape[0]
    assert len(vec_y) == n_sample, "length mismatch detected. sample size must be %d." % n_sample
    if n_sample >= 10000:
        warnings.warn(f"sample size is too large. there is the risk of out-of-memory error.")

    if normalize:
        mat_x = mat_x / np.linalg.norm(mat_x, axis=1, keepdims=True)
    # dist[i,j] = exp(-|x_i - x_j|_2^2 / T)
    mat_dist = np.exp(- _l2_distance_sq(mat_x, mat_x) / temperature)

    # denominator
    # d[i] = \sum_{k != i}{dist[i,k]} = \sum_{k}{dist[i,k]} - dist[i,i]
    vec_denominator = np.sum(mat_dist, axis=1) - np.diag(mat_dist)

    # numerator
    # n[i] = \sum_{k != i \and y[k] == y[i}}{dist[i,k]}
    vec_numerator = np.zeros_like(vec_denominator)
    for y in np.unique(vec_y):
        idx_y = np.where(vec_y == y)[0]
        mat_dist_y = mat_dist[idx_y,:][:,idx_y]
        vec_num_y = np.sum(mat_dist_y, axis=1) - np.diag(mat_dist_y)
        vec_numerator[idx_y] = vec_num_y

    # mask out isolated samples
    vec_mask = (vec_numerator > 0.0)

    # calculate mean of the log value.
    vec_values = np.log(vec_numerator) - np.log(vec_denominator)
    loss = - np.sum(vec_values[vec_mask]) / np.sum(vec_mask)

    return loss