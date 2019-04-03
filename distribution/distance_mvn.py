#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from distribution.continuous import MultiVariateNormal


# kullback-leibler divergence
def kldiv_between_mvn(p_x: "MultiVariateNormal", p_y: "MultiVariateNormal"):

    if p_x.is_cov_diag and p_y.is_cov_diag:
        vec_mu_x = p_x.mean
        vec_cov_x = np.diag(p_x.covariance)
        vec_mu_y = p_y.mean
        vec_cov_y = np.diag(p_y.covariance)
        kldiv = _kldiv_mvn_diag(vec_mu_x, vec_cov_x, vec_mu_y, vec_cov_y)
    else:
        vec_mu_x = p_x.mean
        mat_cov_x = p_x.covariance
        vec_mu_y = p_y.mean
        mat_cov_y = p_y.covariance
        kldiv = _kldiv_mvn_full(vec_mu_x, mat_cov_x, vec_mu_y, mat_cov_y)

    return kldiv


def _kldiv_mvn_diag(vec_mu_x: np.ndarray, vec_cov_x: np.ndarray, vec_mu_y: np.ndarray, vec_cov_y: np.ndarray):

    d = vec_mu_x.size
    ln_det_cov1 = np.sum(np.log(vec_cov_x))
    ln_det_cov2 = np.sum(np.log(vec_cov_y))
    tr_cov12 = np.sum((vec_cov_y/vec_cov_x))
    quad_12 = np.sum(((vec_mu_x-vec_mu_y)/np.sqrt(vec_cov_x))**2)

    kldiv = 0.5*(tr_cov12 + quad_12 - d - ln_det_cov2 + ln_det_cov1)
    return kldiv


def _kldiv_mvn_full(vec_mu_x: np.ndarray, mat_cov_x: np.ndarray, vec_mu_y: np.ndarray, mat_cov_y: np.ndarray):

    vec_cov_x = np.diag(mat_cov_x)
    vec_cov_y = np.diag(mat_cov_y)
    mat_prec_x = np.linalg.inv(mat_cov_x)
    vec_mu_xy = vec_mu_x-vec_mu_y

    d = vec_mu_x.size
    _, ln_det_cov1 = np.linalg.slogdet(mat_cov_x)
    _, ln_det_cov2 = np.linalg.slogdet(mat_cov_y)
    tr_cov12 = np.sum((vec_cov_y/vec_cov_x))
    quad_12 = vec_mu_xy.dot(mat_prec_x).dot(vec_mu_xy.T)

    kldiv = 0.5*(tr_cov12 + quad_12 - d - ln_det_cov2 + ln_det_cov1)
    return kldiv