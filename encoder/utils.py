#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


def softmax(vec_z, tau=1.0):
    # scale
    vec_z *= tau
    # normalize
    vec_z -= np.max(vec_z)

    return np.exp(vec_z) / np.sum(np.exp(vec_z))

def sigmoid(x, tau):
    return 1. / (1. + np.exp(-tau*x))

def scaled_sigmoid(vec_z, tau=1.0):
    vec_h = sigmoid(vec_z, tau)
    return vec_h/ np.sum(vec_h)
