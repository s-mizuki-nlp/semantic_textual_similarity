#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Tuple, Optional, Union, Callable, Dict
import warnings
import regex as re
import pandas as pd
import numpy as np
from scipy.special import gamma
from scipy.special import digamma
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

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

_obj_elmo_conf = re.compile(r"elmo_[1-9]x[0-9]{1,4}_[0-9]{1,4}_[0-9]{1,4}")


def extract_elmo_conf(path_elmo: str):
    m = _obj_elmo_conf.search(path_elmo)
    assert m is not None, "couldn't find elmo model specification."
    return m.group()


def continuous_to_discrete(vec_x: np.ndarray, n_bin: int,
                           interval_type: Optional[str] = "percentile", interval_format: Optional[str] = "range",
                           label_prefix: Optional[str] = "", value_format: Optional[str] = ".3g"):
    # define bins
    if interval_type == "percentile":
        vec_q = np.linspace(start=0, stop=100, num=n_bin+1)
        bins = np.percentile(vec_x, q=vec_q)
    elif interval_type == "eq":
        bins = np.linspace(start=np.min(vec_x), stop=np.max(vec_x), num=n_bin+1)
    elif isinstance(interval_type, (list,tuple)):
        x_min, x_max = interval_type
        bins = np.linspace(start=x_min, stop=x_max, num=n_bin+1)
    else:
        raise NotImplementedError(f"unsupported interval type:{interval_type}")

    # drop duplicates
    bins_dd = np.unique(bins)
    if len(bins_dd) < len(bins):
        warnings.warn(f"there are duplicate bins. de-dupicated bincount will be: {len(bins_dd)-1}")
        bins = bins_dd

    # define labels
    if interval_format == "range":
        labels = [f"{label_prefix}{b:{value_format}}~{e:{value_format}}" for b,e in zip(bins[:-1], bins[1:])]
    elif interval_format == "mean":
        labels = [f"{label_prefix}{(b+e)/2:{value_format}}" for b,e in zip(bins[:-1], bins[1:])]
    else:
        raise NotImplementedError(f"unsupported interval format:{interval_format}")

    vec_p = pd.cut(vec_x, bins=bins, labels=labels, include_lowest=True)
    return vec_p

def continuous_to_cluster(vec_x: np.ndarray, n_bin: int, **kwargs_kmeans):
    kmeans = KMeans(n_clusters=n_bin, random_state=0)
    if vec_x.ndim == 1:
        kmeans.fit(vec_x.reshape(-1,1))
    elif vec_x.ndim == 2:
        kmeans.fit(vec_x)
    else:
        raise NotImplementedError(f"too many axes: {vec_x.ndim}")

    return kmeans.labels_

def wrapper_mutual_information(dataframe: pd.DataFrame,
                               target_variable: str, lst_explanatory_variables: List[str],
                               lst_categorical_exp_variables: List[str]):
    df_t = dataframe.copy()
    for c in lst_categorical_exp_variables:
        le = LabelEncoder()
        df_t[c] = le.fit(df_t[c]).transform(df_t[c])

    mat_x = df_t[lst_explanatory_variables].values
    vec_y = df_t[target_variable]
    categorical = np.where([v in lst_categorical_exp_variables for v in lst_explanatory_variables])[0]

    vec_mi = mutual_info_regression(mat_x, vec_y, discrete_features=categorical)

    return vec_mi


def conditional_mutual_information(dataframe: pd.DataFrame,
                                   target_variable: str, lst_explanatory_variables: List[str],
                                   condition_variable: str,
                                   lst_categorical_exp_variables: List[str],
                                   discretizer_function: Optional[Callable] = None,
                                   verbose: bool = False):
    """
    calculates conditional mutual information: I(X;Y|Z)
    :param dataframe: dataframe which stores variables
    :param target_variable: target variable(=X)
    :param lst_explanatory_variables: explanatory varibles(=list of Y)
    :param condition_variable: condition variable(=Z)
    :param discretizer_function: optional discretizer function that is applied to variable Z
    :param verbose: output verbosity
    :return: conditional mutual information; I(X;Y|Z)
    """

    df_t = dataframe.copy()

    # discretize conditional variable if discretizer is specified.
    if discretizer_function is not None:
        df_t[condition_variable + "_c"] = discretizer_function(df_t[condition_variable].values)
        condition_variable = condition_variable + "_c"

    # convert categorical variables to K-ary index
    for c in lst_categorical_exp_variables:
        le = LabelEncoder()
        df_t[c] = le.fit(df_t[c]).transform(df_t[c])
    categorical = np.where([v in lst_categorical_exp_variables for v in lst_explanatory_variables])[0]
    if len(categorical) == 0:
        categorical = False

    # calculate p[cond_var == z]
    s_prob_z = df_t[condition_variable].value_counts()
    s_prob_z = s_prob_z / s_prob_z.sum()

    lst_mi_x_y_cond_z = []
    lst_prob_z = []
    for z, prob_z in zip(s_prob_z.index, s_prob_z.values):
        if verbose:
            print(f"{condition_variable} = {z}, P(Z=z): {prob_z:.3g}")
        df_z = df_t.loc[df_t[condition_variable] == z].reset_index(drop=True)
        mat_x = df_z[lst_explanatory_variables].values
        vec_y = df_z[target_variable].values

        try:
            vec_mi = mutual_info_regression(mat_x, vec_y, discrete_features=categorical)
        except Exception as e:
            print(e)
            warnings.warn(f"could not calculate mutual information: {condition_variable}={z}")
            continue

        lst_mi_x_y_cond_z.append(vec_mi)
        lst_prob_z.append(prob_z)

    # stack I(X;Y|Z=z). shape will be: (cardinality(N_cond_var), N_exp_var)
    mat_mi = np.stack(lst_mi_x_y_cond_z)
    # rescale P(Z=z)
    vec_prob_z = np.array(lst_prob_z)
    vec_prob_z /= np.sum(vec_prob_z)
    # I(X;Y|Z) = \sum_{z}{P(Z=z)I(X;Y|Z=z)}
    vec_mi_x_y_cond_z = np.sum(mat_mi * vec_prob_z.reshape(-1,1), axis=0)

    return vec_mi_x_y_cond_z


def cross_mutual_information(dataframe: pd.DataFrame,
                             target_variable: str, lst_explanatory_variables: List[str],
                             lst_categorical_exp_variables: List[str],
                             discretizer_function: Optional[Union[Callable, Dict[str, Callable]]] = None,
                             verbose: bool = False):

    n_variable = len(lst_explanatory_variables)
    mat_mi_x_y_z = np.zeros((n_variable, n_variable), dtype=np.float)

    # cross mutual information, I(X;Y,Z); Y,Z \in lst_explanatory_variables
    ## I(X;Z); Z \in lst_explanatory_variables
    vec_mi_x_z = wrapper_mutual_information(dataframe, target_variable, lst_explanatory_variables, lst_categorical_exp_variables)

    ## I(X;Y|Z); Z \in lst_explanatory_variables
    for idx_z, var_z in enumerate(lst_explanatory_variables):

        if isinstance(discretizer_function, dict):
            func_z = discretizer_function.get(var_z, None)
        else:
            func_z = discretizer_function

        # calculate conditional mutual information: I(X;Y|Z)
        lst_exp_var_excl_z = [var for var in lst_explanatory_variables if var != var_z]
        lst_exp_var_cat_excl_z = [var for var in lst_categorical_exp_variables if var != var_z]
        vec_mi_x_y_cond_z = conditional_mutual_information(dataframe, target_variable, lst_exp_var_excl_z,
                                                           condition_variable=var_z,
                                                           lst_categorical_exp_variables=lst_exp_var_cat_excl_z,
                                                           discretizer_function=func_z,
                                                           verbose=verbose
                                                           )
        ## insert I(X;Y|Z=Y) = 0.
        vec_mi_x_y_cond_z = np.insert(vec_mi_x_y_cond_z, obj=idx_z, values=0.)
        ## I(X;Y,Z) = I(X;Z) + I(X;Y|Z)
        mat_mi_x_y_z[idx_z,:] = vec_mi_x_z[idx_z] + vec_mi_x_y_cond_z

    return mat_mi_x_y_z


def _beta_function(a, b):
    ret = gamma(a)*gamma(b) / gamma(a+b)
    return ret

def _beta_distribution_entropy(a, b):
    ret = np.log(_beta_function(a, b)) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)
    if np.isfinite(ret):
        return -ret
    else:
        return np.nan

def estimate_beta_distribution_params(mean, std):
    if std == 0.:
        return np.nan, np.nan

    z = mean*(1. - mean)/(std**2) - 1
    a = mean*z
    b = (1. - mean)*z
    return a, b

def relative_position_entropy(mean, std):
    a, b = estimate_beta_distribution_params(mean, std)
    if a is np.nan:
        return np.nan
    else:
        return _beta_distribution_entropy(a, b)
