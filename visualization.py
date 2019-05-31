#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Tuple, List, Iterable
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib import colors
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np

from utils import continuous_to_discrete

_DEFAULT_FIG_SIZE = (8,8)

def word_barplot(lst_word: Iterable[str], lst_score: Iterable[float], lst_word_attr: Optional[Iterable[str]] = None,
                 color: Optional[str] = "orange", horizontal: Optional[bool] = True,
                 fig_and_ax = None, figsize = None):

    lst_y_dummy = [f"{word}_{idx}" for idx, word in enumerate(lst_word)]
    if horizontal:
        x = lst_score
        y = lst_y_dummy
    else:
        x = lst_y_dummy
        y = lst_score
    if fig_and_ax is None:
        fig, ax = plt.subplots(figsize=_DEFAULT_FIG_SIZE if figsize is None else figsize)
    else:
        fig, ax = fig_and_ax[0], fig_and_ax[1]
    if lst_word_attr is not None:
        color = None

    ax = sns.barplot(x=x, y=y, dodge=False, hue=lst_word_attr, color=color, ax=ax)
    if horizontal:
        ax.set_yticklabels(lst_word)
    else:
        ax.set_xticklabels(lst_word)

    return fig, ax

def continuous_heatmap(dataframe, target_variable, explanatory_variable_x, explanatory_variable_y, discretizer_function=None,
                       fig_and_ax=None, figsize=None, **kwargs):

    if discretizer_function is None:
        discretizer_function = lambda vec: continuous_to_discrete(vec, n_bin=30, interval_type="eq", interval_format="mean", value_format=".2g")

    if fig_and_ax is None:
        fig, ax = plt.subplots(figsize=_DEFAULT_FIG_SIZE if figsize is None else figsize)
    else:
        fig, ax = fig_and_ax[0], fig_and_ax[1]


    lst_vars = [target_variable, explanatory_variable_x, explanatory_variable_y]
    df_view = dataframe[lst_vars].copy()

    for field_to, field_from in zip(["c_x","c_y"], [explanatory_variable_x, explanatory_variable_y]):
        if isinstance(discretizer_function, dict):
            func_c = discretizer_function.get(field_from, None)
        else:
            func_c = discretizer_function

        df_view[field_to] = func_c(df_view[field_from])

    df_view = df_view.groupby(by=["c_x","c_y"])[target_variable].mean()
    df_view = df_view.reset_index(drop=False).pivot(index="c_y", columns="c_x", values=target_variable)
    df_view = df_view.iloc[::-1]

    ax = sns.heatmap(df_view, cmap="coolwarm", ax=ax, **kwargs)

    return fig, ax


def tile_plot(mat_dist, fig_and_ax=None, lst_ticker_x=None, lst_ticker_y=None, figsize=None, cmap="Reds",
              colorbar=True, tup_dist_min_max: Optional[Tuple[float, float]] = None, **kwargs):

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{x+0.5:2.0f}"

    if fig_and_ax is None:
        fig, ax = plt.subplots(figsize=_DEFAULT_FIG_SIZE if figsize is None else figsize)
    else:
        fig, ax = fig_and_ax[0], fig_and_ax[1]

    n_x, n_y = mat_dist.shape
    mesh_x, mesh_y = np.meshgrid(np.arange(n_x+1), np.arange(n_y+1))

    ax.pcolormesh(mesh_x, mesh_y, mat_dist.T, cmap=cmap, **kwargs)
    # adjust axes labels
    ## X=axis
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5, n_x, 1.0)))
    if lst_ticker_x is None:
        ax.xaxis.set_major_formatter(major_formatter)
    else:
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(lst_ticker_x))
        ax.tick_params(axis="x", rotation=90)
    ## Y-axis
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5, n_y, 1.0)))
    if lst_ticker_y is None:
        ax.yaxis.set_major_formatter(major_formatter)
    else:
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(lst_ticker_y))
    ## colorbar
    if colorbar:
        if tup_dist_min_max is None:
            vmin, vmax = mat_dist.min(), mat_dist.max()
        else:
            vmin, vmax = tup_dist_min_max
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable._A = []
        fig.colorbar(mappable)

    return fig, ax


def tile_bar_plot(mat_dist: np.ndarray, vec_x_weight, vec_y_weight, lst_ticker_x: List[str], lst_ticker_y: List[str],
                  figsize=None, annotation_format=".2f",
                  cmap_dist="Reds", cmap_bar="Blues",
                  tup_dist_min_max: Optional[Tuple[float, float]] = None,
                  tup_bar_min_max: Tuple[float, float] = (0.0, 1.0),
                  **kwargs):

    def min_and_max(x: np.ndarray):
        return x.min(), x.max()

    df_dist = pd.DataFrame(data=mat_dist.T, columns=lst_ticker_x, index=lst_ticker_y).iloc[::-1]
    df_bar_x = pd.DataFrame(data=vec_x_weight, index=lst_ticker_x)
    df_bar_y = pd.DataFrame(data=vec_y_weight, index=lst_ticker_y).iloc[::-1]

    figsize = _DEFAULT_FIG_SIZE if figsize is None else figsize
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((20,20), (1,0), colspan=19, rowspan=19) # tile
    ax2 = plt.subplot2grid((20,20), (0,0), colspan=19, rowspan=1) # horizontal
    ax3 = plt.subplot2grid((20,20), (1,19), colspan=1, rowspan=19) # vertical

    # visualize tile plot
    vmin, vmax = min_and_max(mat_dist) if tup_dist_min_max is None else tup_dist_min_max
    sns.heatmap(df_dist, ax=ax1, annot=True, cmap=cmap_dist, linecolor='b', fmt=annotation_format, cbar=False, vmin=vmin, vmax=vmax, **kwargs)
    # ax1.xaxis.tick_bottom()
    ax1.set_xticklabels(df_dist.columns, rotation=90)
    ax1.set_yticklabels(df_dist.index, rotation=0)

    # visualize horizontal axis
    vmin, vmax = min_and_max(vec_x_weight) if tup_bar_min_max is None else tup_bar_min_max
    sns.heatmap(df_bar_x.transpose(), ax=ax2,  annot=True, cmap=cmap_bar, fmt=annotation_format, cbar=False, xticklabels=False, yticklabels=False,
                vmin=vmin, vmax=vmax, **kwargs)
    # visualize vertical axis
    vmin, vmax = min_and_max(vec_y_weight) if tup_bar_min_max is None else tup_bar_min_max
    sns.heatmap(df_bar_y, ax=ax3,  annot=True, annot_kws={"rotation":90}, fmt=annotation_format, cmap=cmap_bar, cbar=False, xticklabels=False, yticklabels=False,
                vmin=vmin, vmax=vmax, **kwargs)

    return fig, (ax1, ax2, ax3)