#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io, csv
import warnings

def save_csv_generator(path_dir):

    print(f"csv files will be saved in: {path_dir}")

    def save_csv(fname, dataframe, overwrite=True, index=False):
        path = os.path.join(path_dir, fname)
        if os.path.exists(path) and not(overwrite):
            raise IOError(f"specified path exists: {path}")
        dataframe.to_csv(path, encoding="utf-8", header=True, index=index, sep="\t", quoting=csv.QUOTE_NONE)

    return save_csv


def save_picture_generator(path_dir):

    print(f"figures will be saved in: {path_dir}")

    def save_picture(fname, fig, format="eps", overwrite=True):
        basename, ext = os.path.splitext(fname)
        if ext.replace(".","") != format:
            fname = basename + "." + format
            warnings.warn(f"filename will be changed to {fname}")

        path = os.path.join(path_dir, fname)
        if os.path.exists(path) and not(overwrite):
            raise IOError(f"specified path exists: {path}")
        if hasattr(fig, "savefig"):
            fig.savefig(path, format=format)
        elif hasattr(fig, "save_result"):
            fig.save_result(path, format=format)
        else:
            raise AttributeError("specified object seems not figure.")

    return save_picture