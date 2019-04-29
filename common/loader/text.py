#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from abc import ABCMeta, abstractmethod

class AbstractLoader(object):

    __metaclass__ = ABCMeta

    def __init__(self, file_path, n_minibatch=1, encoding="utf8"):
        self._path = file_path
        self._n_mb = n_minibatch
        self._encoding = encoding
        self._n_rows = None

    @abstractmethod
    def __iter__(self):
        pass

    def __len__(self):
        return self.n_rows

    @property
    def n_rows(self):
        if self._n_rows is not None:
            return self._n_rows

        ret = 0
        with io.open(self._path, mode="r", encoding=self._encoding) as ifs:
            for _ in ifs:
                ret += 1
        self._n_rows = ret
        return ret

class TextLoader(AbstractLoader):

    def __init__(self, file_path, strip=None, encoding="utf8"):
        super(TextLoader, self).__init__(file_path, n_minibatch=0, encoding=encoding)
        self._strip = strip

    def __iter__(self):
        with io.open(self._path, mode="r", encoding=self._encoding) as ifs:
            for line in ifs:
                yield line.strip(self._strip)

class MinibatchTextLoader(AbstractLoader):

    def __init__(self, file_path, n_minibatch, strip=None, encoding="utf8"):
        super(MinibatchTextLoader, self).__init__(file_path, n_minibatch, encoding)
        self._strip = strip

    def __iter__(self):
        with io.open(self._path, mode="r", encoding=self._encoding) as ifs:
            lst_ret = []
            for line in ifs:
                lst_ret.append(line.strip(self._strip))
                if len(lst_ret) >= self._n_mb:
                    yield lst_ret
                    lst_ret = []
            if len(lst_ret) > 0:
                yield lst_ret