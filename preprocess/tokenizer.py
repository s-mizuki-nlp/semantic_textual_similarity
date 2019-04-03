#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from abc import ABCMeta, abstractmethod
from typing import List
from .wordpiece import FullTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.moses import MosesTokenizer

class AbstractTokenizer(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def tokenize_single(self, sentence: str) -> List[str]:
        pass

    def tokenize(self, corpus):
        if (iter(corpus) is iter(corpus)) or isinstance(corpus, str):
            raise TypeError("`corpus` must be a container.")
        for sentence in corpus:
            yield self.tokenize_single(sentence)


class CharacterTokenizer(AbstractTokenizer):

    def __init__(self, separator: str=" ", do_lower_case = False, remove_punctuation_mark = False):
        self._sep = separator
        self._do_lower_case = do_lower_case
        self._remove_punct_mark = remove_punctuation_mark

    def _remove_punctuation(self, token: str):
        if token.endswith((".",",")):
            token = token[:-1]
        return token

    def tokenize_single(self, sentence):
        if self._do_lower_case:
            sentence = sentence.lower()
        lst_token = sentence.split(self._sep)
        if self._remove_punct_mark:
            lst_token = list(map(self._remove_punctuation, lst_token))

        return lst_token


class WordPieceTokenizer(AbstractTokenizer):

    def __init__(self, vocab_file: str, do_lower_case = False):
        self._tokenizer = FullTokenizer(vocab_file, do_lower_case)

    def tokenize_single(self, sentence):
        return self._tokenizer.tokenize(sentence)


class TreeBankWordTokenizerWrapper(AbstractTokenizer):

    def __init__(self, do_lower_case: bool = False):
        self._tokenizer = TreebankWordTokenizer()
        self._do_lower_case = do_lower_case

    def tokenize_single(self, sentence: str):
        if self._do_lower_case:
            sentence = sentence.lower()
        return self._tokenizer.tokenize(sentence)

class MosesTokenizerWrapper(AbstractTokenizer):

    def __init__(self, do_lower_case: bool = False, escape: bool = False):
        self._tokenizer = MosesTokenizer()
        self._do_lower_case = do_lower_case
        self._escape = escape

    def tokenize_single(self, sentence: str):
        if self._do_lower_case:
            sentence = sentence.lower()
        return self._tokenizer.tokenize(sentence, escape=self._escape)
