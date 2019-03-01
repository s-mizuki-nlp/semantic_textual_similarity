#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import math
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Iterable, List, Dict, Tuple, Union, Any, Optional

from more_itertools import chunked

from .tokenizer import AbstractTokenizer
from .wordpiece import FullTokenizer
from .corpora import Dictionary
from common.loader.table import AnnotatedTextLoader, PayloadContainer

class AbstractFeeder(object):

    __metaclass__ = ABCMeta

    def __init__(self, n_minibatch=1, validation_split=0.0):
        self._n_mb = n_minibatch
        self._validation_split = validation_split
        self._n_validation = math.ceil(n_minibatch*validation_split)

    @abstractmethod
    def _init_iter_batch(self):
        pass

    def __iter__(self):

        iter_dataset = self._init_iter_batch()
        iter_batch = chunked(iter_dataset, self._n_mb)
        for lst_batch in iter_batch:
            valid = lst_batch[:self._n_validation]
            train = lst_batch[self._n_validation:]

            yield train, valid


class GeneralSequenceFeeder(AbstractFeeder):

    def __init__(self, corpus: Iterable, tokenizer: AbstractTokenizer, dictionary: Dictionary, n_minibatch=1, validation_split=0.0):
        super(__class__, self).__init__(n_minibatch, validation_split)

        self._corpus = corpus
        self._tokenizer = tokenizer
        self._dictionary = dictionary

    def _init_iter_batch(self):

        iter_token = self._tokenizer.tokenize(self._corpus)
        iter_token_idx = self._dictionary.iter_transform(iter_token)

        return iter_token_idx

    def process_single_sentence(self, sentence: str):
        lst_token = self._tokenizer.tokenize_single(sentence)
        lst_token_idx = self._dictionary.transform(lst_token)
        return lst_token_idx

    def __len__(self):
        return len(self._corpus)

    @property
    def size(self):
        return len(self._corpus)

class GeneralSentenceFeeder(GeneralSequenceFeeder):

    def __init__(self, corpus: Iterable, tokenizer: AbstractTokenizer, dictionary: Optional[Dictionary] = None,
                 n_minibatch: int = 1, validation_split: float = 0.0,
                 min_seq_len: Optional[int] = None,
                 max_seq_len: Optional[int] = None,
                 bos_symbol: Optional[str] = None, eos_symbol: Optional[str] = None):
        """
        extension of the GeneralSequenceFeeder class that is customized to sentence feed.
        it can prepend/append special token at the beginning/end of the sentence.

        :param corpus: collection of the sentence
        :param tokenizer: tokenizer for the sentence
        :param dictionary: token-to-index encoder
        :param n_minibatch: minibatch size
        :param validation_split: if enabled, the fraction of the minibatch is splitted into validation set.
        :param min_seq_len: minimum sqquence length. shorter sequence will be discarded. DEFAULT:None
        :param max_seq_len: maximum sequence length. longer sequence will be trimmed. DEFAULT:None
        :param bos_symbol: beginning-of-sentence symbol. e.g. `<bos>`
        :param eos_symbol: end-of-sentence symbol. e.g. `<eos>`
        """
        super(__class__, self).__init__(corpus, tokenizer, dictionary, n_minibatch, validation_split)

        self._min_seq_len = min_seq_len
        self._max_seq_len = max_seq_len
        self._bos = bos_symbol
        self._eos = eos_symbol
        if self._bos is not None:
            if self._eos is not None:
                self._mode = "both"
            else:
                self._mode = "bos"
        else:
            if self._eos is not None:
                self._mode = "eos"
            else:
                self._mode = "none"

    def _init_iter_batch(self):

        iter_sentences = self._tokenizer.tokenize(self._corpus)
        # discard too short sequence
        if self._min_seq_len is not None:
            iter_sentences = (lst_token for lst_token in iter_sentences if len(lst_token) >= self._min_seq_len)
        # trim sequence length
        if self._max_seq_len is not None:
            iter_sentences = (lst_token[:self._max_seq_len] for lst_token in iter_sentences)

        # prepend/append special tokens
        if self._mode == "bos":
            iter_sentences = ([self._bos] + lst_token for lst_token in iter_sentences)
        elif self._mode == "eos":
            iter_sentences = (lst_token + [self._eos] for lst_token in iter_sentences)
        elif self._mode == "both":
            iter_sentences = ([self._bos] + lst_token + [self._eos] for lst_token in iter_sentences)

        # transform to integer sequence
        if self._dictionary is not None:
            iter_sentences = self._dictionary.iter_transform(iter_sentences)

        return iter_sentences

    def process_single_sentence(self, sentence: str):
        lst_token = self._tokenizer.tokenize_single(sentence)

        # prepend/append special tokens
        if self._mode == "bos":
            lst_token = [self._bos] + lst_token
        elif self._mode == "eos":
            lst_token = lst_token + [self._eos]
        elif self._mode == "both":
            lst_token = [self._bos] + lst_token + [self._eos]

        lst_token_idx = self._dictionary.transform(lst_token)
        return lst_token_idx


class SeqToGMMFeeder(AbstractFeeder):

    def __init__(self, corpus: Iterable, tokenizer: AbstractTokenizer, dictionary: Dictionary,
                 dict_lst_gmm_param: Dict[str, Any],
                 convert_var_to_std: bool = True,
                 n_minibatch=1, validation_split=0.0):
        super(__class__, self).__init__(n_minibatch, validation_split)

        self._corpus = corpus
        self._tokenizer = tokenizer
        self._dictionary = dictionary
        self._gmm_param = dict_lst_gmm_param
        self._gmm_param_name = "alpha,mu,scale".split(",")

        if convert_var_to_std:
            self._gmm_param["scale"] = [np.sqrt(v_cov) for v_cov in self._gmm_param["cov"]]
        else:
            self._gmm_param["scale"] = self._gmm_param["cov"]
        del self._gmm_param["cov"]


    def _init_iter_batch(self):

        iter_token = self._tokenizer.tokenize(self._corpus)
        iter_token_idx = self._dictionary.iter_transform(iter_token)
        lst_gmm_param = map(self._gmm_param.get, self._gmm_param_name)

        iter_trainset = zip(iter_token_idx, *lst_gmm_param)

        return iter_trainset

class AnnotatedTextFeeder(AbstractFeeder):

    def __init__(self, corpus: AnnotatedTextLoader, tokenizer: AbstractTokenizer,
                 dictionary: Optional[Dictionary] = None, n_minibatch: int = 1, validation_split: float = 0.0,
                 min_seq_len: Optional[int] = None,
                 max_seq_len: Optional[int] = None,
                 bos_symbol: Optional[str] = None, eos_symbol: Optional[str] = None):
        """
        extension of the GeneralSequenceFeeder class that is customized to sentence feed.
        it can prepend/append special token at the beginning/end of the sentence.

        :param corpus: collection of the sentence
        :param tokenizer: tokenizer for the sentence
        :param dictionary: token-to-index encoder
        :param n_minibatch: minibatch size
        :param bos_symbol: beginning-of-sentence symbol. e.g. `<bos>`
        :param eos_symbol: end-of-sentence symbol. e.g. `<eos>`
        """
        super(__class__, self).__init__(n_minibatch, validation_split=validation_split)

        self._corpus = corpus
        self._tokenizer = tokenizer
        self._dictionary = dictionary
        self._min_seq_len = min_seq_len
        self._max_seq_len = max_seq_len
        self._bos_symbol = bos_symbol
        self._eos_symbol = eos_symbol

    def _init_iter_batch(self):

        iter_labels = PayloadContainer(container=self._corpus, payload_key="label")
        iter_text_a = PayloadContainer(container=self._corpus, payload_key="text_a")
        iter_text_b = PayloadContainer(container=self._corpus, payload_key="text_b")

        text_feeder_a = GeneralSentenceFeeder(corpus=iter_text_a, tokenizer=self._tokenizer, dictionary=self._dictionary,
                                                    n_minibatch=1, validation_split=0.,
                                                    min_seq_len=self._min_seq_len, max_seq_len=self._max_seq_len,
                                                    bos_symbol=self._bos_symbol, eos_symbol=self._eos_symbol)
        text_feeder_b = GeneralSentenceFeeder(corpus=iter_text_b, tokenizer=self._tokenizer, dictionary=self._dictionary,
                                                    n_minibatch=1, validation_split=0.,
                                                    min_seq_len=self._min_seq_len, max_seq_len=self._max_seq_len,
                                                    bos_symbol=self._bos_symbol, eos_symbol=self._eos_symbol)
        iter_payloads = zip(text_feeder_a, text_feeder_b, iter_labels)

        for (token_a,_), (token_b,_), label in iter_payloads:
            payload = {
                "text_a":token_a[0],
                "text_b":token_b[0],
                "label":label
            }
            yield payload

    def process_single_sentence(self, sentence: str):
        text_feeder = GeneralSentenceFeeder(corpus=[], tokenizer=self._tokenizer, dictionary=self._dictionary,
                                            n_minibatch=1, validation_split=0.,
                                            min_seq_len=self._min_seq_len, max_seq_len=self._max_seq_len,
                                            bos_symbol=self._bos_symbol, eos_symbol=self._eos_symbol)
        lst_token_idx = text_feeder.process_single_sentence(sentence)
        return lst_token_idx
