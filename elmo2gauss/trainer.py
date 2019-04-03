#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional, Tuple, Iterable

import os, sys, io
import pickle

import warnings
import progressbar
from allennlp.commands.elmo import ElmoEmbedder
from preprocess.corpora import Dictionary
from preprocess.dataset_feeder import GeneralSentenceFeeder
import numpy as np


class ELMo2Gauss(object):

    __w2g_dtype = np.float32

    def __init__(self, model_elmo: ElmoEmbedder, dictionary: Dictionary, extract_layer_ids: Tuple[int] = (0,1,2),
                 pooling_method: str = "mean", verbose: bool = False):

        self._elmo = model_elmo
        self._dictionary = dictionary
        self._n_vocab = dictionary.n_vocab
        self._elmo_layer_ids = extract_layer_ids
        self._verbose = verbose

        n_layers = model_elmo.elmo_bilm.num_layers
        assert max(extract_layer_ids) < n_layers, f"valid layer id is 0 to {n_layers-1}."

        lst_accepted_pooling_method = "mean,max,concat".split(",")
        assert pooling_method in lst_accepted_pooling_method, "invalid pooling method was specified. valid methods are: " + ",".join(lst_accepted_pooling_method)
        if pooling_method == "mean":
            self._pooling_function = lambda t_mat_w2v: np.mean(t_mat_w2v[self._elmo_layer_ids,:,:], axis=0)
            self._vector_size = self._elmo.elmo_bilm.get_output_dim()
        elif pooling_method == "max":
            self._pooling_function = lambda t_mat_w2v: np.max(t_mat_w2v[self._elmo_layer_ids,:,:], axis=0)
            self._vector_size = self._elmo.elmo_bilm.get_output_dim()
        elif pooling_method == "concat":
            self._pooling_function = lambda t_mat_w2v: np.hstack(t_mat_w2v[self._elmo_layer_ids,:,:])
            self._vector_size = self._elmo.elmo_bilm.get_output_dim() * len(extract_layer_ids)

        self._w2g = {
            "mu": np.zeros(shape=(0, self._vector_size), dtype=np.float32),
            "sigma": np.zeros(shape=(0, self._vector_size), dtype=np.float32),
            "count": np.zeros(shape=(0,), dtype=np.int64)
        }

    @property
    def vector_size(self):
        return self._vector_size

    @property
    def n_vocab(self):
        return self._n_vocab

    def save(self, file_path: str):
        if len(self._w2g["count"]) == 0:
            warnings.warn("distribution parameter is empty. did you call `train()` method?")
        with io.open(file_path, mode="wb") as ofs:
            pickle.dump(self, ofs)

    @classmethod
    def load(cls, file_path: str):
        with io.open(file_path, mode="rb") as ifs:
            obj = pickle.load(ifs)
        obj.__class__ = cls
        return obj

    def init_params(self):
        assert len(self._w2g["count"]) == 0, "model parameters are already trained. abort."

        shp = (self.n_vocab, self.vector_size)
        self._w2g["mu"] = np.zeros(shape=shp, dtype=self.__w2g_dtype)
        self._w2g["sigma"] = np.zeros(shape=shp, dtype=self.__w2g_dtype)
        self._w2g["count"] = np.zeros(shape=(self.n_vocab,), dtype=np.int64)

    def sentence_to_word_vectors(self, sentence: List[str], normalize: bool = False, subtract_sentence_mean: bool = False):
        """
        encode sentence into word vectors.
        you can specify which ELMo layer to extract by passing layer ids to `extract_layer_ids` argument.
        output of each layer will be concatenated horizontally.

        :param sentence: list of tokens.
        :param extract_layer_ids: tuple of layer ids. max id will be 2.
        :param normalize:
        :return: word vectors. size will be (n_tokens, n_dim*len(extract_layer_ids))
        """
        mat_w2v = self._elmo.embed_sentence(sentence)
        # mat_w2v = np.hstack(mat_w2v[self._elmo_layer_ids,:,:])
        mat_w2v = self._pooling_function(mat_w2v)

        if normalize:
            mat_w2v = mat_w2v / np.linalg.norm(mat_w2v, axis=1, keepdims=True)
        if subtract_sentence_mean:
            mat_w2v = mat_w2v - np.mean(mat_w2v, axis=0, keepdims=True)

        return mat_w2v

    def train(self, dataset_feeder: GeneralSentenceFeeder):

        assert dataset_feeder._dictionary is None, "dataset feeder mustn't have dictionary instance."
        if dataset_feeder._validation_split != 0.0:
            warnings.warn("you should disable validation split.")

        # initialize parameters
        self.init_params()

        oov_id = self._dictionary.oov_id
        mat_mu = self._w2g["mu"]
        mat_sigma = self._w2g["sigma"]
        vec_count = self._w2g["count"]

        n_sentence = dataset_feeder.size
        n_processed = 0
        q = progressbar.ProgressBar(max_value=n_sentence+1)
        for lst_lst_tokens, _ in dataset_feeder:
            lst_mat_s = self._elmo.embed_batch(lst_lst_tokens)
            for lst_tokens, mat_s in zip(lst_lst_tokens, lst_mat_s):
                lst_token_idx = self._dictionary.transform(lst_tokens)
                for token_idx, vec_w in zip(lst_token_idx, mat_s):
                    if token_idx == oov_id:
                        continue

                    vec_count[token_idx] += 1
                    mat_mu[token_idx] += vec_w
                    mat_sigma[token_idx] += vec_w**2

            n_processed += len(lst_lst_tokens)
            q.update(n_processed)