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
from distribution.continuous import MultiVariateNormal


class ELMo2Gauss(object):

    __w2g_dtype = np.float32
    __available_pooling_methods = tuple("mean,max,concat".split(","))

    def __init__(self, model_elmo: ElmoEmbedder, dictionary: Dictionary, extract_layer_ids: Tuple[int] = (0,1,2),
                 pooling_method: str = "mean", verbose: bool = False):

        self._elmo = model_elmo
        self._dictionary = dictionary
        self._n_vocab = dictionary.n_vocab
        self._elmo_layer_ids = extract_layer_ids
        self._pooling_method = pooling_method
        self._verbose = verbose

        n_layers = model_elmo.elmo_bilm.num_layers
        assert max(extract_layer_ids) < n_layers, f"valid layer id is 0 to {n_layers-1}."

        assert pooling_method in self.__available_pooling_methods, \
            "invalid pooling method was specified. valid methods are: " + ",".join(self.__available_pooling_methods)
        if pooling_method == "mean":
            self._pooling_function = self._pool_mean
            self._vector_size = self._elmo.elmo_bilm.get_output_dim()
        elif pooling_method == "max":
            self._pooling_function = self._pool_max
            self._vector_size = self._elmo.elmo_bilm.get_output_dim()
        elif pooling_method == "concat":
            self._pooling_function = self._pool_concat
            self._vector_size = self._elmo.elmo_bilm.get_output_dim() * len(extract_layer_ids)

        self._w2g = {
            "mu": np.zeros(shape=(0, self._vector_size), dtype=np.float32),
            "sigma": np.zeros(shape=(0, self._vector_size), dtype=np.float32),
            "count": np.zeros(shape=(0,), dtype=np.int64)
        }

    def _pool_mean(self, t_mat_w2v: np.ndarray):
        return np.mean(t_mat_w2v[self._elmo_layer_ids,:,:], axis=0)

    def _pool_max(self, t_mat_w2v: np.ndarray):
        return np.max(t_mat_w2v[self._elmo_layer_ids,:,:], axis=0)

    def _pool_concat(self, t_mat_w2v: np.ndarray):
        return np.hstack(t_mat_w2v[self._elmo_layer_ids,:,:])

    def __getitem__(self, word: str):
        idx = self._dictionary[word]
        if idx is None:
            raise KeyError(f"word `{word}` not found in the dictionary.")

        count = self._w2g["count"][idx]
        if count == 0:
            raise KeyError(f"word `{word}` has never observed.")

        mean = self._w2g["mu"][idx]
        cov = self._w2g["sigma"][idx]
        logdet = np.sum(np.log(cov))
        return (mean, cov, logdet, count)

    @classmethod
    def available_pooling_methods(cls):
        return cls.__available_pooling_methods

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

    def init_sims(self, inplace: bool = True):
        """
        initialize word vectors to unit vectors. original parameters will be updated.
        :return: ELMo2Gauss if inplace = True else Nothing
        """
        mat_mu = self._w2g["mu"]
        mat_sigma = self._w2g["sigma"]

        if inplace:
            mat_scale = np.linalg.norm(mat_mu, axis=-1, keepdims=True)
            mat_mu /= mat_scale
            mat_sigma /= mat_scale**2
            return True

        # create new instance
        new_instance = ELMo2Gauss(model_elmo=self._elmo, dictionary=self._dictionary, extract_layer_ids=self._elmo_layer_ids,
                                  pooling_method=self._pooling_method, verbose=self._verbose)
        new_instance.init_params()
        for param_name, value in self._w2g.items():
            new_instance._w2g[param_name] = value.copy()
        mat_mu = new_instance._w2g["mu"]
        mat_sigma = new_instance._w2g["sigma"]
        mat_scale = np.linalg.norm(mat_mu, axis=-1, keepdims=True)
        mat_mu /= mat_scale
        mat_sigma /= mat_scale**2

        return new_instance

    def _normalize(self, mat_w2v):
        return mat_w2v / np.linalg.norm(mat_w2v, axis=1, keepdims=True)

    def _subtract_mean(self, mat_w2v):
        return mat_w2v - np.mean(mat_w2v, axis=0, keepdims=True)

    def sentence_to_word_vectors(self, sentence: List[str], normalize: bool = False, subtract_sentence_mean: bool = False):
        """
        encode sentence into word vectors.
        you can specify which ELMo layer to extract by passing layer ids to `extract_layer_ids` argument.
        output of each layer will be concatenated horizontally.

        :param sentence: list of tokens.
        :param normalize: apply length-normalization on each word vector
        :return: word vectors. size will be (n_tokens, n_dim*len(extract_layer_ids))
        """
        mat_w2v = self._elmo.embed_sentence(sentence)
        mat_w2v = self._pooling_function(mat_w2v)

        if normalize:
            mat_w2v = self._normalize(mat_w2v)
        if subtract_sentence_mean:
            mat_w2v = self._subtract_mean(mat_w2v)

        return mat_w2v

    def sentence_to_word_vectors_batch(self, sentences: List[List[str]], normalize: bool = False, subtract_sentence_mean: bool = False):
        """
        batch process version of sentence encoding method.
        :param sentences: list of sentences. i.e. list of list of tokens.
        :param normalize: apply length-normalization on each word vector
        :param subtract_sentence_mean:
        :return: list of word vectors. each element corresponds to input sentence.
        """
        lst_mat_w2v = self._elmo.embed_batch(sentences)
        lst_mat_w2v = map(self._pooling_function, lst_mat_w2v)

        if normalize:
            lst_mat_w2v = map(self._normalize, lst_mat_w2v)
        if subtract_sentence_mean:
            lst_mat_w2v = map(self._subtract_mean, lst_mat_w2v)

        return list(lst_mat_w2v)

    def train(self, dataset_feeder: GeneralSentenceFeeder,
              normalize: bool = False, subtract_sentence_mean: bool = False, ddof: int = 0):

        assert ddof >= 0, "degree of freedom must be positive integer."
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
            lst_mat_w2v = self.sentence_to_word_vectors_batch(lst_lst_tokens, normalize, subtract_sentence_mean)
            for lst_tokens, mat_w2v in zip(lst_lst_tokens, lst_mat_w2v):
                lst_token_idx = self._dictionary.transform(lst_tokens)
                for token_idx, vec_w in zip(lst_token_idx, mat_w2v):
                    if token_idx == oov_id:
                        continue

                    vec_count[token_idx] += 1
                    mat_mu[token_idx] += vec_w # \sum{v_w}
                    mat_sigma[token_idx] += vec_w**2 # \sum{v_w^2}

            n_processed += len(lst_lst_tokens)
            q.update(n_processed)

        # calculate mean and variance
        mat_mu /= vec_count.reshape(-1,1) # mat_mu[idx(w)] = \sum{v_w} / freq[w]
        mat_sigma /= vec_count.reshape(-1,1) # mat_sigma[idx(w)] = \sum{v_w^2} / freq[w]
        mat_sigma -= mat_mu**2 # mat_sigma[idx(w)] = \sum{v_w^2} / freq[w] - mat_mu[idx(w)]^2

        # (optional) adjust degree of freedom
        if ddof > 0:
            vec_dof_adjust = vec_count / np.maximum(vec_count-1, 1)
            mat_sigma *= vec_dof_adjust.reshape(-1,1)

        if self._verbose:
            print("finished training.")

        return True