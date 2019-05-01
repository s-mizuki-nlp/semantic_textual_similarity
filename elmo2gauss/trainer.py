#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional, Tuple, Iterable, Callable, Any, Union

import os, sys, io
import pickle

import warnings
import progressbar
from allennlp.commands.elmo import ElmoEmbedder
from preprocess.corpora import Dictionary
from preprocess.dataset_feeder import GeneralSentenceFeeder
import numpy as np
from distribution.continuous import MultiVariateNormal
from distribution import distance


def _ignore_error(func: Callable[[Any], Any], *args, **kwargs):
    try:
        return func.__call__(*args, **kwargs)
    except:
        return None


class ELMo2Gauss(object):

    __min_cov_value = 1E-10
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
        self._init_sims = False

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
            "count": np.zeros(shape=(0,), dtype=np.int64),
            "l2_norm_mean": np.zeros(shape=(0,), dtype=np.float32),
            "l2_norm_var": np.zeros(shape=(0,), dtype=np.float32)
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
        cov = np.maximum(cov, self.__min_cov_value)
        logdet = np.sum(np.log(cov))

        if "l2_norm_mean" in self._w2g:
            l2_mean = self._w2g["l2_norm_mean"][idx]
            l2_var = self._w2g["l2_norm_var"][idx]
            return (mean, cov, logdet, count, l2_mean, l2_var)
        else:
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

    @property
    def n_vocab_wo_special(self):
         return self._n_vocab - (self._dictionary.offset + 1)

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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_elmo"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._elmo = None

    @property
    def elmo_embedder(self):
        return self._elmo

    @elmo_embedder.setter
    def elmo_embedder(self, model_elmo):
        self._elmo = model_elmo

    def _mask_oov_word(self, sentence: List[str], oov_symbol: Union[str,None] = "<oov>"):

        def _mask(s: str):
            try:
                _ = self.__getitem__(s)
                return s
            except:
                return oov_symbol

        return list(map(_mask, sentence))

    def _remove_oov_word(self, sentence: List[str]) -> List[str]:
        sentence = self._mask_oov_word(sentence, oov_symbol=None)
        return list(filter(bool, sentence))

    def _calc_total_freq(self):
        if self._total_freq is not None:
            return self._total_freq
        ret = sum(self._w2g["count"])
        self._total_freq = ret
        return ret

    def _calc_min_freq(self):
        if self._min_freq is not None:
            return self._min_freq
        ret = min(self._w2g["count"])
        self._min_freq = ret
        return ret

    def _word_to_freq(self, word):
        """
        if oov, it returns None. else it returns word frequency within the corpus which is used to estimate gaussian embeddings.
        """
        try:
            freq = self.__getitem__(word)[3]
            return freq
        except:
            return None

    def weighting_unif(self, sentence: List[str]):
        n_word = len(sentence)
        return np.full(shape=n_word, fill_value=1./n_word, dtype=np.float64)

    def weighting_sif(self, sentence: List[str], alpha: float = 1E-4, scale: bool = False, ignore_oov: bool = False):
        n_count = self._calc_total_freq()
        def word_to_weight(word):
            freq = self._word_to_freq(word)
            if freq is None:
                w = None
            else:
                w = alpha / (alpha + freq/n_count)
            return w

        if ignore_oov:
            vec_weight = np.array(list(filter(bool, map(word_to_weight, sentence))))
        else:
            vec_weight = np.array(list(map(word_to_weight, sentence)))
        if scale:
            vec_weight = vec_weight / np.nansum(vec_weight)

        return vec_weight

    def init_params(self):
        assert len(self._w2g["count"]) == 0, "model parameters are already trained. abort."

        shp = (self._dictionary.max_id+1, self.vector_size)
        self._w2g["mu"] = np.zeros(shape=shp, dtype=self.__w2g_dtype)
        self._w2g["sigma"] = np.zeros(shape=shp, dtype=self.__w2g_dtype)
        self._w2g["count"] = np.zeros(shape=shp[0], dtype=np.int64)
        self._w2g["l2_norm_mean"] = np.zeros(shape=shp[0], dtype=self.__w2g_dtype)
        self._w2g["l2_norm_var"] = np.zeros(shape=shp[0], dtype=self.__w2g_dtype)
        self._total_freq = None
        self._min_freq = None


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
            self._init_sims = True
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
        new_instance._init_sims = True

        return new_instance

    def _normalize(self, mat_w2v):
        return mat_w2v / np.linalg.norm(mat_w2v, axis=1, keepdims=True)

    def _subtract_mean(self, mat_w2v):
        return mat_w2v - np.mean(mat_w2v, axis=0, keepdims=True)

    def word_to_gaussian(self, word: str):
        mu, cov = self.__getitem__(word)[:2]
        p_x = MultiVariateNormal(vec_mu=mu, vec_cov=cov)
        return p_x

    def sentence_to_gaussians(self, sentence: List[str], ignore_error: bool = True) -> List["MultiVariateNormal"]:
        """
        encode sentence into list of gaussian distributions.
        WARNING: mean and covariance is context-independent.

        :param sentence: list of tokens.
        :param ignore_error: ignore out-of-vocabulary token(=True) or replace with None(=False)
        :return: list of MultiVariateNormal class instances.
        """
        if ignore_error:
            lst_ret = [_ignore_error(self.word_to_gaussian, word) for word in sentence]
        else:
            lst_ret = [self.word_to_gaussian(word) for word in sentence]

        return lst_ret

    def sentence_to_gaussian_mixture(self, sentence: List[str], weighting_method: str = "unif"):

        # remove oov tokens
        sentence = self._remove_oov_word(sentence)
        lst_mvn = self.sentence_to_gaussians(sentence, ignore_error=True)

        if weighting_method == "unif":
            vec_weight = self.weighting_unif(sentence)
        elif weighting_method == "sif":
            vec_weight = self.weighting_sif(sentence, scale=True)
        else:
            raise NotImplementedError(f"unknown weighting method was specified: {weighting_method}")

        gmm = MultiVariateNormal.mvn_to_gmm(lst_mvn, vec_weight)
        return gmm

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

        assert self._elmo is not None, "you have to assign ElmoEmbedder class instance."
        assert normalize == False, "experimental: do not enable word-level normalization."
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
        vec_l2_norm_mean = self._w2g["l2_norm_mean"]
        vec_l2_norm_var = self._w2g["l2_norm_var"]

        n_sentence = dataset_feeder.size
        n_processed = 0
        q = progressbar.ProgressBar(max_value=n_sentence+1)
        for lst_lst_tokens, _ in dataset_feeder:
            lst_mat_w2v = self.sentence_to_word_vectors_batch(lst_lst_tokens, normalize, subtract_sentence_mean)
            for lst_tokens, mat_w2v in zip(lst_lst_tokens, lst_mat_w2v):
                lst_token_idx = self._dictionary.transform(lst_tokens)
                vec_w2v_l2_norm = np.linalg.norm(mat_w2v, axis=-1)
                for token_idx, vec_w, l2_norm in zip(lst_token_idx, mat_w2v, vec_w2v_l2_norm):
                    if token_idx == oov_id:
                        continue

                    vec_count[token_idx] += 1
                    mat_mu[token_idx] += vec_w # \sum{v_w}
                    mat_sigma[token_idx] += vec_w**2 # \sum{v_w^2}
                    vec_l2_norm_mean[token_idx] += l2_norm
                    vec_l2_norm_var[token_idx] += l2_norm**2

            n_processed += len(lst_lst_tokens)
            q.update(n_processed)

        # calculate mean and variance
        mat_mu /= vec_count.reshape(-1,1) # mat_mu[idx(w)] = \sum{v_w} / freq[w]
        mat_sigma /= vec_count.reshape(-1,1) # mat_sigma[idx(w)] = \sum{v_w^2} / freq[w]
        mat_sigma -= mat_mu**2 # mat_sigma[idx(w)] = \sum{v_w^2} / freq[w] - mat_mu[idx(w)]^2
        vec_l2_norm_mean /= vec_count # vec_l2_norm_mean[idx(w)] = \sum{|v_w|} / freq[w]
        vec_l2_norm_var /= vec_count
        vec_l2_norm_var -= vec_l2_norm_mean**2 # vec_l2_norm_var[idx(w)] = \sum{|v_w|^2} / freq[w] - vec_l2_norm_mean[idx(w)]^2

        # (optional) adjust degree of freedom
        if ddof > 0:
            vec_dof_adjust = vec_count / np.maximum(vec_count-1, 1)
            mat_sigma *= vec_dof_adjust.reshape(-1,1)
            vec_l2_norm_var *= vec_dof_adjust

        if self._verbose:
            print("finished training.")

        return True

    def _cosine_similarity(self, vec_mean_w: np.ndarray):
        assert self._init_sims, "please call `init_sims(inplace=True)` beforehand."
        vec_sim = self._w2g["mu"].dot(vec_mean_w)
        return vec_sim

    def _expected_likelihood_kernel(self, vec_mean_w: np.ndarray, vec_cov_w: np.ndarray):

        mat_mu_x = vec_mean_w.reshape(1, -1)
        mat_cov_x = vec_cov_w.reshape(1, -1)
        mat_mu_y = self._w2g["mu"]
        mat_cov_y = self._w2g["sigma"]
        vec_sim = distance._expected_likelihood_kernel_multivariate_normal_diag_parallel(mat_mu_x, mat_cov_x, mat_mu_y, mat_cov_y)
        return vec_sim

    def _wasserstein_distance_similarity(self, vec_mean_w: np.ndarray, vec_cov_w: np.ndarray):
        # the larger, the more similar
        mat_mu_x = vec_mean_w.reshape(1, -1)
        mat_std_x = np.sqrt(vec_cov_w).reshape(1, -1)
        mat_mu_y = self._w2g["mu"]
        mat_std_y = np.sqrt(self._w2g["sigma"])
        vec_dist = distance._wasserstein_distance_sq_between_multivariate_normal_diag_parallel(mat_mu_x, mat_std_x, mat_mu_y, mat_std_y)
        return -vec_dist

    def most_similar(self, word: str, topn: int = 10, similarity: str = "cosine"):

        vec_mean_w, vec_cov_w, _, _ = self.__getitem__(word)
        if similarity == "cosine":
            vec_sim = self._cosine_similarity(vec_mean_w)
        elif similarity == "elk":
            vec_sim = self._expected_likelihood_kernel(vec_mean_w, vec_cov_w)
        elif similarity == "wd_sq":
            vec_sim = self._wasserstein_distance_similarity(vec_mean_w, vec_cov_w)
        else:
            raise NotImplementedError(f"unknown similarity measure was specified: {similarity}")

        vec_idx = np.argsort(-vec_sim)[:topn]
        vec_result_sim = vec_sim[vec_idx]
        lst_result_words = self._dictionary.inverse_transform(vec_idx)
        ret = list(zip(lst_result_words, vec_result_sim))

        return ret
