#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional
from gensim.models import Word2Vec
import numpy as np


class BagOfWordVectors(object):

    def __init__(self, model_word2vec: Word2Vec, init_sims=False):

        self._w2v = model_word2vec
        self._init_sims = init_sims
        if init_sims:
            self._w2v.init_sims(replace=False)
        self._total_freq = None

    def sentence_to_word_vectors(self, sentence: List[str], normalize: bool = False):

        vec_idx = np.array([self._w2v.wv.vocab[word].index for word in sentence if word in self._w2v])

        if normalize:
            if self._init_sims:
                mat_w2v = self._w2v.wv.syn0norm[vec_idx,:]
            else:
                mat_w2v = self._w2v.wv.syn0[vec_idx,:]
                mat_w2v = mat_w2v / np.linalg.norm(mat_w2v, axis=1, keepdims=True)
        else:
            mat_w2v = self._w2v.wv.syn0[vec_idx,:]

        return mat_w2v

    def cbow(self, sentence: List[str], normalize: bool=False, init_sims: bool=True):

        mat_w2v = self.sentence_to_word_vectors(sentence, normalize=init_sims)
        vec_ret = np.mean(mat_w2v, axis=0)
        if normalize:
            vec_ret = vec_ret / np.linalg.norm(vec_ret)

        return vec_ret


    def _calc_total_freq(self):
        if self._total_freq is not None:
            return self._total_freq

        ret = sum([word.count for word in self._w2v.wv.vocab.values()])
        self._total_freq = ret
        return ret

    def _word_to_freq(self, word):
        if word in self._w2v:
            return self._w2v.wv.vocab[word].count
        else:
            return None

    def weighting_sif(self, sentence: List[str], alpha: float = 1E-4, scale: bool = False):

        n_count = self._calc_total_freq()

        def word_to_weight(word):
            freq = self._word_to_freq(word)
            if freq is None:
                w = None
            else:
                w = alpha / (alpha + freq/n_count)
            return w

        vec_weight = np.array(list(filter(bool, map(word_to_weight, sentence))))
        if scale:
            vec_weight = vec_weight / np.sum(vec_weight)

        return vec_weight

    def weighting_freq(self, sentence: List[str], alpha: float=1.0):

        vec_freq = np.array( list(map(self._word_to_freq, sentence)) )
        if alpha != 1.0:
            vec_freq = vec_freq**alpha
        vec_weight = vec_freq / np.sum(vec_freq)

        return vec_weight

    def weighting_unif(self, sentence: List[str]):
        n_word = sum([word in self._w2v for word in sentence])

        return np.full(shape=n_word, fill_value=1./n_word, dtype=np.float64)


    def _sif_weighting_vector(self, sentence: List[str], alpha: float, normalize: bool=False, init_sims: bool=True):

        mat_w2v = self.sentence_to_word_vectors(sentence, normalize=init_sims)
        vec_weight = self.weighting_sif(sentence, alpha, scale=False)
        mat_w2v = mat_w2v * vec_weight.reshape(-1,1)
        vec_ret = np.mean(mat_w2v, axis=0)
        if normalize:
            vec_ret = vec_ret / np.linalg.norm(vec_ret)

        return vec_ret

    def _usif_weighting_vector(self, sentence: List[str], normalize: bool=False, init_sims: bool=True):
        raise NotImplementedError("not implemented yet.")

    def _calc_singular_vector_from_corpus(self, corpus: List[List[str]], method_name: str="sif", n_sv: int=1,
                                          return_eigenvalues: bool=False, **kwargs):

        n_sentence = len(corpus)
        mat_corpus = np.zeros(shape=(n_sentence, self._w2v.vector_size), dtype=np.float32)
        for idx, sentence in enumerate(corpus):
            if method_name == "sif":
                mat_corpus[idx] = self._sif_weighting_vector(sentence, **kwargs)
            elif method_name == "usif":
                mat_corpus[idx] = self._usif_weighting_vector(sentence, **kwargs)
            else:
                raise NotImplementedError(f"unsupported method name was specified: {method_name}")

        # first singular vector
        _, vec_lambda, mat_v = np.linalg.svd(mat_corpus, full_matrices=False)
        mat_ret = mat_v[:n_sv,:]
        vec_ret = vec_lambda[:n_sv]

        if return_eigenvalues:
            return vec_ret, mat_ret
        else:
            return mat_ret

    def sif(self, sentence: List[str], alpha: float = 1E-4, corpus: Optional[List[List[str]]] = None,
            singular_vector: Optional[np.ndarray] = None, normalize: bool=False, init_sims: bool=True):

        vec_s = self._sif_weighting_vector(sentence, alpha, init_sims=init_sims)

        if corpus is not None:
            vec_u = self._calc_singular_vector_from_corpus(corpus, alpha=alpha)
        else:
            vec_u = singular_vector

        if vec_u is not None:
            vec_s = vec_s - vec_u * np.dot(vec_s, vec_u)

        if normalize:
            vec_s = vec_s / np.linalg.norm(vec_s)

        return vec_s
