#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from vocab import Vocabulary
from word2gauss import GaussianEmbedding, iter_pairs
import argparse
import numpy as np
import pandas as pd
import psutil
import json
from pprint import pprint

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

def _parse_args():
    parser = argparse.ArgumentParser(description="build vocabulary ")
    parser.add_argument("--corpus", "-c", required=True, type=str, help="path to the tokenized corpus.")
    parser.add_argument("--vocab", "-v", required=True, type=str, help="pre-built vocabulry file.")
    parser.add_argument("--model", "-m", required=True, type=str, help="path to the word2gauss model to be saved.")
    parser.add_argument("--n_dim", "-n", required=True, type=int, help="number of embedding dimensions.")
    parser.add_argument("--cov_type", required=True, type=str, choices=["spherical","diagonal"], help="covariance type.")
    parser.add_argument("--n_thread", required=False, type=int, default=psutil.cpu_count(logical=False), help="number of threads.")
    parser.add_argument("--kwargs", required=False, type=int, default=None, help="optional keyword arguments to be passed to GaussianEmbedding class.")
    args = parser.parse_args()
    return args


def main():

    args = _parse_args()

    assert not(os.path.exists(args.model)), f"specified file already exists: {args.model}"

    pprint(args.__dict__)

    vocab_params = {
        "power":0.75
    }
    vocab = Vocabulary.load(args.vocab, **vocab_params)
    n_vocab = len(vocab)
    print(f"vocabulary size: {n_vocab}")

    kwargs = {} if args.kwargs is None else json.loads(args.kwargs)
    pprint(kwargs)

    init_params = {
        'mu0': 0.1,
        'sigma_mean0': 1.0,
        'sigma_std0': 0.01
    }
    model_params = {
        "mu_max":1.0,
        "sigma_min":0.1,
        "sigma_max":10.0,
        "eta":0.01,
        "Closs":4.0
    }

    print("start training...")
    model = GaussianEmbedding(n_vocab, args.n_dim, covariance_type=args.cov_type, energy_type="KL", init_params=init_params, **model_params)
    with io.open(args.corpus, mode="r") as corpus:
        it = iter_pairs(corpus, vocab, batch_size=20, nsamples=20, window=5)
        model.train(it, n_workers=args.n_thread)

    print(f"finished. saving models: {args.model}")
    model.save(args.model)

    # sanity check
    print("done. now execute sanity check...")

    def ln_det_sigma(word):
        vec_sigma = model.sigma[vocab.word2id(word)]
        return np.sum(np.log(vec_sigma))
    
    w = "food"
    print(f"word: {w}")
    lst_result = model.nearest_neighbors(w, vocab=vocab, sort_order="sigma", num=100)
    df_result = pd.DataFrame(lst_result)
    df_result["sigma_ln_det"] = df_result["word"].map(ln_det_sigma)
    print(df_result.sort_values(by="sigma_ln_det", ascending=False).head(n=10))

    print("finished. good-bye.")


if __name__ == "__main__":
    main()
