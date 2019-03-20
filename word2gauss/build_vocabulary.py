#!/usr/bin/env python
# -*- coding:utf-8 -*-



import os, sys, io
from vocab import Vocabulary
import argparse
import numpy as np

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

def _parse_args():
    parser = argparse.ArgumentParser(description="build vocabulary ")
    parser.add_argument("--corpus", "-c", required=True, type=str, help="path to the tokenized corpus.")
    parser.add_argument("--model", "-m", required=True, type=str, help="path to the vocabulary file to be saved.")
    parser.add_argument("--n_vocab", "-n", required=True, type=int, help="maximum number of vocabulary.")
    parser.add_argument("--n_min_freq", "-f", required=True, type=int, help="minimum number of occurence in the corpus.")
    args = parser.parse_args()
    return args


def main():

    args = _parse_args()

    assert not(os.path.exists(args.model)), f"specified file already exists: {args.model}"

    with io.open(args.corpus, mode="r") as corpus:
        v = Vocabulary(table_size=int(2E7))
        v.create(corpus, [(args.n_vocab, args.n_min_freq, args.n_min_freq)])

    print(f"finished. saving models: {args.model}")
    v.save(args.model)

    # sanity check
    print("done. now execute sanity check...")
    print(f"n_vocab: {len(v)}, total_freq:{sum(v.counts)}")

    s = "Knox County Health Department is following national Centers for Disease Control and Prevention Protocol to contain infection."
    print(f"sentence: {s}")
    s_tokenized = "/".join(v.tokenize(s, remove_oov=False))
    print(f"tokenized: {s_tokenized}")
    print(f"random sampling...")
    n_sample = 100
    x = v.random_ids(n_sample)
    w, f = np.unique(list(map(v.id2word, x)), return_counts=True)
    for idx in np.argsort(f)[::-1]:
        print(f"{w[idx]} -> {f[idx]}")

    print("finished. good-bye.")


if __name__ == "__main__":
    main()
