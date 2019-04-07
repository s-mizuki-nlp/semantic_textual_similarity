#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import argparse
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

from pprint import pprint
from preprocess.corpora import Dictionary
from elmo2gauss.trainer import ELMo2Gauss
from preprocess.tokenizer import CharacterTokenizer
from common.loader.text import TextLoader
from preprocess.dataset_feeder import GeneralSentenceFeeder


def _parse_args():

    parser = argparse.ArgumentParser(description="train context-independent word embeddings by Gaussian distribution using ELMo embeddings")
    parser.add_argument("--elmo_weight", required=True, type=str, help="path to the ELMo model weight file.")
    parser.add_argument("--elmo_config", required=True, type=str, help="path to the ELMo model config file.")
    parser.add_argument("--elmo_extract_layer_ids", required=False, type=str, default="0,1,2", help="layer ids of ELMo embeddings. DEFAULT: `0,1,2`")
    parser.add_argument("--elmo_pooling_method", required=False, type=str, default="mean", choices=ELMo2Gauss.available_pooling_methods(),
                        help="pooling method applied to extracted embeddings. DEFAULT: `mean`")
    parser.add_argument("--corpus", "-c", required=True, type=str, help="path to the pre-tokenized corpus.")
    parser.add_argument("--dictionary", "-d", required=True, type=str, help="path to the pre-defined dictionary.")
    parser.add_argument("--do_lower_case", required=False, type=bool, default=False, help="lowercase or not. default: False")
    parser.add_argument("--n_minibatch", required=False, type=int, default=128, help="minibatch size when encoding sentences.")
    parser.add_argument("--cuda_device", required=False, type=int, default=-1, help="cuda device ID. default: -1 (=disabled)")
    parser.add_argument("--save", "-s", required=True, type=str, help="path to the trained ELMo2Gauss encoder.")
    args = parser.parse_args()

    args.elmo_extract_layer_ids = tuple(map(int, args.elmo_extract_layer_ids.split(",")))

    return args


def main():

    args = _parse_args()
    pprint(vars(args))
    print(f"ELMo2Gauss model will be saved as: {args.save}")

    # ELMo embedder
    elmo = ElmoEmbedder(args.elmo_config, args.elmo_weight, args.cuda_device)
    # pre-defined vocabulary
    dictionary = Dictionary.load(args.dictionary)
    # corpus and tokenizer
    corpus = TextLoader(file_path=args.corpus)
    tokenizer = CharacterTokenizer(do_lower_case=args.do_lower_case)
    feeder = GeneralSentenceFeeder(corpus=corpus, tokenizer=tokenizer, n_minibatch=args.n_minibatch)

    # instanciate ELMo2Gauss encoder
    encoder = ELMo2Gauss(model_elmo=elmo, dictionary=dictionary,
                         extract_layer_ids=args.elmo_extract_layer_ids, pooling_method=args.elmo_pooling_method, verbose=True)
    print("start training...")
    encoder.train(dataset_feeder=feeder)
    encoder.save(args.save)
    print("finished. good-bye.")


if __name__ == "__main__":
    main()
