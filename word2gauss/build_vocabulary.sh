#!/usr/bin/env bash

./build_vocabulary.py \
--corpus=/home/sakae/Windows/dataset/hypernym_detection/semeval_2018/corpus/UMBC_tokenized_utf8.txt \
--model=/home/sakae/Windows/public_model/gaussian_embedding/umbc_200k_100.gz \
--n_vocab=200000 \
--n_min_freq=100