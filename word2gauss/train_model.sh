#!/usr/bin/env bash

./train_model.py \
--corpus=/home/sakae/Windows/dataset/hypernym_detection/semeval_2018/corpus/sample.txt \
--vocab=/home/sakae/Windows/public_model/gaussian_embedding/sample.gz \
--model=/home/sakae/Windows/public_model/gaussian_embedding/sample.w2g \
--n_dim=100 \
--cov_type=diagonal