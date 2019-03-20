#!/usr/bin/env bash

./train_model.py \
--corpus=/home/sakae/dataset/hypernym_detection/semeval_2018/corpus/UMBC_tokenized_utf8.txt \
--vocab=/home/sakae/dataset/public_model/gaussian_embedding/umbc_200k_100.gz \
--model=/home/sakae/dataset/public_model/gaussian_embedding/umbc_200k_100_spherical_20.w2g \
--n_dim=100 \
--n_thread=22 \
--cov_type=spherical

./train_model.py \
--corpus=/home/sakae/dataset/hypernym_detection/semeval_2018/corpus/UMBC_tokenized_utf8.txt \
--vocab=/home/sakae/dataset/public_model/gaussian_embedding/umbc_200k_100.gz \
--model=/home/sakae/dataset/public_model/gaussian_embedding/umbc_200k_100_diagonal_20.w2g \
--n_dim=100 \
--n_thread=22 \
--cov_type=diagonal

./train_model.py \
--corpus=/home/sakae/dataset/hypernym_detection/semeval_2018/corpus/UMBC_tokenized_utf8.txt \
--vocab=/home/sakae/dataset/public_model/gaussian_embedding/umbc_2m_10.gz \
--model=/home/sakae/dataset/public_model/gaussian_embedding/umbc_2m_100_diagonal_20.w2g \
--n_dim=100 \
--n_thread=22 \
--cov_type=diagonal
