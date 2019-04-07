#!/bin/sh

ELMO_DIR="/home/sakae/Windows/public_model/ELMo"
CORPUS_DIR="/home/sakae/Windows/dataset/language_modeling"
MODEL_DIR="/home/sakae/Windows/public_model"

python ./train_elmo2gauss.py \
--elmo_weight="${ELMO_DIR}/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5" \
--elmo_config="${ELMO_DIR}/elmo_2x1024_128_2048cnn_1xhighway_options.json" \
--elmo_extract_layer_ids="0,1,2" \
--elmo_pooling_method="mean" \
--corpus="${CORPUS_DIR}/umbc/sample.txt" \
--dictionary="${CORPUS_DIR}/umbc/vocabulary_freq10.pkl" \
--save="${MODEL_DIR}/ELMo2Gauss/umbc_freq10_elmo_2x1024_128_2048_layer0-1-2_mean.model" \
--do_lower_case=False \
--n_minibatch=256
