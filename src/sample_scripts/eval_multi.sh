#!/usr/bin/env bash

# common
LOG_PATH="../logs"
mkdir -p ${LOG_PATH}

# Evaluation with test datasets
DATASET=report
TASK=abs
#TASK=ext
MODE=test
# ‘^’ symbol is used to convert the first character of any string to uppercase and
# ‘^^’ symbol is used to convert the whole string to the uppercase.
MODEL_PATH="../models/MultiSum${TASK^}_${DATASET}_512"
DATA_PATH="../bert_data/${DATASET}/${DATASET}"     # 파일 prefix까지 (예. bert_data/report/report_novel.test.3.bert.pt)

python -m train \
	-task $TASK \
	-mode $MODE \
	-test_all False \
	-test_from ${MODEL_PATH}/model_step_20000.pt \
	-batch_size 10 \
	-test_batch_size 10 \
	-bert_data_path ${DATA_PATH} \
	-log_file ${LOG_PATH}/eval_${TASK}_multi_${DATASET}_512 \
	-model_path ${MODEL_PATH} \
	-sep_optim true \
	-use_interval true \
	-visible_gpus 0 \
	-max_pos 256 \
	-max_length 200 \
	-alpha 0.95 \
	-min_length 8 \
	-result_path ../logs/ \
	-tokenizer multi \
	-tgt_bos [unused1] \
	-tgt_eos [unused2] \
	-tgt_sent_split [unused3] \
	-beam_size 5
