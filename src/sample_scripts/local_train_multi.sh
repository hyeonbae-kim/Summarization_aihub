#!/usr/bin/env bash

# common
LOG_PATH="../logs"
mkdir -p ${LOG_PATH}

# Train model
DATASET=report
#DATASET=broadcast
TASK=abs
#TASK=ext
# ‘^’ symbol is used to convert the first character of any string to uppercase and
# ‘^^’ symbol is used to convert the whole string to the uppercase.
MODEL_PATH="../models/MultiSum${TASK^}_${DATASET}_512"
DATA_PATH="../bert_data/${DATASET}/${DATASET}"     # 파일 prefix까지 (예. bert_data/report/report_novel.train.9.bert.pt)
python -m train \
	-task $TASK \
	-mode train \
	-model_path $MODEL_PATH \
	-result_path ../results/ \
	-bert_data_path $DATA_PATH \
	-dec_dropout 0.2 \
	-sep_optim true \
	-lr_bert 0.002 \
	-lr_dec 0.2 \
	-save_checkpoint_steps 10 \
	-batch_size 8 \
	-train_steps 1000 \
	-report_every 10 \
	-accum_count 10 \
	-use_bert_emb true \
	-use_interval true \
	-warmup_steps_bert 1 \
	-warmup_steps_dec 1 \
	-max_pos 256 \
	-visible_gpus 0 \
	-log_file ${LOG_PATH}/train_${TASK}_multi_${DATASET}_512 \
	-tokenizer multi \
	-tgt_bos [unused1] \
	-tgt_eos [unused2] \
	-tgt_sent_split [unused3]
