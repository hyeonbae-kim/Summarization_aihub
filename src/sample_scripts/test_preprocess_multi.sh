#!/usr/bin/env bash

# common
LOG_PATH="../test_logs"
mkdir -p ${LOG_PATH}

# Clean data & Split sentences
DATASET=report
SUB_DATASET="news"
if [ ${DATASET} = "report" ]; then
    news_VALID_END_IDX=-1
    briefing_VALID_END_IDX=-1
    hiscul_VALID_END_IDX=-1
    paper_VALID_END_IDX=-1
    minute_VALID_END_IDX=-1
    edit_VALID_END_IDX=-1
    public_VALID_END_IDX=-1
    speech_VALID_END_IDX=-1
    literature_VALID_END_IDX=-1
    narration_VALID_END_IDX=-1
fi

DATA_TYPE="train valid test"
for data_type in $DATA_TYPE
do
    for sub in $SUB_DATASET
    do
        VALID_END_IDX_STR=${sub}_VALID_END_IDX
        VALID_END_IDX=${!VALID_END_IDX_STR}                 # Indirect variable reference
        if [ $VALID_END_IDX -ge 0 ]; then
            TEST_START_IDX=$((VALID_END_IDX+1))
            TEST_END_IDX=$((TEST_START_IDX+VALID_END_IDX))
        else
            TEST_START_IDX=0
            TEST_END_IDX=$VALID_END_IDX
        fi

        RAW_PATH="../sample_data/${DATASET}/${data_type}/${sub}"
        SAVE_PATH="../test_json_data/${DATASET}"
        mkdir -p ${SAVE_PATH}
        python -m prepro.split_sentence \
            -raw_path $RAW_PATH \
            -save_path $SAVE_PATH \
            -save_pattern ${DATASET}_${sub}.${data_type}.{}.json \
            -log_file ${LOG_PATH}/split_sentence_multi_${DATASET}_${sub}.log \
            -data_type ${DATASET} \
            -chunk_size 100 \
            -clean_data False \
            -n_cpus 15

        # Split valid and test data
        for i in $(seq 0 $VALID_END_IDX)
        do
            mv ${SAVE_PATH}/${DATASET}_${sub}.train.${i}.json ${SAVE_PATH}/${DATASET}_${sub}.valid.${i}.json
        done

        for i in $(seq $TEST_START_IDX $TEST_END_IDX)
        do
            mv ${SAVE_PATH}/${DATASET}_${sub}.train.${i}.json ${SAVE_PATH}/${DATASET}_${sub}.test.${i}.json
        done
    done
done

# Tokenizing & Format to PyTorch files
RAW_PATH="../test_json_data/${DATASET}"
SAVE_PATH="../test_bert_data/${DATASET}"
mkdir -p ${SAVE_PATH}
python -m preprocess \
	-mode format_to_bert \
	-raw_path $RAW_PATH \
	-save_path $SAVE_PATH \
	-n_cpus 15 \
	-log_file ${LOG_PATH}/preprocess_multi_${DATASET}.log \
	-tokenizer multi \
	-tgt_bos [unused1] \
	-tgt_eos [unused2] \
	-tgt_sent_split [unused3]

