#!/usr/bin/env bash
size=large #Can be: small OR large
type=unique # can be unique or repetition

CURRENT_DIR=`pwd`;
HOME_DIR=..
DATA_DIR=../../data/$type/split/$size  # change here $type 
OUTPUT_DIR=$HOME_DIR/processed_data/$size
SPM_DIR=${HOME_DIR}/sentencepiece;
MAX_LEN=510

TOK_DIR=$SPM_DIR
DICT_FILE=${TOK_DIR}/dict.txt # dict.txt
if [[ ! -f $DICT_FILE ]]; then
    SPM_VOCAB=${TOK_DIR}/sentencepiece.bpe.vocab
    cut -f1 $SPM_VOCAB | tail -n +4 | sed "s/$/ 100/g" > $DICT_FILE
fi


function spm_preprocess () {
    for SPLIT in train val test; do
        python encode.py \
            --model-file ${SPM_DIR}/sentencepiece.bpe.model \
            --src_file $DATA_DIR/src-${SPLIT}.txt \
            --tgt_file $DATA_DIR/tgt-${SPLIT}.txt \
            --output_dir $OUTPUT_DIR \
            --src_lang source --tgt_lang target \
            --pref $SPLIT --max_len $MAX_LEN \
            --workers 60;
    done
}

function binarize () {


    fairseq-preprocess \
        --source-lang source \
        --target-lang target \
        --trainpref $OUTPUT_DIR/train.spm \
        --validpref $OUTPUT_DIR/val.spm \
        --testpref $OUTPUT_DIR/test.spm \
        --destdir $OUTPUT_DIR/data-bin \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --workers 60 \
        --srcdict ${SPM_DIR}/dict.txt \
        --tgtdict ${SPM_DIR}/dict.txt;

}

spm_preprocess
binarize
