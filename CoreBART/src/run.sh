#!/usr/bin/env bash

# change save dir and pretrain path and fp16 if needed

export PYTHONIOENCODING=utf-8;
size=large #Can be: small OR large
type=unique #can be unique or repetition
model=plbart
MAX_LEN=510


CURRENT_DIR=`pwd`;
HOME_DIR=..
DATA_DIR=../../data/$type/split/$size
OUTPUT_DIR=$HOME_DIR/processed_data/$size

PRETRAINED_MODEL_NAME=checkpoint_11_100000.pt
PRETRAIN=${HOME_DIR}/plbart/${PRETRAINED_MODEL_NAME}

SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX


export CUDA_VISIBLE_DEVICES=0

SOURCE=source
TARGET=target

PATH_2_DATA=$HOME_DIR/processed_data/$size

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=${HOME_DIR}/models/$model/$type/${size} # change this $type for inferance with different model
mkdir -p ${SAVE_DIR}

#PRETRAIN=$SAVE_DIR/checkpoint_last.pt

USER_DIR=${HOME_DIR}/user_dir


BATCH_SIZE=4;
UPDATE_FREQ=4;



function fine_tune () {

OUTPUT_FILE=${SAVE_DIR}/finetune.log

# approx. 50k train examples, use a batch size of 16 gives us 3000 steps
# we run for a maximum of 30 epochs
# setting the batch size to 8 with update-freq to 2
# performing validation at every 2000 steps, saving the last 10 checkpoints

fairseq-train $PATH_2_DATA/data-bin \
    --user-dir $USER_DIR \
    --truncate-source \
    --langs $langs \
    --task translation_without_lang_token \
    --arch mbart_base \
    --layernorm-embedding \
    --source-lang $SOURCE \
    --target-lang $TARGET \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --batch-size $BATCH_SIZE \
    --update-freq $UPDATE_FREQ \
    --max-epoch 3000 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay \
    --lr 5e-05 --min-lr -1 \
    --warmup-updates 500 \
    --max-update 200000 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.0 \
    --seed 1234 \
    --log-format json \
    --log-interval 100 \
    --restore-file $PRETRAIN \
    --reset-dataloader \
    --reset-optimizer \
    --reset-meters \
    --reset-lr-scheduler \
    --eval-bleu \
    --eval-bleu-detok space \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-args '{"beam": 5}' \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --patience 100 \
    --ddp-backend no_c10d \
    --save-dir $SAVE_DIR \
    --num-workers 64 \
    2>&1 | tee ${OUTPUT_FILE};

}

function train () {

OUTPUT_FILE=${SAVE_DIR}/train.log

# approx. 50k train examples, use a batch size of 16 gives us 3000 steps
# we run for a maximum of 30 epochs
# setting the batch size to 8 with update-freq to 2
# performing validation at every 2000 steps, saving the last 10 checkpoints

fairseq-train $PATH_2_DATA/data-bin \
    --user-dir $USER_DIR \
    --truncate-source \
    --langs $langs \
    --task translation_without_lang_token \
    --arch mbart_base \
    --layernorm-embedding \
    --source-lang $SOURCE \
    --target-lang $TARGET \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --batch-size $BATCH_SIZE \
    --update-freq $UPDATE_FREQ \
    --max-epoch 3000 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay \
    --lr 5e-05 --min-lr -1 \
    --warmup-updates 500 \
    --max-update 100000 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.0 \
    --seed 1234 \
    --log-format json \
    --log-interval 100 \
    --reset-dataloader \
    --reset-optimizer \
    --reset-meters \
    --reset-lr-scheduler \
    --eval-bleu \
    --eval-bleu-detok space \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-args '{"beam": 5}' \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --patience 10 \
    --ddp-backend no_c10d \
    --num-workers 40 \
    --save-dir $SAVE_DIR \
    --fp16 \
    2>&1 | tee ${OUTPUT_FILE};

}


function generate () {

model=${SAVE_DIR}/checkpoint_best.pt
FILE_PREF=${SAVE_DIR}/output
RESULT_FILE=${SAVE_DIR}/result.txt
GOUND_TRUTH_PATH=../../data/$type/split/$size/tgt-test.txt
echo $GOUND_TRUTH_PATH
echo $FILE_PREF
fairseq-generate $PATH_2_DATA/data-bin \
    --user-dir $USER_DIR \
    --path $model \
    --truncate-source \
    --task translation_without_lang_token \
    --gen-subset test \
    -t $TARGET -s $SOURCE \
    --sacrebleu \
    --remove-bpe 'sentencepiece' \
    --max-len-b $MAX_LEN \
    --beam 5 \
    --batch-size 64 \
    --langs $langs > $FILE_PREF

cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp

python match_files.py --file1 $GOUND_TRUTH_PATH --file2 $FILE_PREF.hyp
}
fine_tune
#train
generate