#!/bin/bash
WANDB_KEY="" # if wandb is enabled, set WANDB_KEY to log in
base_dir="."
data_dir="${base_dir}/gec_private_train_data"
detect_vocab_path="./data/vocabulary/d_tags.txt"
correct_vocab_path="./data/vocabulary/labels.txt"
train_path="${data_dir}/train-stage1.edits"
valid_path="${data_dir}/test-stage1.edits"
config_path="configs/ds_config_basic.json"
timestamp=`date "+%Y%0m%0d_%T"`
save_dir="../ckpts/ckpt_$timestamp"
pretrained_transformer_path="xlm-roberta-base"
mkdir -p $save_dir
cp $0 $save_dir
cp $config_path $save_dir

    # --wandb \
    # --wandb_key $WANDB_KEY \

run_cmd="deepspeed --include localhost:0 --master_port 42997 train.py \
    --deepspeed \
    --config_path $config_path \
    --num_epochs 20 \
    --max_len 256 \
    --train_batch_size 32 \
    --accumulation_size 1 \
    --valid_batch_size 256 \
    --cold_step_count 0 \
    --lr 1e-5 \
    --cold_lr 1e-3 \
    --skip_correct 0 \
    --skip_complex 0 \
    --sub_token_mode average \
    --special_tokens_fix 1 \
    --unk2keep 0 \
    --tp_prob 1 \
    --tn_prob 1 \
    --detect_vocab_path $detect_vocab_path \
    --correct_vocab_path $correct_vocab_path \
    --do_eval \
    --train_path $train_path \
    --valid_path $valid_path \
    --use_cache 1 \
    --save_dir $save_dir \
    --pretrained_transformer_path $pretrained_transformer_path \
    --amp \
    2>&1 | tee ${save_dir}/train-${timestamp}.log"

echo ${run_cmd}
eval ${run_cmd}
