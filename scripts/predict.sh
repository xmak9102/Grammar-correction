#!/bin/bash
base_dir="." #your workspace 'home/user/workspace'
mkdir result
deepspeed --include localhost:0 --master_port 42991 predict.py \
    --batch_size 256 \
    --iteration_count 5 \
    --min_len 3 \
    --max_len 128 \
    --min_error_probability 0.0 \
    --additional_confidence 0.0 \
    --sub_token_mode "average" \
    --max_pieces_per_token 5 \
    --model_dir "Grammar-correction" \
    --ckpt_id "epoch-5" \
    --deepspeed_config "./configs/ds_config_zero1.json" \
    --detect_vocab_path "./data/vocabulary/d_tags.txt" \
    --correct_vocab_path "./data/vocabulary/labels.txt" \
    --pretrained_transformer_path "roberta-base" \
    --input_path "${base_dir}/data/input.txt" \
    --out_path "result/output.txt" \
    --special_tokens_fix 1 \
    --detokenize 1 \
    --amp
