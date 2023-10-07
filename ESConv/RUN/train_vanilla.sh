#!/bin/bash
gpu_id=$1
export CUDA_VISIBLE_DEVICES=${gpu_id}
echo "CUDA_VISIBLE_DEVICES=${gpu_id} python train_ESConv.py \
  --config_name vanilla \
  --inputter_name vanilla \
  --eval_input_file ./DATA/valid.txt \
  --seed 13 \
  --max_input_length 160 \
  --max_decoder_input_length 40 \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_num 2 \
  --warmup_steps 100 \
  --fp16 false \
  --loss_scale 0.0 \
  --pbar true"
python train_ESConv.py \
  --config_name vanilla \
  --inputter_name vanilla \
  --eval_input_file ./DATA/valid.txt \
  --seed 13 \
  --max_input_length 160 \
  --max_decoder_input_length 40 \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_num 2 \
  --warmup_steps 100 \
  --fp16 false \
  --loss_scale 0.0 \
  --pbar true