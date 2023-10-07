#!/bin/bash

gpu_id=$1
model_name=$2
model_path=$3

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo "CUDA_VISIBLE_DEVICES=${gpu_id} python infer_ESConv.py \
  --config_name ${model_name} \
  --inputter_name ${model_name} \
  --add_nlg_eval \
  --seed 0 \
  --load_checkpoint $model_path/pytorch_model.bin \
  --fp16 false \
  --max_input_length 160 \
  --max_decoder_input_length 40 \
  --max_length 40 \
  --min_length 10 \
  --infer_batch_size 16 \
  --infer_input_file ./DATA/test.txt \
  --temperature 0.7 \
  --top_k 0 \
  --top_p 0.9 \
  --num_beams 1 \
  --repetition_penalty 1 \
  --no_repeat_ngram_size 0"

python infer_ESConv.py \
  --config_name $model_name \
  --inputter_name $model_name \
  --add_nlg_eval \
  --seed 0 \
  --load_checkpoint $model_path/pytorch_model.bin \
  --fp16 false \
  --max_input_length 160 \
  --max_decoder_input_length 40 \
  --max_length 40 \
  --min_length 10 \
  --infer_batch_size 16 \
  --infer_input_file ./DATA/test.txt \
  --temperature 0.7 \
  --top_k 0 \
  --top_p 0.9 \
  --num_beams 1 \
  --repetition_penalty 1 \
  --no_repeat_ngram_size 0
