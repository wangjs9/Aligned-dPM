#!/bin/bash
gpu_id=$1
model_path=$2

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo "CUDA_VISIBLE_DEVICES=${gpu_id} python align_ESConv.py \
  --config_name strat \
  --inputter_name strat \
  --eval_input_file DATA/valid.txt \
  --infer_input_file DATA/test.txt \
  --preference_model_dir ../preference_modeling/output/esc_d-PM_23-0527-2227_fold1 \
  --checkpoint_dir ${model_path} \
  --warmup_step 400 \
  --max_lr 3e-4"

python align_ESConv.py \
  --config_name strat \
  --inputter_name strat \
  --eval_input_file DATA/valid.txt \
  --infer_input_file DATA/test.txt \
  --preference_model_dir ../preference_modeling/output/esc_d-PM_23-0527-2227_fold1 \
  --checkpoint_dir $model_path \
  --warmup_step 400 \
  --max_lr 5e-4
