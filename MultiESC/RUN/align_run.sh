nohup python -m align_MultiESC --data_type=8 \
  --model_name_or_path=./final_output/lwg_whlookahead_generate \
  --learning_rate=5e-5 \
  --lr2=1e-4 \
  --num_train_epochs=10 \
  --with_cause \
  --with_strategy \
  --lookahead \
  --model_type=1 \
  --candidate_num=10 \
  --preference_model_dir=../preference_modeling/output/esc_d-PM_23-0527-2227_fold1 \
  --per_device_train_batch_size=7 \
  --seed=1 \
  --output_dir=./aligned_output/align_sample_10_d-PM > update_MultiESC.log 2>&1 &

