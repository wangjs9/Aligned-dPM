#!/bin/bash

MODELS=$1
DATA=$2
FORMAT=$3
i=$4
seed=$5

# NOTE : Quote it else use array to avoid problems #
#FILES="output_10k/*Q_A_TARGET_rot*"

for f in $MODELS
do
  echo "python -m  rot_generation.generate_rots_decoder --model_directory ${f}/model --input ${DATA} --output ${f} --format_string ${FORMAT} --gpu $i --seed $seed --beams 3 --skip_special_tokens &"
  python -m  rot_generation.generate_rots_decoder --model_directory "${f}/model" --input "${DATA}" --output "${f}" --format_string "${FORMAT}" --gpu $i --seed $seed --beams 3 --skip_special_tokens &
  echo "python -m  rot_generation.generate_rots_decoder --model_directory ${f}/model --input ${DATA} --output ${f} --format_string ${FORMAT} --gpu $i --seed $seed --top_p 0.9 --skip_special_tokens &"
  python -m  rot_generation.generate_rots_decoder --model_directory "${f}/model" --input "${DATA}" --output "${f}" --format_string "${FORMAT}" --gpu $i --seed $seed --top_p 0.9 --skip_special_tokens &
  echo "python -m  rot_generation.generate_rots_decoder --model_directory ${f}/model --input ${DATA} --output ${f} --format_string ${FORMAT} --gpu $i --seed $seed --skip_special_tokens &"
  python -m  rot_generation.generate_rots_decoder --model_directory "${f}/model" --input "${DATA}" --output "${f}" --format_string "${FORMAT}" --gpu $i --seed $seed --skip_special_tokens &
  ((i+=1))
done