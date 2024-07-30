#!/bin/sh
#data: only a name to name a file...
python ./performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data Solar\
  --data_path Solar_AL.txt \
  --root_path  ./data/ \
  --checkpoints ./output/solar/run-1/checkpoint-final \
  --seq_len 512 \
  --label_len 0 \
  --token_len 64  \
  --pred_len 64 \
  --batch_size 32



