#!/bin/sh
#data: only a name to name a file...
python ./performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data Solar \
  --data_path Solar_AL.txt \
  --root_path  ./data/ \
  --checkpoints ./checkpoints/chronos-t5-small \
  --seq_len 512 \
  --label_len 0 \
  --token_len 24  \
  --pred_len 24 \
  --batch_size 256



