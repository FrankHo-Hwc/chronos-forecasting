#!/bin/sh
#data: only a name to name a file...
python ./performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --root_path  ./data/ETT-small/ \
  --checkpoints ./output/ETTh1/64/run-0/checkpoint-final \
  --seq_len 512 \
  --label_len 0 \
  --token_len 64  \
  --pred_len 64 \
  --batch_size 256



