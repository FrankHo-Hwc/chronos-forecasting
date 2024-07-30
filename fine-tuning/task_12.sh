#!/bin/sh
#data: only a name to name a file...
python performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data ETTm2 \
  --data_path ETTm2.csv \
  --root_path  ./data/ETT-small/ \
  --checkpoints ./output/ETTm2/96/run-0/checkpoint-final \
  --seq_len 512 \
  --label_len 0 \
  --token_len 96  \
  --pred_len 96 \
  --batch_size 64



