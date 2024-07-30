#!/bin/sh
#data: only a name to name a file...accelerate launch 
accelerate launch  performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --root_path  ./data/ETT-small/ \
  --checkpoints ./output/ETTh1/24/run-0/checkpoint-final \
  --seq_len 512 \
  --token_len 24  \
  --label_len 48 \
  --pred_len 24 \
  --batch_size 32



