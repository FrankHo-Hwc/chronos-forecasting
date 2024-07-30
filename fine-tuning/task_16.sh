#!/bin/sh
#data: only a name to name a file...
accelerate launch  performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data custom \
  --data_path traffic.csv \
  --root_path  ./data/ \
  --checkpoints ./output/traffic/run-0/checkpoint-final \
  --seq_len 512 \
  --label_len 0 \
  --token_len 24  \
  --pred_len 24 \
  --batch_size 128 >  ./logs/test_results/fine-tuning/traffic_512_24.log




