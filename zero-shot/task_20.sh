#!/bin/sh
#data: only a name to name a file...
accelerate launch  performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data Solar \
  --data_path solar_AL.txt \
  --root_path  ./data/ \
  --checkpoints ./checkpoints/chronos-t5-small \
  --seq_len 512 \
  --label_len 0 \
  --token_len 64  \
  --pred_len 64 \
  --batch_size 256 > ./logs/test_results/zero-shot/solar_512_64.log




