#!/bin/sh
#data: only a name to name a file...
accelerate launch  performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data PEMS03 \
  --data_path PEMS03.npz \
  --root_path  ./data/PEMS/ \
  --checkpoints ./checkpoints/chronos-t5-small \
  --seq_len 512 \
  --label_len 0 \
  --token_len 96  \
  --pred_len 96 \
  --batch_size 32  > ./logs/test_results/zero-shot/PEMS_512_96.log


