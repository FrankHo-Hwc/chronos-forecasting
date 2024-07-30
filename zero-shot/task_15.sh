#!/bin/sh
#data: only a name to name a file...
python ./performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data custom \
  --data_path weather.csv \
  --root_path  ./data/ \
  --checkpoints ./checkpoints/chronos-t5-small \
  --seq_len 512 \
  --label_len 0 \
  --token_len 96  \
  --pred_len 96 \
  --batch_size 256 \
  --which_gpu  cuda:2 > ./logs/test_results/zero-shot/weather_512_96.log



