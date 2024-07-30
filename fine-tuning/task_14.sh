#!/bin/sh
#data: only a name to name a file...
python ./performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data custom \
  --data_path weather.csv \
  --root_path  ./data/ \
  --checkpoints ./output/weather/run-1/checkpoint-final \
  --seq_len 512 \
  --label_len 0 \
  --token_len 64  \
  --pred_len 64 \
  --batch_size 32  \
  --which_gpu cuda:7  > ./logs/test_results/fine-tuning/weather_512_64.log



