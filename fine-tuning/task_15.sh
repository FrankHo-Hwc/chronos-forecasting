#!/bin/sh
#data: only a name to name a file...
python ./performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data custom \
  --data_path weather.csv \
  --root_path  ./data/ \
  --checkpoints ./output/weather/run-2/checkpoint-final \
  --seq_len 512 \
  --label_len 0 \
  --token_len 96  \
  --pred_len 96 \
  --batch_size 128   \
  --which_gpu cuda:7  > ./logs/test_results/fine-tuning/weather_512_96.log



