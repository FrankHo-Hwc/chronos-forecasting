#!/bin/sh
#data: only a name to name a file...
python scripts/test/forecast_toy_lines.py \
  --model chronos_small \
  --data ETTh1 \
  --data_path ETT-small/ETTh1.csv \
  --root_path /public/home/renkan/hwc/exp/chronos-forecasting/data/ \
  --checkpoints ./output/finetune/ETTh1/run-37/checkpoint-final \
  --seq_len 512 \
  --label_len 488 \
  --token_len 24  \
  --pred_len 24 \
  --batch_size 512



