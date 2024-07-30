#!/bin/sh
#data: only a name to name a file...
python  performance_evaluation/forecast_toy.py \
  --model chronos_small \
  --data PEMS04 \
  --data_path PEMS04.npz \
  --root_path  ./data/PEMS/ \
  --checkpoints ./checkpoints/chronos-t5-small \
  --seq_len 512 \
  --label_len 0 \
  --token_len 24  \
  --pred_len 24 \
  --batch_size 128 > ./logs/test_results/zero-shot/PEMS04_512_24.log


# accelerate launch  performance_evaluation/forecast_toy.py \
#   --model chronos_small \
#   --data PEMS04 \
#   --data_path PEMS04.npz \
#   --root_path  ./data/PEMS/ \
#   --checkpoints ./checkpoints/chronos-t5-small \
#   --seq_len 512 \
#   --label_len 0 \
#   --token_len 64  \
#   --pred_len 64 \
#   --batch_size 128 > ./logs/test_results/zero-shot/PEMS04_512_64.log




# accelerate launch  performance_evaluation/forecast_toy.py \
#   --model chronos_small \
#   --data PEMS04 \
#   --data_path PEMS04.npz \
#   --root_path  ./data/PEMS/ \
#   --checkpoints ./checkpoints/chronos-t5-small \
#   --seq_len 512 \
#   --label_len 0 \
#   --token_len 96  \
#   --pred_len 96 \
#   --batch_size 128 > ./logs/test_results/zero-shot/PEMS04_512_96.log



