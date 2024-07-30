#!/bin/bash
#SBATCH --job-name=chronos-finetuning 
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=hewc0930@163.com
#SBATCH -t 5-00:00:00
#SBATCH -p critical
#SBATCH -A renkan-critical
#SBATCH --mem=200G 
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/task_%a.out
#SBATCH --error=./logs/task%a.err
#SBATCH --array=14

source  ~/.bashrc
pwd
nvidia -smi
conda activate fyc

CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py --config  scripts/fine-tuning/task_$SLURM_ARRAY_TASK_ID.yaml

#torchrun --nproc-per-node=2 scripts/training/train.py --config  scripts/fine-tuning/task_$SLURM_ARRAY_TASK_ID.yaml