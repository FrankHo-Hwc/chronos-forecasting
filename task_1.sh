#!/bin/bash
#SBATCH --job-name=chronos-testing 
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=hewc0930@163.com
#SBATCH -t 5-00:00:00
#SBATCH -p critical
#SBATCH -A renkan-critical
#SBATCH --mem=200G 
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/test_results/task_%a.out
#SBATCH --error=./logs/test_results/task%a.err
#SBATCH --array=1,2

source  ~/.bashrc
pwd
nvidia -smi
conda activate fyc

sh fine-tuning/task_$SLURM_ARRAY_TASK_ID.sh