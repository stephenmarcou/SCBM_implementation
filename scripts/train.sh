#!/bin/bash

#SBATCH --job-name=SCBM
#SBATCH --output="/cluster/home/smarcou/work/logs_scbm/experiment_%j.out" 
#SBATCH --error="/cluster/home/smarcou/work/logs_scbm/experiment_%j.err"
#SBATCH --cpus-per-task=2
#SBATCH --time=0-00:05:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda deactivate
conda activate scbm
cd /cluster/home/smarcou/SCBM_implementation

python -u train.py "$@"