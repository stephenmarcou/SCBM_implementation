#!/bin/bash

#SBATCH --job-name=SCBM_HPARAM
#SBATCH --output="/cluster/home/smarcou/work/logs_scbm/hyperparameter_search_%j.out"
#SBATCH --error="/cluster/home/smarcou/work/logs_scbm/hyperparameter_search_%j.err"
#SBATCH --cpus-per-task=2
#SBATCH --time=0-03:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda deactivate
conda activate scbm
cd /cluster/home/smarcou/SCBM_implementation

for weight in 0.001 0.01 0.1 1.0 10.0
do
  python -u train.py model.use_L_int_loss=True model.L_int_loss_weight=$weight model.j_epochs=4  "$@"
done