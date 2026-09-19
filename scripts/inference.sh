#!/bin/bash

#SBATCH --job-name=SCBM_inf
#SBATCH --output="/cluster/home/smarcou/work/logs_scbm/inference_%j.out"
#SBATCH --error="/cluster/home/smarcou/work/logs_scbm/inference_%j.err"
#SBATCH --cpus-per-task=2
#SBATCH --time=0-02:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Post-hoc inference / interventions on a saved run. Same calling convention as train.sh:
#   sbatch scripts/inference.sh +model=SCBM_RES +data=CUB inference.ex_name=<run> \
#       incomplete=True run_inference=True run_interventions=True \
#       inference.noise.type=salt_pepper inference.noise.amount=0.05

source ~/.bashrc
conda deactivate
conda activate scbm
cd /cluster/home/smarcou/SCBM_implementation

python -u inference.py "$@"
