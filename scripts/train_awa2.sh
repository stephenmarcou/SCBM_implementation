#!/bin/bash

#SBATCH --job-name=SCBM_awa2
#SBATCH --output=/cluster/home/smarcou/work/logs_scbm/experiment_%j.out
#SBATCH --error=/cluster/home/smarcou/work/logs_scbm/experiment_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1

# AwA2 is dataloader-bound, not GPU-bound: unlike CUB_DatasetGenerator, AWA2_DatasetGenerator
# has no in-memory cache, so all ~30k train+val JPEGs are decoded from disk every epoch.
# The lever that buys the 4-hour budget is CPU workers, not the GPU.

source ~/.bashrc

conda deactivate
conda activate scbm

# Keep the per-worker BLAS threads from oversubscribing the 16 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /cluster/home/smarcou/SCBM_implementation

python -u train.py workers=16 "$@"
