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

for weight in 1,5,10,50
do
  python -u train.py \
    +model=SCBM_RES \
    +data=CUB \
    model.use_L_int_loss=True \
    model.L_int_loss_weight=$weight \
    model.j_epochs=50 \
    data.num_residuals=50 \
    ratio_attributes_remove=0.75 \
    save_name=hyper_search_${weight}_Mateo_025 \
    data.pkl_file_dir=Mateo_025/ \
    train_only=True \
    save_model=True \
    incomplete=True \
    hyperparameter_search=True \
    "$@"
done