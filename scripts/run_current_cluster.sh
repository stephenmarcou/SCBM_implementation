#!/bin/bash
#SBATCH --job-name=scbm_cub
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

set -euo pipefail

# -------------------------
# Paths
# -------------------------
# Override these when submitting if you keep the script outside the repo.
BASE_DIR="${BASE_DIR:-$PWD}"
LOG_DIR="${LOG_DIR:-$BASE_DIR/cluster_logs}"
DATA_DIR="${DATA_DIR:-$BASE_DIR/datasets}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$BASE_DIR/outputs}"

# -------------------------
# Runtime parameters
# -------------------------
SEED=${SEED:-0}
J_EPOCHS=${J_EPOCHS:-1}
TRAIN_ONLY=${TRAIN_ONLY:-True}
SAVE_MODEL=${SAVE_MODEL:-True}
HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-}

mkdir -p "$LOG_DIR" "$EXPERIMENT_DIR"

export WANDB_CACHE_DIR="$BASE_DIR/wandb/.cache/wandb"

# -------------------------
# Go to project directory
# -------------------------
cd "$BASE_DIR" || exit 1

# -------------------------
# Activate environment
# -------------------------
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate scbm

# -------------------------
# Debug info
# -------------------------
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "LOG_DIR: $LOG_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "EXPERIMENT_DIR: $EXPERIMENT_DIR"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# -------------------------
# Run training
# -------------------------
echo "Starting SCBM training..."

python -u train.py \
	+model=SCBM \
	+data=CUB \
	"data.data_path=$DATA_DIR" \
	"experiment_dir=$EXPERIMENT_DIR" \
	"seed=$SEED" \
	"model.j_epochs=$J_EPOCHS" \
	"save_model=$SAVE_MODEL" \
	"train_only=$TRAIN_ONLY" \
	$HYDRA_OVERRIDES