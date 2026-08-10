#!/bin/bash
#SBATCH --job-name=tsenn-kkr-opt
#SBATCH --output=logs/tsenn_kkr_opt_%j.out
#SBATCH --error=logs/tsenn_kkr_opt_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# Adjust for Vulcan as needed:
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH --account=YOUR_ACCOUNT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs models_latest_opt
export PYTHONUNBUFFERED=1

# Activate your env if the job doesn't inherit it:
source ~/ENV/bin/activate

# Fresh TSENN weights → ε_xx(iξ) on LMDB KKR grid (~500 pts), paper arch hypers.
python -u train_tsenn_kkr.py \
  --lmdb_path latest_data_opt/crystal_graphs.lmdb \
  --models_dir models_latest_opt \
  --model_name tsenn_kkr_xx_latest_opt \
  --use_log \
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-2 \
  --layers 4
