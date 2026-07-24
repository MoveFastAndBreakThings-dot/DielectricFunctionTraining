#!/bin/bash
#SBATCH --job-name=cgcnn-mbj
#SBATCH --output=logs/cgcnn_mbj_%j.out
#SBATCH --error=logs/cgcnn_mbj_%j.err
#SBATCH --time=8:00:00
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
mkdir -p logs
export PYTHONUNBUFFERED=1

# Activate your env if the job doesn't inherit it:
source ~/ENV/bin/activate

python -u train_mace_frequency.py \
  --model cgcnn \
  --use_log \
  --lmdb_path latest_data_mbj/crystal_graphs.lmdb \
  --models_dir models_latest_mbj \
  --model_name cgcnn_kkr_latest_mbj \
  --batch_size 32
