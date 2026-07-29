#!/bin/bash
#SBATCH --job-name=mace-opt
#SBATCH --output=logs/mace_opt_%j.out
#SBATCH --error=logs/mace_opt_%j.err
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

# Scratch MACE with MP-0 *small* architecture (128x0e, L0, 2 layers, r_max=6).
# No --foundation_model → random weights, not pretrained MP-0.
python -u train_mace_frequency.py \
  --model mace \
  --use_log \
  --lmdb_path latest_data_opt/crystal_graphs.lmdb \
  --models_dir models_latest_opt \
  --model_name mace_small_scratch_opt \
  --batch_size 32 \
  --mace_hidden_irreps "128x0e" \
  --mace_num_interactions 2 \
  --mace_max_ell 0 \
  --mace_r_max 6.0
