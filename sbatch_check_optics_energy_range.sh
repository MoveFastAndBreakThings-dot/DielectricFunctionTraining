#!/bin/bash
#SBATCH --job-name=check-opt-en
#SBATCH --output=logs/check_opt_en_%j.out
#SBATCH --error=logs/check_opt_en_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Adjust for Vulcan as needed:
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH --account=YOUR_ACCOUNT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export PYTHONUNBUFFERED=1

source ~/ENV/bin/activate

python -u check_optics_energy_range.py \
  --lmdb_path latest_data_opt/crystal_graphs.lmdb \
  --threshold_ev 100
