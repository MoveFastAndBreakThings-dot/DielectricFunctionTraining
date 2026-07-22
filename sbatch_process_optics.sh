#!/bin/bash
#SBATCH --job-name=optics-lmdb
#SBATCH --output=logs/optics_lmdb_%j.out
#SBATCH --error=logs/optics_lmdb_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
# Adjust partition/account for Vulcan as needed:
# #SBATCH --partition=YOUR_PARTITION

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Activate your env if needed:
# source ~/ENV/bin/activate

SOURCE="${SOURCE:-opt}"          # opt | mbj
RAW_DIR="${RAW_DIR:-raw_${SOURCE}}"
OUT_DIR="${OUT_DIR:-latest_data_${SOURCE}}"
WORKERS="${SLURM_CPUS_PER_TASK:-32}"

echo "source=$SOURCE raw=$RAW_DIR out=$OUT_DIR workers=$WORKERS"
python process_optics_lmdb.py \
  --source "$SOURCE" \
  --raw_dir "$RAW_DIR" \
  --out_dir "$OUT_DIR" \
  --workers "$WORKERS"
