#!/bin/bash
#SBATCH --job-name=tsenn-trunc
#SBATCH --output=logs/tsenn_trunc_%j.out
#SBATCH --error=logs/tsenn_trunc_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Optional GPU unused (CPU KKR); keep for queue consistency:
# #SBATCH --gres=gpu:1

# OFFLINE: needs tsenn_opt/predictions.lmdb + latest_data_opt/crystal_graphs.lmdb

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tsenn_opt
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

source ~/ENV/bin/activate

PRED=tsenn_opt/predictions.lmdb
LMDB=latest_data_opt/crystal_graphs.lmdb
for f in "$PRED" "$LMDB"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f"
    exit 1
  fi
done

LIMIT_ARGS=()
if [[ -n "${TSENN_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${TSENN_LIMIT}")
fi

PAIR_ARGS=()
if [[ -n "${TSENN_PAIRS_DIR:-}" ]]; then
  PAIR_ARGS=(--pairs_dir "${TSENN_PAIRS_DIR}" --pair_split "${TSENN_PAIR_SPLIT:-test}")
fi

python -u eval_tsenn_opt_truncated.py \
  --pred_lmdb "$PRED" \
  --lmdb_path "$LMDB" \
  --omega_max "${TSENN_OMEGA_MAX:-30}" \
  --component "${TSENN_COMPONENT:-xx}" \
  --split "${TSENN_SPLIT:-all}" \
  --out_json tsenn_opt/eval_truncated.json \
  "${LIMIT_ARGS[@]}" \
  "${PAIR_ARGS[@]}"
