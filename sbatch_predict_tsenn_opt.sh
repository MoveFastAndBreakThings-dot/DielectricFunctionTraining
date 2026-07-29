#!/bin/bash
#SBATCH --job-name=tsenn-opt
#SBATCH --output=logs/tsenn_opt_%j.out
#SBATCH --error=logs/tsenn_opt_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# Adjust for Vulcan as needed:
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH --account=YOUR_ACCOUNT

# OFFLINE ONLY — do not call setup_tsenn.sh or build_jid_gap_table.py here.
# Prepare on login node / laptop first:
#   1) ./setup_tsenn.sh --tar TSENN_models_*.tar.gz
#   2) python build_jid_gap_table.py --from_manifest raw_opt/manifest.jsonl
#      (or --from_dft3d on a machine with network)
#   3) ensure latest_data_opt/crystal_graphs.lmdb exists
# Smoke:  sbatch --export=ALL,TSENN_LIMIT=5 sbatch_predict_tsenn_opt.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tsenn_opt
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

source ~/ENV/bin/activate

CKPT=external/TSENN/model/revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch
GAP=tsenn_opt/jid_gaps.jsonl
LMDB=latest_data_opt/crystal_graphs.lmdb

for f in "$CKPT" "$GAP" "$LMDB"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required local file: $f"
    echo "Prepare offline on login node (see header comments). Refusing to use network."
    exit 1
  fi
done

LIMIT_ARGS=()
if [[ -n "${TSENN_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${TSENN_LIMIT}")
  echo "Smoke / limited run: --limit ${TSENN_LIMIT}"
fi

python -u predict_tsenn_opt.py \
  --lmdb_path "$LMDB" \
  --gap_table "$GAP" \
  --out_lmdb tsenn_opt/predictions.lmdb \
  --checkpoint "$CKPT" \
  --batch_size "${TSENN_BATCH_SIZE:-8}" \
  "${LIMIT_ARGS[@]}"
