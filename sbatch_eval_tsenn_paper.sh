#!/bin/bash
#SBATCH --job-name=tsenn-paper
#SBATCH --output=logs/tsenn_paper_%j.out
#SBATCH --error=logs/tsenn_paper_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# Adjust for Vulcan as needed:
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH --account=YOUR_ACCOUNT

# OFFLINE ONLY — prepare on login node first:
#   ./setup_tsenn.sh --tar TSENN_models_*.tar.gz --dataset-tar TSENN_dataset_*.tar.gz
# Smoke subset: sbatch --export=ALL,TSENN_LIMIT=8 sbatch_eval_tsenn_paper.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tsenn_opt
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

source ~/ENV/bin/activate

CKPT=external/TSENN/model/revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch
DATA=external/TSENN/dataset/symmetrized_dataset_with_bandgap.pkl

for f in "$CKPT" "$DATA"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required local file: $f"
    echo "Prepare offline on login node (see header). Refusing network."
    exit 1
  fi
done

LIMIT_ARGS=()
if [[ -n "${TSENN_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${TSENN_LIMIT}")
  echo "Limited smoke: --limit ${TSENN_LIMIT}"
fi

python -u eval_tsenn_paper.py \
  --dataset "$DATA" \
  --checkpoint "$CKPT" \
  --batch_size "${TSENN_BATCH_SIZE:-8}" \
  --out_json tsenn_opt/paper_eval.json \
  "${LIMIT_ARGS[@]}"
