#!/bin/bash
# Offline readiness check before sbatch (no network).
set -euo pipefail
cd "$(dirname "$0")"

CKPT=external/TSENN/model/revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch
DATA=external/TSENN/dataset/symmetrized_dataset_with_bandgap.pkl
GAP=tsenn_opt/jid_gaps.jsonl
LMDB=latest_data_opt/crystal_graphs.lmdb
ok=1

check() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    echo "OK  [$label] $path ($(du -h "$path" | cut -f1))"
  else
    echo "MISS [$label] $path"
    ok=0
  fi
}

echo "=== Paper sanity-check prerequisites ==="
check ckpt "$CKPT"
check dataset "$DATA"

echo
echo "=== OPT semiconductor run prerequisites ==="
check gaps "$GAP"
check opt_lmdb "$LMDB"

if [[ -d external/TSENN/utils ]]; then
  echo "OK  [code] external/TSENN/utils"
else
  echo "MISS [code] external/TSENN (run ./setup_tsenn.sh on login node)"
  ok=0
fi

if [[ "$ok" -ne 1 ]]; then
  echo
  echo "Login-node prep (not sbatch):"
  echo "  ./setup_tsenn.sh --tar TSENN_models_*.tar.gz --dataset-tar TSENN_dataset_*.tar.gz"
  echo "  python build_jid_gap_table.py --from_manifest raw_opt/manifest.jsonl"
  exit 1
fi

echo
echo "Ready:"
echo "  sbatch --export=ALL,TSENN_LIMIT=8 sbatch_eval_tsenn_paper.sh   # paper MAE smoke"
echo "  sbatch sbatch_eval_tsenn_paper.sh                               # full paper test MAE (~0.127)"
echo "  sbatch --export=ALL,TSENN_LIMIT=5 sbatch_predict_tsenn_opt.sh   # OPT smoke"
echo "  sbatch sbatch_predict_tsenn_opt.sh                               # full OPT semis"
