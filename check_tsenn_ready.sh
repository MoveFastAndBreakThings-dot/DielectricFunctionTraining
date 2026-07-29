#!/bin/bash
# Offline readiness check before sbatch (no network).
set -euo pipefail
cd "$(dirname "$0")"

CKPT=external/TSENN/model/revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch
GAP=tsenn_opt/jid_gaps.jsonl
LMDB=latest_data_opt/crystal_graphs.lmdb
ok=1

check() {
  if [[ -f "$1" ]]; then
    echo "OK  $1 ($(du -h "$1" | cut -f1))"
  else
    echo "MISS $1"
    ok=0
  fi
}

check "$CKPT"
check "$GAP"
check "$LMDB"

if [[ -d external/TSENN/utils ]]; then
  echo "OK  external/TSENN/utils"
else
  echo "MISS external/TSENN (run ./setup_tsenn.sh on login node)"
  ok=0
fi

if [[ "$ok" -ne 1 ]]; then
  echo
  echo "Login-node prep (not sbatch):"
  echo "  ./setup_tsenn.sh --tar TSENN_models_*.tar.gz"
  echo "  python build_jid_gap_table.py --from_manifest raw_opt/manifest.jsonl"
  exit 1
fi

echo
echo "Ready for offline sbatch:"
echo "  sbatch --export=ALL,TSENN_LIMIT=5 sbatch_predict_tsenn_opt.sh   # smoke"
echo "  sbatch sbatch_predict_tsenn_opt.sh                               # full"
