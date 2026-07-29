#!/bin/bash
# Clone TSENN sidecar + extract main checkpoint (and optionally paper dataset)
# from manually downloaded Figshare tarballs.
#
# LOGIN NODE / laptop only (git clone needs network if sidecar missing).
# Do NOT run under sbatch.
#
# Usage:
#   ./setup_tsenn.sh
#   ./setup_tsenn.sh --tar TSENN_models_20260128.tar.gz
#   ./setup_tsenn.sh --dataset-tar TSENN_dataset_20260128.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TSENN_DIR="$ROOT/external/TSENN"
CHECKPOINT_NAME='revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch'
DATASET_NAME='symmetrized_dataset_with_bandgap.pkl'
TAR=""
DATASET_TAR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tar) TAR="$2"; shift 2 ;;
    --dataset-tar) DATASET_TAR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--tar MODELS.tar.gz] [--dataset-tar DATASET.tar.gz]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$ROOT/external"

if [[ ! -d "$TSENN_DIR/.git" ]]; then
  echo "Cloning qmatyanlab/TSENN → $TSENN_DIR"
  git clone --depth 1 https://github.com/qmatyanlab/TSENN.git "$TSENN_DIR"
else
  echo "TSENN already present at $TSENN_DIR"
fi

mkdir -p "$TSENN_DIR/model" "$TSENN_DIR/dataset"

extract_member() {
  local archive="$1"
  local name="$2"
  local dest_dir="$3"
  local dest="$dest_dir/$name"
  if [[ -f "$dest" ]]; then
    echo "Already present: $dest"
    return 0
  fi
  echo "Looking for $name in $archive ..."
  local member
  member="$(tar -tzf "$archive" | grep -F "$name" | grep -v '/$' | head -n 1 || true)"
  if [[ -z "$member" ]]; then
    echo "ERROR: $name not found inside $archive"
    return 1
  fi
  local tmp
  tmp="$(mktemp -d)"
  tar -xzf "$archive" -C "$tmp" "$member"
  local found
  found="$(find "$tmp" -name "$name" -type f | head -n 1)"
  cp "$found" "$dest"
  rm -rf "$tmp"
  echo "Done: $dest ($(du -h "$dest" | cut -f1))"
}

# --- models tar ---
if [[ -z "$TAR" ]]; then
  shopt -s nullglob
  cands=("$ROOT"/TSENN_models*.tar.gz "$ROOT"/external/TSENN_models*.tar.gz
         "$ROOT"/TSENN_models*.tgz "$ROOT"/external/TSENN_models*.tgz)
  shopt -u nullglob
  if ((${#cands[@]} > 0)); then TAR="${cands[0]}"; fi
fi

DEST_CKPT="$TSENN_DIR/model/$CHECKPOINT_NAME"
if [[ -f "$DEST_CKPT" ]]; then
  echo "Checkpoint already extracted: $DEST_CKPT"
elif [[ -n "${TAR:-}" && -f "$TAR" ]]; then
  extract_member "$TAR" "$CHECKPOINT_NAME" "$TSENN_DIR/model"
else
  echo "WARN: models tarball not found; checkpoint not extracted."
  echo "  Place TSENN_models_*.tar.gz in repo root/external or pass --tar PATH"
fi

# --- dataset tar ---
if [[ -z "$DATASET_TAR" ]]; then
  shopt -s nullglob
  dcands=("$ROOT"/TSENN_dataset*.tar.gz "$ROOT"/external/TSENN_dataset*.tar.gz
          "$ROOT"/TSENN_dataset*.tgz "$ROOT"/external/TSENN_dataset*.tgz)
  shopt -u nullglob
  if ((${#dcands[@]} > 0)); then DATASET_TAR="${dcands[0]}"; fi
fi

DEST_DATA="$TSENN_DIR/dataset/$DATASET_NAME"
if [[ -f "$DEST_DATA" ]]; then
  echo "Dataset already extracted: $DEST_DATA"
elif [[ -n "${DATASET_TAR:-}" && -f "$DATASET_TAR" ]]; then
  extract_member "$DATASET_TAR" "$DATASET_NAME" "$TSENN_DIR/dataset"
else
  echo "WARN: dataset tarball not found; paper eval needs $DATASET_NAME"
  echo "  Place TSENN_dataset_*.tar.gz in repo root/external or pass --dataset-tar PATH"
fi

if [[ ! -f "$DEST_CKPT" ]]; then
  echo "ERROR: missing checkpoint $DEST_CKPT"
  exit 1
fi
echo "Setup OK (checkpoint ready$([ -f "$DEST_DATA" ] && echo ', dataset ready'))."
