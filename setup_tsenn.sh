#!/bin/bash
# Clone TSENN sidecar (if needed) and extract the main paper checkpoint
# from a manually downloaded Figshare models tarball.
#
# LOGIN NODE / laptop only (git clone needs network if sidecar missing).
# Do NOT run under sbatch. Extract itself is offline once the tar is local.
#
# Usage:
#   ./setup_tsenn.sh
#   ./setup_tsenn.sh --tar /path/to/TSENN_models_20260128.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TSENN_DIR="$ROOT/external/TSENN"
CHECKPOINT_NAME='revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch'
TAR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tar) TAR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--tar PATH_TO_TSENN_models*.tar.gz]"
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

mkdir -p "$TSENN_DIR/model"
DEST="$TSENN_DIR/model/$CHECKPOINT_NAME"

if [[ -f "$DEST" ]]; then
  echo "Checkpoint already extracted: $DEST"
  exit 0
fi

if [[ -z "$TAR" ]]; then
  shopt -s nullglob
  cands=("$ROOT"/TSENN_models*.tar.gz "$ROOT"/external/TSENN_models*.tar.gz
         "$ROOT"/TSENN_models*.tgz "$ROOT"/external/TSENN_models*.tgz)
  shopt -u nullglob
  if ((${#cands[@]} > 0)); then
    TAR="${cands[0]}"
  fi
fi

if [[ -z "${TAR:-}" || ! -f "$TAR" ]]; then
  echo "ERROR: models tarball not found."
  echo "Place TSENN_models_20260128.tar.gz in the repo root or external/, or pass --tar PATH"
  exit 1
fi

echo "Looking for $CHECKPOINT_NAME in $TAR ..."
MEMBER="$(tar -tzf "$TAR" | grep -F "$CHECKPOINT_NAME" | grep -v '/$' | head -n 1 || true)"
if [[ -z "$MEMBER" ]]; then
  echo "ERROR: $CHECKPOINT_NAME not found inside $TAR"
  echo "Available *revision_250909* members:"
  tar -tzf "$TAR" | grep -F 'revision_250909' || true
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Extracting $MEMBER ..."
tar -xzf "$TAR" -C "$TMP" "$MEMBER"
FOUND="$(find "$TMP" -name "$CHECKPOINT_NAME" -type f | head -n 1)"
cp "$FOUND" "$DEST"
echo "Done: $DEST ($(du -h "$DEST" | cut -f1))"
