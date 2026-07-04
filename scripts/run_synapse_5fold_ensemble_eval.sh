#!/usr/bin/env bash
# Official Synapse 5-fold ensemble eval (mean logits across fold checkpoints).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

METHOD="${1:-ours}"
RECIPE="${2:-v4}"
FOLDS="${3:-0,1,2,3,4}"
OVERLAP="${4:-0.625}"

if [[ "$METHOD" == "scratch" ]]; then
  CKPT_DIR="outputs/downstream/synapse/scratch"
  CFG_METHOD="scratch"
elif [[ "$RECIPE" == "v5" ]]; then
  CKPT_DIR="outputs/downstream/synapse/ours_v5"
  CFG_METHOD="ours"
else
  CKPT_DIR="outputs/downstream/synapse/ours"
  CFG_METHOD="ours"
fi

IFS=',' read -ra FOLD_ARR <<< "$FOLDS"
CKPTS=()
for f in "${FOLD_ARR[@]}"; do
  p="${CKPT_DIR}/fold_${f}/best_model.pt"
  if [[ ! -f "$p" ]]; then
    echo "Missing checkpoint: $p" >&2
    exit 1
  fi
  CKPTS+=("$p")
done

CFG="configs/generated/synapse/eval/unetrpp_${CFG_METHOD}_fold0_official.yaml"
if [[ ! -f "$CFG" ]]; then
  python scripts/materialize_configs.py --task synapse --phase eval --method "$CFG_METHOD" --fold 0 --recipe "$RECIPE"
fi

LIST=$(IFS=,; echo "${CKPTS[*]}")
OUT="outputs/eval/synapse/5fold_ensemble/${METHOD}_${RECIPE}/overlap_${OVERLAP}"

python scripts/eval.py \
  --config "$CFG" \
  --official_npz \
  --overlap "$OVERLAP" \
  --checkpoint_list "$LIST" \
  --output_dir "$OUT"
