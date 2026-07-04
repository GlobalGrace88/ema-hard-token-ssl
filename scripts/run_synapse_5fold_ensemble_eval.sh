#!/usr/bin/env bash
# Official Synapse 5-fold ensemble eval (mean logits across fold checkpoints).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

METHOD="${1:-ours}"
FOLDS="${2:-0,1,2,3,4}"
OVERLAP="${3:-0.625}"

CKPT_DIR="outputs/downstream/synapse/${METHOD}"

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

CFG="configs/generated/synapse/eval/unetrpp_${METHOD}_fold0_official.yaml"
if [[ ! -f "$CFG" ]]; then
  python scripts/materialize_configs.py --task synapse --phase eval --method "$METHOD" --fold 0
fi

LIST=$(IFS=,; echo "${CKPTS[*]}")
OUT="outputs/eval/synapse/5fold_ensemble/${METHOD}/overlap_${OVERLAP}"

python scripts/eval.py \
  --config "$CFG" \
  --official_npz \
  --overlap "$OVERLAP" \
  --checkpoint_list "$LIST" \
  --output_dir "$OUT"
