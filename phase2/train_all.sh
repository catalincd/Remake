#!/usr/bin/env bash
# Train Phase-2 specialists for one model family across groups, then summarise.
#
# Usage:
#   bash phase2/train_all.sh lgbm                 # all 11 groups, LightGBM
#   bash phase2/train_all.sh transformer raw archive   # only these groups
#   bash phase2/train_all.sh tcn
#
# Configs must exist (run phase2/gen_configs.py first; pass --transformer-from /
# --tcn-from there to enable warm-start for the NN families).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/.pydeps:$PWD:${PYTHONPATH:-}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"

MODEL="${1:-lgbm}"; shift || true
# NB: do NOT name this GROUPS — that's a reserved bash array (caller's group IDs).
groups=("$@")
if [ ${#groups[@]} -eq 0 ]; then
  groups=(raw bitmap vector video archive executable office published text audio other)
fi

for g in "${groups[@]}"; do
  cfg="phase2/configs/spec_${g}_${MODEL}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "skip $g: $cfg not found (run: python3 phase2/gen_configs.py)"; continue
  fi
  echo "================ specialist: $g ($MODEL) ================"
  python3 -m remake.cli train --config "$cfg" || echo "FAILED: $g ($MODEL)"
done

echo
python3 phase2/summarize.py --model "$MODEL"
