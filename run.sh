#!/usr/bin/env bash
# Easy entrypoints for the Remake harness. Run INSIDE the ROCm container
# (./run_docker_torch.sh first), or anywhere the deps are installed.
#
#   ./run.sh setup                 install python deps into ./.pydeps (once)
#   ./run.sh data                  (re)build the corrected dataset from the NPZ
#   ./run.sh features [groups]     build feature cache (default: stats,hist)
#   ./run.sh list                  list all registered models
#   ./run.sh train <config> [...]  train a model (passes extra args to CLI)
#   ./run.sh eval  <run_dir> [split]
#   ./run.sh stack --runs A B C --label-space coarse11 --meta logreg --name S
#   ./run.sh board [port]          launch TensorBoard on runs/
#   ./run.sh zoo-smoke             quick tiny-subset sweep of every model
#   ./run.sh zoo-coarse            full coarse-11 sweep of every model
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/.pydeps:$REPO_DIR:${PYTHONPATH:-}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
PY=python3
CLI="$PY -m remake.cli"

cmd="${1:-help}"; shift || true

case "$cmd" in
  setup)
    mkdir -p .pydeps
    $PY -m pip install --target=.pydeps -r requirements.txt
    echo "deps installed into .pydeps" ;;

  data)
    $PY scripts/relabel_from_npz.py "$@" ;;

  features)
    groups="${1:-stats,hist}"
    $CLI features --split all --groups "$groups" ;;

  list)
    $CLI list ;;

  train)
    cfg="$1"; shift || true
    $CLI train --config "$cfg" "$@" ;;

  eval)
    $CLI eval --run "$1" --split "${2:-test}" ;;

  stack)
    $CLI stack "$@" ;;

  board)
    $PY -m tensorboard.main --logdir runs --port "${1:-6006}" --bind_all ;;

  zoo-smoke)
    # tiny subset, few epochs — proves every model runs end-to-end fast
    S='--set name=SMOKE data.max_per_class=1500 data.val_max_per_class=1000 train.epochs=2'
    for c in configs/lgbm_coarse.yaml configs/xgb_coarse.yaml configs/rf_coarse.yaml \
             configs/feature_mlp_coarse.yaml configs/cnn_bigru_coarse.yaml \
             configs/tcn_coarse.yaml configs/transformer_coarse.yaml \
             configs/mamba_coarse.yaml configs/gnn_coarse.yaml; do
      echo "=== smoke: $c ==="; $CLI train --config "$c" $S || echo "FAILED: $c"
    done ;;

  zoo-coarse)
    for c in configs/lgbm_coarse.yaml configs/xgb_coarse.yaml configs/rf_coarse.yaml \
             configs/feature_mlp_coarse.yaml configs/cnn_bigru_coarse.yaml \
             configs/tcn_coarse.yaml configs/transformer_coarse.yaml \
             configs/mamba_coarse.yaml configs/gnn_coarse.yaml; do
      echo "=== $c ==="; $CLI train --config "$c" || echo "FAILED: $c"
    done ;;

  help|*)
    sed -n '2,20p' "$0" ;;
esac
