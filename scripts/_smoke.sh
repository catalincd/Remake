#!/usr/bin/env bash
# End-to-end smoke test INSIDE the ROCm container. Validates: deps, imports,
# registry, GPU, every NN model (1 epoch tiny), the feature cache, every tree
# model, the feature-MLP, and stacking. Tiny subsets — proves it RUNS.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/.pydeps:$PWD:${PYTHONPATH:-}"
CLI="python3 -m remake.cli"

echo "############ 1. deps ############"
python3 -m pip install -q --target=.pydeps -r requirements.txt 2>&1 | tail -3

echo "############ 2. torch/gpu ############"
python3 - <<'EOF'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available(): print("device:", torch.cuda.get_device_name(0))
EOF

echo "############ 3. registry ############"
$CLI list

echo "############ 4. NN smoke (tiny, 1 epoch each) ############"
NNSET='--set name=SMOKE data.max_per_class=200 data.val_max_per_class=200 train.epochs=1 train.batch_size=64 train.num_workers=2'
for c in cnn_bigru tcn transformer mamba gnn; do
  echo "---- $c ----"
  $CLI train --config configs/${c}_coarse.yaml $NNSET 2>&1 | tail -5 || echo "FAILED:$c"
done

echo "############ 5. feature cache (all splits, stats+hist) ############"
time $CLI features --split all --groups stats,hist 2>&1 | tail -4

echo "############ 6. tree smoke (tiny) ############"
TSET='--set name=SMOKE data.max_per_class=1000 data.val_max_per_class=2000'
for c in lgbm xgb rf; do
  echo "---- $c ----"
  $CLI train --config configs/${c}_coarse.yaml $TSET 2>&1 | tail -4 || echo "FAILED:$c"
done
echo "---- feature_mlp ----"
$CLI train --config configs/feature_mlp_coarse.yaml \
  --set name=SMOKE data.max_per_class=1000 data.val_max_per_class=2000 train.epochs=2 2>&1 | tail -5 || echo "FAILED:feature_mlp"

echo "############ 7. stacking smoke (two matched tree runs) ############"
A=$($CLI train --config configs/lgbm_coarse.yaml --set name=STK_A data.max_per_class=1000 data.val_max_per_class=2000 2>&1 | grep RUN_DIR | awk '{print $2}')
B=$($CLI train --config configs/rf_coarse.yaml   --set name=STK_B data.max_per_class=1000 data.val_max_per_class=2000 2>&1 | grep RUN_DIR | awk '{print $2}')
echo "bases: $A | $B"
$CLI stack --runs "$A" "$B" --label-space coarse11 --meta logreg --name STK 2>&1 | tail -3 || echo "FAILED:stack"

echo "############ SMOKE DONE ############"
