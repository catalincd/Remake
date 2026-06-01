#!/usr/bin/env bash
# One-shot setup on a fresh vast.ai ROCm instance.
# Run once after spinning up the instance.
#
# Usage:
#   bash scripts/setup_vastai.sh --hf-user YOUR_HF_USERNAME
#   bash scripts/setup_vastai.sh --hf-user YOUR_HF_USERNAME --no-features
set -euo pipefail

HF_USER=""; NO_FEATURES=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-user) HF_USER="$2"; shift 2 ;;
    --no-features) NO_FEATURES="--no-features"; shift ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done
[[ -z "$HF_USER" ]] && { echo "Usage: $0 --hf-user YOUR_HF_USERNAME"; exit 1; }

echo "====== 1. deps ======"
pip install -q huggingface_hub huggingface-cli 2>/dev/null || true
pip install -q -r requirements.txt

echo "====== 2. dataset from HuggingFace ======"
mkdir -p data/4k_1
huggingface-cli download "$HF_USER/fft75-remake-data" \
    --repo-type dataset \
    --local-dir data/4k_1

# HF CLI flattens the directory; reorganise into binary/ and features/
mkdir -p data/4k_1/binary data/4k_1/features
for f in data/4k_1/binary/*; do true; done 2>/dev/null || true
# move if HF put them at the top level
for f in data/4k_1/*.bin data/4k_1/*.json; do
  [[ -f "$f" ]] && mv "$f" data/4k_1/binary/ || true
done
for f in data/4k_1/*.npy; do
  [[ -f "$f" ]] && mv "$f" data/4k_1/features/ || true
done

echo "====== 3. verify ======"
python3 - <<'PY'
import json, pathlib
for sp in ("train","val","test"):
    m=pathlib.Path(f"data/4k_1/binary/{sp}_meta.json")
    if m.exists():
        d=json.loads(m.read_text())
        print(f"[{sp}] n={d['n_samples']}  classes={d['n_classes']}")
PY

echo "====== done — run: ./run.sh list  then  ./run.sh train configs/<model>.yaml ======"
