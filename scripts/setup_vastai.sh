#!/usr/bin/env bash
# One-shot setup on a fresh vast.ai instance.
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

# Everything via `python3 -m pip` and the huggingface_hub *Python API* — no
# dependency on a CLI binary being on PATH (the CLI was renamed huggingface-cli
# -> hf and isn't reliably on root's PATH).
PY=python3

echo "====== 1. deps ======"
$PY -m pip install -q --upgrade huggingface_hub
$PY -m pip install -q -r requirements.txt

echo "====== 2. dataset from HuggingFace (~38 GB, be patient) ======"
# snapshot_download preserves the repo layout (binary/ and features/ subdirs).
HF_REPO="$HF_USER/fft75-remake-data" $PY - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["HF_REPO"], repo_type="dataset",
    local_dir="data/4k_1", local_dir_use_symlinks=False,
    max_workers=8,
)
print("download complete")
PY

echo "====== 3. verify ======"
$PY - <<'PY'
import json, pathlib, sys
ok=True
for sp in ("train","val","test"):
    m=pathlib.Path(f"data/4k_1/binary/{sp}_meta.json")
    if m.exists():
        d=json.loads(m.read_text())
        print(f"  [{sp}] n={d['n_samples']}  classes={d['n_classes']}")
    else:
        print(f"  [{sp}] MISSING — download may have failed"); ok=False
sys.exit(0 if ok else 1)
PY

echo "====== done ======"
echo "  ./run.sh list"
echo "  ./run.sh train configs/tcn_full.yaml"
