#!/usr/bin/env python3
"""
Upload the corrected FFT-75 binary dataset + feature cache to a HuggingFace
dataset repo so it can be pulled on vast.ai (or anywhere) in one command.

What gets uploaded
------------------
  data/4k_1/binary/   -- corrected labels (.bin), meta (.json)
                          fragment .bin files are resolved from their symlinks
                          and streamed directly (no copy needed locally)
  data/4k_1/features/ -- pre-built feature cache (.npy + .names.json)
                          saves the 14-min rebuild on every new instance

Usage (inside the ROCm container or wherever huggingface_hub is installed)
--------------------------------------------------------------------------
  python3 scripts/upload_dataset_hf.py --repo YOUR_HF_USERNAME/fft75-remake-data
  python3 scripts/upload_dataset_hf.py --repo YOUR_HF_USERNAME/fft75-remake-data --no-features
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="HuggingFace dataset repo, e.g. T0karev/fft75-remake-data")
    ap.add_argument("--no-features", action="store_true",
                    help="Skip the feature cache (~8 GB); useful for a quick first push")
    ap.add_argument("--token", default=None,
                    help="HF token (default: use cached login / HF_TOKEN env var)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError:
        sys.exit("huggingface_hub not installed — run: pip install huggingface_hub")

    api = HfApi(token=args.token)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    print(f"repo: https://huggingface.co/datasets/{args.repo}")

    repo = args.repo
    root = Path(__file__).resolve().parent.parent

    def upload(local: Path, remote: str):
        real = local.resolve()   # follow symlinks
        size_mb = real.stat().st_size / 1e6
        print(f"  uploading {remote}  ({size_mb:.0f} MB) ...", flush=True)
        api.upload_file(path_or_fileobj=str(real), path_in_repo=remote,
                        repo_id=repo, repo_type="dataset")
        print(f"  done: {remote}")

    # ---- binary dir (labels, meta, fragments) ----
    binary_dir = root / "data/4k_1/binary"
    for split in ("train", "val", "test"):
        for ext in ("_labels.bin", "_meta.json"):
            f = binary_dir / f"{split}{ext}"
            if f.exists():
                upload(f, f"binary/{split}{ext}")
        frag = binary_dir / f"{split}_fragments.bin"
        if frag.exists() or frag.is_symlink():
            upload(frag, f"binary/{split}_fragments.bin")

    # ---- feature cache (optional) ----
    if not args.no_features:
        feat_dir = root / "data/4k_1/features"
        for f in sorted(feat_dir.iterdir()):
            if f.suffix in (".npy", ".json"):
                upload(f, f"features/{f.name}")

    print(f"\nDone. Pull on vast.ai with:")
    print(f"  bash scripts/setup_vastai.sh --hf-user {repo.split('/')[0]}")
    print(f"  (which calls huggingface_hub.snapshot_download — no CLI needed)")


if __name__ == "__main__":
    main()
