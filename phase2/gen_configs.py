#!/usr/bin/env python3
"""Generate per-group Phase-2 specialist configs.

Always writes a LightGBM specialist config for each of the 11 groups. Optionally
also writes transformer / tcn specialist configs that *warm-start from a trained
phase-1 NN run* — the byte encoder transfers (shapes match because we copy the
phase-1 model_args) and only the k-way head is reinitialised.

Usage
-----
  python phase2/gen_configs.py                                  # lgbm, all groups
  python phase2/gen_configs.py --transformer-from runs/transformer_large_XXXX
  python phase2/gen_configs.py --tcn-from runs/tcn_full_XXXX
  # (NN configs without a --*-from source are written to train fresh.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from remake import taxonomy  # noqa: E402

OUT = Path(__file__).resolve().parent / "configs"
HARD = {"raw", "archive"}   # near-identical leaves -> give the model more capacity


def lgbm_cfg(group: str) -> dict:
    hard = group in HARD
    return {
        "name": f"spec_{group}_lgbm",
        "model": "lgbm",
        "label_space": f"specialist:{group}",
        "features": "stats_hist",
        "data": {"max_per_class": None, "val_max_per_class": None},
        "model_args": {"n_estimators": 1000 if hard else 500,
                       "num_leaves": 255 if hard else 127, "learning_rate": 0.05},
        "notes": f"LightGBM specialist for '{group}' "
                 f"({len(taxonomy.GROUP_LEAVES[group])} leaves).",
    }


def nn_cfg(group: str, model: str, phase1_run: str | None) -> dict:
    # Copy phase-1 model_args so the encoder tensor shapes match -> warm-start
    # actually loads the backbone. init_from points at the phase-1 best.pt.
    margs, init = {}, None
    if phase1_run:
        p1 = yaml.safe_load((Path(phase1_run) / "config.yaml").read_text())
        if p1.get("model") != model:
            print(f"  WARNING: {phase1_run} is model='{p1.get('model')}', not "
                  f"'{model}' — warm-start won't transfer; training fresh.")
        else:
            margs = p1.get("model_args", {})
            init = str(Path(phase1_run) / "ckpt" / "best.pt")
    hard = group in HARD
    train = {"epochs": 20 if hard else 12, "batch_size": 192, "lr": 5.0e-4,
             "weight_decay": 0.05, "warmup_pct": 0.1, "augment": True,
             "gbflip_sigma": 0.01, "amp": True, "early_stop_patience": 4,
             "num_workers": 6}
    if init:
        train["init_from"] = init
    return {
        "name": f"spec_{group}_{model}",
        "model": model,
        "label_space": f"specialist:{group}",
        "data": {"max_per_class": 40000, "val_max_per_class": None},
        "train": train,
        "model_args": margs,
        "notes": f"{model} specialist for '{group}'"
                 + (f"; warm-started from {phase1_run}" if init else "; fresh"),
    }


def write(cfg: dict):
    OUT.mkdir(exist_ok=True)
    p = OUT / f"{cfg['name']}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer-from", default=None,
                    help="phase-1 transformer run dir to warm-start from")
    ap.add_argument("--tcn-from", default=None,
                    help="phase-1 tcn run dir to warm-start from")
    ap.add_argument("--transformer", action="store_true",
                    help="emit transformer specialist configs trained FRESH (no warm-start)")
    ap.add_argument("--tcn", action="store_true",
                    help="emit tcn specialist configs trained FRESH (no warm-start)")
    a = ap.parse_args()
    for g in taxonomy.GROUP_NAMES:
        write(lgbm_cfg(g))
        if a.transformer_from is not None or a.transformer:
            write(nn_cfg(g, "transformer", a.transformer_from))
        if a.tcn_from is not None or a.tcn:
            write(nn_cfg(g, "tcn", a.tcn_from))
    print(f"\nDone. Train with:  bash phase2/train_all.sh lgbm")


if __name__ == "__main__":
    main()
