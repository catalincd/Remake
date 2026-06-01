"""Run logging — "save everything, graph it later".

Each run gets its own directory under runs/:

    runs/<name>_<timestamp>/
        config.yaml            # exact config used
        training_log.json      # config + per-epoch records + best + final metrics
        events.out.tfevents..  # TensorBoard scalars (the live "sliders")
        ckpt/epoch_0001.pt ...  # a checkpoint every epoch ("every generation")
        ckpt/best.pt           # symlink to the best epoch
        confusion_<split>.json  # confusion matrix + class names
        <split>_probs.npy       # predicted probabilities (for stacking)
        <split>_true.npy        # ground-truth labels aligned with probs

training_log.json is rewritten after every epoch so a crash never loses history.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


class RunLogger:
    def __init__(self, config: dict, log_root: str = "runs",
                 use_tensorboard: bool = True):
        self.config = config
        name = config.get("name", "run")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(log_root) / f"{name}_{ts}"
        self.ckpt_dir = self.dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.epochs: list[dict] = []
        self.best: dict = {}

        (self.dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.dir))
            except Exception as e:  # tensorboard optional
                print(f"[logger] TensorBoard unavailable ({e}); JSON only")

        self._log = {
            "name": name,
            "started_at": datetime.now().isoformat(),
            "config": config,
            "status": "running",
            "best": {},
            "epochs": [],
        }
        self._flush()

    # -- scalars --------------------------------------------------------------
    def scalars(self, step: int, values: dict[str, float], prefix: str = ""):
        if self.tb is None:
            return
        for k, v in values.items():
            if v is None:
                continue
            self.tb.add_scalar(f"{prefix}{k}" if prefix else k, float(v), step)

    def hparams(self, hp: dict, metrics: dict):
        if self.tb is None:
            return
        try:
            flat = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                    for k, v in hp.items()}
            self.tb.add_hparams(flat, metrics)
        except Exception:
            pass

    # -- epochs ---------------------------------------------------------------
    def epoch(self, record: dict):
        self.epochs.append(record)
        self._log["epochs"] = self.epochs
        self.scalars(record["epoch"], {
            "train/loss": record.get("train_loss"),
            "train/acc": record.get("train_acc"),
            "val/loss": record.get("val_loss"),
            "val/acc": record.get("val_acc"),
            "lr": record.get("lr"),
            "time/epoch_s": record.get("elapsed_s"),
            "sys/gpu_mem_mb": record.get("gpu_mem_mb"),
            "train/grad_norm": record.get("grad_norm"),
        })
        self._flush()

    def set_best(self, best: dict):
        self.best = best
        self._log["best"] = best
        self._flush()

    # -- checkpoints ----------------------------------------------------------
    def save_checkpoint(self, state: dict, epoch: int, is_best: bool = False) -> Path:
        import torch
        path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(state, path)
        if is_best:
            link = self.ckpt_dir / "best.pt"
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(path.name, link)
        return path

    def save_blob(self, name: str, obj: Any):
        """Persist an arbitrary picklable object (e.g. a fitted tree model)."""
        import torch
        torch.save(obj, self.dir / name)

    # -- predictions & confusion ---------------------------------------------
    def save_predictions(self, split: str, probs: np.ndarray, y_true: np.ndarray):
        np.save(self.dir / f"{split}_probs.npy", probs.astype(np.float32))
        np.save(self.dir / f"{split}_true.npy", y_true.astype(np.int64))

    def save_summary(self, split: str, summary: dict):
        (self.dir / f"metrics_{split}.json").write_text(json.dumps(summary, indent=2))

    def save_text(self, name: str, text: str):
        (self.dir / name).write_text(text + "\n")

    # -- finalize -------------------------------------------------------------
    def finalize(self, status: str = "complete", extra: Optional[dict] = None):
        self._log["status"] = status
        self._log["finished_at"] = datetime.now().isoformat()
        self._log["total_time_s"] = time.time() - self.started
        if extra:
            self._log.update(extra)
        self._flush()
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        print(f"[logger] run dir: {self.dir}")

    def _flush(self):
        tmp = self.dir / "training_log.json.tmp"
        tmp.write_text(json.dumps(self._log, indent=1))
        tmp.replace(self.dir / "training_log.json")
