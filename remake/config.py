"""Experiment configuration: a YAML file fully describes one run.

A config selects a model (by registry key), a label space, a feature set (for
feature-based models), and training/data knobs. Everything that affects a run is
captured here and copied verbatim into the run log for reproducibility.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataCfg:
    binary_dir: str = "data/4k_1/binary"
    sector_size: int = 4096
    seed: int = 42
    max_per_class: Optional[int] = None       # train subsample cap (None = all)
    val_max_per_class: Optional[int] = None    # eval subsample cap (None = all)
    features_dir: str = "data/4k_1/features"   # feature cache location


@dataclass
class TrainCfg:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1.0e-3
    min_lr: float = 1.0e-5
    weight_decay: float = 0.05
    warmup_pct: float = 0.05
    label_smoothing: float = 0.0
    confusion_lambda: float = 0.0               # >0 penalises predicting confusion_target when true class isn't it
    confusion_target: str = "archive"           # the "sink" class everything bleeds into
    grad_clip: float = 1.0
    grad_accum: int = 1
    amp: bool = True
    compile: bool = False
    num_workers: int = 8
    augment: bool = False
    gbflip_sigma: float = 0.0
    gbflip_max_rate: float = 0.05
    ema_decay: float = 0.0                      # 0 disables EMA
    early_stop_patience: int = 0                # 0 disables
    eval_every: int = 1
    seed: int = 42


@dataclass
class Config:
    name: str
    model: str                                  # registry key
    label_space: str = "coarse11"               # flat75 | coarse11 | specialist:<group>
    features: str = "stats_hist"                # feature set (feature/tree models)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    model_args: dict[str, Any] = field(default_factory=dict)
    log_root: str = "runs"
    notes: str = ""

    @staticmethod
    def from_dict(d: dict) -> "Config":
        d = copy.deepcopy(d)
        data = DataCfg(**(d.pop("data", {}) or {}))
        train = TrainCfg(**(d.pop("train", {}) or {}))
        return Config(data=data, train=train, **d)

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        with open(path) as fh:
            return Config.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> dict:
        return asdict(self)
