"""Data access for the corrected FFT-75 binary dataset.

Two consumers:
  - byte models  -> raw uint8 fragments as Long tensors (B, L)
  - feature/tree models -> dense float32 feature matrices (N, F)

Label spaces (see taxonomy.py): flat75 / coarse11 / specialist:<group>. For
specialist spaces only the samples belonging to that group are kept and labels
are remapped to a local 0..k-1 range.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import taxonomy


# ---------------------------------------------------------------------------
# Raw split access (memmap, zero-copy)
# ---------------------------------------------------------------------------

def load_meta(binary_dir: str | Path, split: str) -> dict:
    return json.loads((Path(binary_dir) / f"{split}_meta.json").read_text())


def open_split(binary_dir: str | Path, split: str):
    """Return (fragments_memmap (N,L) uint8, labels (N,) uint8 raw 0..74, meta)."""
    binary_dir = Path(binary_dir)
    meta = load_meta(binary_dir, split)
    n, L = meta["n_samples"], meta["sector_size"]
    frags = np.memmap(binary_dir / f"{split}_fragments.bin",
                      dtype=np.uint8, mode="r", shape=(n, L))
    labels = np.fromfile(binary_dir / f"{split}_labels.bin", dtype=np.uint8)
    assert labels.shape[0] == n, (labels.shape[0], n)
    return frags, labels, meta


# ---------------------------------------------------------------------------
# Label-space mapping + per-class subsampling
# ---------------------------------------------------------------------------

def map_labels(raw: np.ndarray, label_space: str) -> tuple[np.ndarray, np.ndarray]:
    """Map raw FiFTy labels (0..74) into a label space.

    Returns (keep_indices, mapped_labels) where mapped_labels aligns with
    keep_indices. For flat75/coarse11 keep_indices is all rows.
    """
    raw = raw.astype(np.int64)
    if label_space == "flat75":
        return np.arange(raw.shape[0]), raw.copy()
    if label_space == "coarse11":
        lut = np.asarray(taxonomy.LEAF_TO_GROUP, dtype=np.int64)
        return np.arange(raw.shape[0]), lut[raw]
    if label_space.startswith("specialist:"):
        group = label_space.split(":", 1)[1]
        gidx = taxonomy.GROUP_TO_IDX[group]
        leaf_to_group = np.asarray(taxonomy.LEAF_TO_GROUP, dtype=np.int64)
        keep = np.nonzero(leaf_to_group[raw] == gidx)[0]
        local = np.full(taxonomy.NUM_LEAVES, -1, dtype=np.int64)
        for leaf, loc in taxonomy.GROUP_LOCAL_IDX[group].items():
            local[leaf] = loc
        return keep, local[raw[keep]]
    raise ValueError(f"unknown label_space: {label_space}")


def subsample_per_class(labels: np.ndarray, max_per_class: Optional[int],
                        seed: int) -> np.ndarray:
    """Return indices (into `labels`) with at most max_per_class per label."""
    if max_per_class is None:
        return np.arange(labels.shape[0])
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.nonzero(labels == c)[0]
        if idx.shape[0] > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def split_indices(binary_dir, split, label_space, max_per_class, seed):
    """Full pipeline: load labels, map to label space, subsample per class.

    Returns (row_indices_into_split, mapped_labels) both aligned & sorted.
    """
    _, raw, _ = open_split(binary_dir, split)
    keep, mapped = map_labels(raw, label_space)
    sub = subsample_per_class(mapped, max_per_class, seed)
    return keep[sub], mapped[sub]


# ---------------------------------------------------------------------------
# GBFlip augmentation (storage bit-error model; train only)
# ---------------------------------------------------------------------------

def gbflip(block: np.ndarray, sigma: float, max_rate: float,
           rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0.0:
        return block
    rate = float(min(abs(rng.normal(0.0, sigma)), max_rate))
    L = block.size
    n = int(rate * L * 8)
    if n == 0:
        return block
    out = block.copy()
    bit = rng.integers(0, L * 8, size=n, dtype=np.int64)
    byte = (bit >> 3).astype(np.int64)
    msk = (np.uint8(1) << (bit & 7).astype(np.uint8))
    np.bitwise_xor.at(out, byte, msk)
    return out


# ---------------------------------------------------------------------------
# Torch datasets (imported lazily so non-torch tooling still imports this file)
# ---------------------------------------------------------------------------

def make_byte_dataset(binary_dir, split, label_space, *, indices, labels,
                      augment=False, gbflip_sigma=0.0, gbflip_max_rate=0.05,
                      seed=0):
    import torch
    from torch.utils.data import Dataset

    frags, _, meta = open_split(binary_dir, split)

    class ByteDataset(Dataset):
        def __init__(self):
            self.frags = frags
            self.indices = indices
            self.labels = labels
            self.augment = augment
            self.sigma = float(gbflip_sigma)
            self.max_rate = float(gbflip_max_rate)
            self._rng = None
            self._seed = seed

        def rng(self):
            if self._rng is None:
                info = torch.utils.data.get_worker_info()
                wid = info.id if info is not None else 0
                self._rng = np.random.default_rng(self._seed + 1009 * wid)
            return self._rng

        def __len__(self):
            return self.indices.shape[0]

        def __getitem__(self, i):
            row = int(self.indices[i])
            blk = np.asarray(self.frags[row], dtype=np.uint8)
            if self.augment and self.sigma > 0:
                blk = gbflip(np.ascontiguousarray(blk), self.sigma,
                             self.max_rate, self.rng())
            x = torch.from_numpy(blk.astype(np.int64))
            return x, int(self.labels[i])

    return ByteDataset()


def make_feature_dataset(X: np.ndarray, y: np.ndarray):
    import torch
    from torch.utils.data import TensorDataset
    return TensorDataset(torch.from_numpy(X.astype(np.float32)),
                         torch.from_numpy(y.astype(np.int64)))
