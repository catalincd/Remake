"""Engineered per-fragment features + on-disk cache.

These are the orthogonal, position-free statistics that gradient-boosted trees
and the feature-MLP feed on, and that separate the clusters a byte-CNN struggles
with (entropy/compression for archive, structural-char densities for text).

Feature groups (selectable, '_'-joined in configs, e.g. "stats_hist"):
  stats : ~60 scalar features (entropy, zlib ratio, byte-class ratios,
          structural char + bigram densities, autocorrelation, run stats)
  hist  : 256-bin normalised byte histogram (compression-algorithm fingerprint)
  ncd   : multi-compressor ratios (zlib/bz2/lzma) + deltas — SLOW (lzma), opt-in

Caching: `build_cache(split, groups)` streams the split in chunks and writes one
float32 memmap per group to <features_dir>/<split>.<group>.npy plus a names json.
"""
from __future__ import annotations

import bz2
import json
import lzma
import zlib
from pathlib import Path

import numpy as np

from . import data as _data

# Structural single chars (text-format fingerprints). Densities = hist[c]/L.
_STRUCT_CHARS = [ord(c) for c in "{}[]:<>/=&\"',;|#@\\()!"] + [0x09, 0x0A, 0x0D, 0x20]
_STRUCT_CHAR_NAMES = [f"chr_{c:02x}" for c in _STRUCT_CHARS]

# Structural byte bigrams (e.g. "</", '="', ',\n') — pair occurrence rates.
_STRUCT_BIGRAMS = [
    (ord('"'), ord(':')), (ord('}'), ord(',')), (ord(']'), ord(',')),
    (ord('<'), ord('/')), (ord('='), ord('"')), (ord('/'), ord('>')),
    (ord('<'), ord('!')), (ord(','), 0x0A), (0x0A, ord('"')), (ord(':'), ord(' ')),
]
_STRUCT_BIGRAM_NAMES = [f"bg_{a:02x}{b:02x}" for a, b in _STRUCT_BIGRAMS]


def _batch_hist(block: np.ndarray) -> np.ndarray:
    """(B,L) uint8 -> (B,256) float32 normalised histogram."""
    B, L = block.shape
    flat = block.astype(np.int64) + 256 * np.arange(B, dtype=np.int64)[:, None]
    bc = np.bincount(flat.ravel(), minlength=B * 256).reshape(B, 256)
    return bc.astype(np.float32) / L


def hist_features(block: np.ndarray) -> tuple[np.ndarray, list[str]]:
    h = _batch_hist(block)
    return h, [f"hist_{i:02x}" for i in range(256)]


def stats_features(block: np.ndarray) -> tuple[np.ndarray, list[str]]:
    B, L = block.shape
    bf = block.astype(np.float32)
    h = _batch_hist(block)                                   # (B,256)

    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(h > 0, h * np.log2(h), 0.0).sum(1) / 8.0   # 0..1

    # byte-class ratios (vectorised counts via the histogram)
    def frac(lo, hi):  # inclusive byte-value range fraction
        return h[:, lo:hi + 1].sum(1)
    printable = frac(32, 126)
    nul = h[:, 0]
    high = h[:, 128:256].sum(1)
    ws = h[:, 9] + h[:, 10] + h[:, 13] + h[:, 32]
    alpha = frac(65, 90) + frac(97, 122)
    digit = frac(48, 57)
    ctrl = frac(1, 8) + frac(11, 12) + frac(14, 31) + h[:, 127]

    mean = bf.mean(1) / 255.0
    std = bf.std(1) / 255.0
    nonzero_bins = (h > 0).sum(1).astype(np.float32) / 256.0
    peak = h.max(1)

    # sequence stats
    d = np.abs(np.diff(bf, axis=1))
    mean_delta = d.mean(1) / 255.0
    a = bf[:, :-1] - bf[:, :-1].mean(1, keepdims=True)
    b = bf[:, 1:] - bf[:, 1:].mean(1, keepdims=True)
    denom = np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-6
    autocorr1 = (a * b).sum(1) / denom
    zero_runs = (block == 0)
    # crude longest-null-run proxy: max consecutive nulls via run-length on a sample axis
    null_run = np.zeros(B, dtype=np.float32)
    # cheap: average run via counting transitions
    trans = np.abs(np.diff(zero_runs.astype(np.int8), axis=1)).sum(1) + 1
    null_run = (zero_runs.sum(1) / np.maximum(trans, 1)).astype(np.float32) / L

    # structural char densities (from hist)
    struct = h[:, _STRUCT_CHARS]                              # (B, n_chars)

    # structural bigram densities
    bg = np.empty((B, len(_STRUCT_BIGRAMS)), dtype=np.float32)
    a0 = block[:, :-1]
    b0 = block[:, 1:]
    for j, (ca, cb) in enumerate(_STRUCT_BIGRAMS):
        bg[:, j] = ((a0 == ca) & (b0 == cb)).mean(1)

    # zlib compression ratio (Python loop; strong low/high-entropy separator)
    zr = np.empty(B, dtype=np.float32)
    for i in range(B):
        zr[i] = len(zlib.compress(block[i].tobytes(), 6)) / L

    scalars = np.stack([
        ent, zr, printable, nul, high, ws, alpha, digit, ctrl,
        mean, std, nonzero_bins, peak, mean_delta, autocorr1, null_run,
    ], axis=1).astype(np.float32)
    names = ["entropy", "zlib_ratio", "printable", "null_frac", "high_frac",
             "ws_frac", "alpha_frac", "digit_frac", "ctrl_frac", "mean", "std",
             "nonzero_bins", "peak_freq", "mean_delta", "autocorr1", "null_run"]

    feats = np.concatenate([scalars, struct, bg], axis=1)
    names = names + _STRUCT_CHAR_NAMES + _STRUCT_BIGRAM_NAMES
    return feats.astype(np.float32), names


def ncd_features(block: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Multi-compressor ratios. SLOW (lzma ~ms/fragment); opt-in only."""
    B, L = block.shape
    out = np.empty((B, 5), dtype=np.float32)
    for i in range(B):
        raw = block[i].tobytes()
        z = len(zlib.compress(raw, 9)) / L
        b = len(bz2.compress(raw, 9)) / L
        x = len(lzma.compress(raw, preset=1)) / L
        out[i] = [z, b, x, z - x, b - x]
    return out, ["ncd_zlib", "ncd_bz2", "ncd_lzma", "ncd_z_minus_x", "ncd_b_minus_x"]


_GROUPS = {"stats": stats_features, "hist": hist_features, "ncd": ncd_features}


def extract(block: np.ndarray, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    parts, names = [], []
    for g in groups:
        f, n = _GROUPS[g](block)
        parts.append(f)
        names.extend(n)
    return np.concatenate(parts, axis=1), names


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_paths(features_dir, split, group):
    d = Path(features_dir)
    return d / f"{split}.{group}.npy", d / f"{split}.{group}.names.json"


def build_cache(binary_dir, features_dir, split, groups, chunk=8192,
                progress=True):
    """Compute & cache the requested feature groups for a whole split."""
    Path(features_dir).mkdir(parents=True, exist_ok=True)
    frags, _, meta = _data.open_split(binary_dir, split)
    N = meta["n_samples"]

    for group in groups:
        arr_path, names_path = _cache_paths(features_dir, split, group)
        if arr_path.exists():
            print(f"[features] {split}.{group} exists, skip")
            continue
        # probe dimensionality + names on a tiny batch
        probe, names = _GROUPS[group](np.asarray(frags[:4], dtype=np.uint8))
        F = probe.shape[1]
        mm = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float32,
                                       shape=(N, F))
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            blk = np.asarray(frags[s:e], dtype=np.uint8)
            feats, _ = _GROUPS[group](blk)
            mm[s:e] = feats
            if progress and (s // chunk) % 20 == 0:
                print(f"[features] {split}.{group} {e}/{N}", flush=True)
        mm.flush()
        del mm
        names_path.write_text(json.dumps(names))
        print(f"[features] wrote {arr_path} shape=({N},{F})")


def load_features(features_dir, split, feature_set, rows=None):
    """Load and concat the groups in feature_set (e.g. 'stats_hist').

    rows: optional row index array to gather (for subsampled training).
    Returns (X float32 (n, F), names).
    """
    groups = feature_set.split("_")
    mms, dims, names = [], [], []
    for g in groups:
        arr_path, names_path = _cache_paths(features_dir, split, g)
        if not arr_path.exists():
            raise FileNotFoundError(
                f"feature cache missing: {arr_path}. Run the 'features' command.")
        mm = np.load(arr_path, mmap_mode="r")
        mms.append(mm)
        dims.append(mm.shape[1])
        names.extend(json.loads(names_path.read_text()))
    # If rows cover the whole split contiguously (the max_per_class=null case),
    # drop to a plain slice — fancy-indexing a 6.3 GB memmap would allocate a
    # full temp; a slice streams straight from disk into X.
    full_n = mms[0].shape[0]
    if rows is not None and rows.shape[0] == full_n and \
       rows[0] == 0 and rows[-1] == full_n - 1 and \
       np.array_equal(rows, np.arange(full_n)):
        rows = None
    # Preallocate the final matrix and fill each group in place — avoids the
    # transient doubling of np.concatenate, which matters at full scale (6.14M
    # rows x ~310 features ~ 7.6 GB).
    n = (len(rows) if rows is not None else full_n)
    X = np.empty((n, sum(dims)), dtype=np.float32)
    off = 0
    for mm, d in zip(mms, dims):
        X[:, off:off + d] = mm[rows] if rows is not None else mm[:]
        off += d
    return X, names


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-set z-score parameters (float64 for stability, guarded std)."""
    mean = X.mean(0, dtype=np.float64).astype(np.float32)
    std = X.std(0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def feature_dim(features_dir, split, feature_set) -> int:
    groups = feature_set.split("_")
    total = 0
    for g in groups:
        _, names_path = _cache_paths(features_dir, split, g)
        total += len(json.loads(names_path.read_text()))
    return total
