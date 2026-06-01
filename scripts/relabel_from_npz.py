#!/usr/bin/env python3
"""
Rebuild correct label files from the original FiFTy NPZ, anchored to FiFTy's
canonical 75-class order (see remake/taxonomy.py).

Why this exists
---------------
The previous `convert_npz_to_binary.py` assumed the NPZ integer labels matched a
hand-written 58-class order. They don't — they're in FiFTy's official order.
That silently (a) permuted every label and (b) collapsed FiFTy classes 58..74
(incl. JSON/HTML/XML/LOG/CSV) into class 0. Result: "text" was really compressed
archives, the real text formats vanished, and the model hit a hard 87.8% wall.

The fragment bytes were never corrupted and are still in NPZ row order (verified:
bin_label[i] == remap(npz_y[i]) for all rows, fragment row bytes identical). So
the fix is just: write the labels straight from npz `y` (0..74), reuse the
existing fragment .bin via symlink, and write correct metadata.

Usage
-----
    python scripts/relabel_from_npz.py \
        --npz-dir   4k_1 \
        --frag-src  ../FFT/FFT-75-Hierarchical/data/4k_1/binary \
        --out-dir   data/4k_1/binary \
        [--copy-fragments]      # copy instead of symlink (30 GB)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import zipfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from remake.taxonomy import LEAF_NAMES, LEAF_TAG, NUM_LEAVES  # noqa: E402

SECTOR = 4096
SPLITS = ("train", "val", "test")


def read_npy_member(zf: zipfile.ZipFile, name: str,
                    with_data: bool = True) -> tuple[dict, bytes]:
    """Return (header_dict, raw_data_bytes) for a .npy member of an .npz.

    with_data=False reads only the header (cheap) — never decompress the giant
    x.npy array just to learn its shape.
    """
    with zf.open(name) as fh:
        assert fh.read(6) == b"\x93NUMPY", "not a .npy stream"
        fh.read(2)                                   # version
        hlen = int.from_bytes(fh.read(2), "little")
        header = ast.literal_eval(fh.read(hlen).decode())
        return header, (fh.read() if with_data else b"")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, default=Path("4k_1"))
    ap.add_argument("--frag-src", type=Path,
                    default=Path("../FFT/FFT-75-Hierarchical/data/4k_1/binary"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/4k_1/binary"))
    ap.add_argument("--copy-fragments", action="store_true",
                    help="copy fragment .bin instead of symlinking (uses 30 GB)")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    frag_src = args.frag_src.resolve()

    for split in SPLITS:
        npz_path = args.npz_dir / f"{split}.npz"
        if not npz_path.exists():
            print(f"[{split}] SKIP — {npz_path} not found")
            continue

        with zipfile.ZipFile(npz_path) as zf:
            yhdr, ydata = read_npy_member(zf, "y.npy")
            xhdr, _ = read_npy_member(zf, "x.npy", with_data=False)  # header only

        n = yhdr["shape"][0]
        sector = xhdr["shape"][1]
        assert sector == SECTOR, f"unexpected sector size {sector}"
        assert yhdr["descr"] in ("|u1", "<u1", "u1"), f"label dtype {yhdr['descr']}"
        assert len(ydata) == n, (len(ydata), n)
        assert max(ydata) < NUM_LEAVES, f"label {max(ydata)} >= {NUM_LEAVES}"

        # ---- labels: write npz y verbatim (already correct 0..74) ----
        (out / f"{split}_labels.bin").write_bytes(ydata)

        # ---- fragments: symlink (or copy) the verified-correct bytes ----
        dst = out / f"{split}_fragments.bin"
        src = frag_src / f"{split}_fragments.bin"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy_fragments:
            shutil.copy(src, dst)
        else:
            os.symlink(src, dst)

        # ---- meta ----
        meta = {
            "split": split,
            "n_samples": n,
            "sector_size": SECTOR,
            "n_classes": NUM_LEAVES,
            "all_types": LEAF_NAMES,        # FiFTy order, the truth
            "tag_per_class": LEAF_TAG,
            "frag_dtype": "uint8",
            "label_dtype": "uint8",
            "source": "fixed from NPZ y (FiFTy canonical order)",
        }
        (out / f"{split}_meta.json").write_text(json.dumps(meta, indent=2))

        from collections import Counter
        c = Counter(ydata)
        spread = f"min={min(c.values())} max={max(c.values())}"
        print(f"[{split}] n={n:>9}  classes={len(c)}  per-class {spread}  "
              f"-> {'copied' if args.copy_fragments else 'symlinked'} fragments")

    print("\nDone. Corrected dataset at:", out)


if __name__ == "__main__":
    main()
