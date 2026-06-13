#!/usr/bin/env python3
"""End-to-end two-stage cascade evaluator (the *real* headline number).

Phase-1 router (coarse-11) predicts a group for every test fragment; the
fragment is then handed to that group's leaf specialist, whose local prediction
is mapped back to a global FiFTy leaf (0..74). Unlike `summarize.py` (which
scores each specialist on its group's *true* members and so assumes perfect
routing), this pays for routing errors: a fragment misrouted into `archive`
is judged by the archive specialist, which cannot recover it.

Reports three numbers on the test split:
  - coarse-11 routing accuracy        (phase-1 only)
  - oracle leaf accuracy              (route by TRUE group -> specialist ceiling)
  - full cascade leaf accuracy        (route by PREDICTED group -> honest 75-way)

Usage
-----
  python phase2/cascade.py                         # RESULTS recipe, full test
  python phase2/cascade.py --cap 4000              # subsample per leaf (fast)
  python phase2/cascade.py --router runs/tcn_ft_90_XXXX \
      --spec raw=tcn archive=tcn                   # override router / per-group model
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from remake import data as D          # noqa: E402
from remake import features as Feat   # noqa: E402
from remake import registry           # noqa: E402
from remake import taxonomy           # noqa: E402
from remake import models             # noqa: F401,E402  (registers models)

RUNS = Path("runs")

# Default per-group specialist model = the RESULTS.md recipe.
DEFAULT_SPEC = {g: "lgbm" for g in taxonomy.GROUP_NAMES}
DEFAULT_SPEC.update({"raw": "tcn", "archive": "tcn"})


def latest(prefix: str) -> Path | None:
    cands = sorted(RUNS.glob(f"{prefix}_2*"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _read_cfg(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "config.yaml").read_text())


# --------------------------------------------------------------------------- #
# Per-model inference on an arbitrary subset of test rows
# --------------------------------------------------------------------------- #

class Predictor:
    """Wraps a finished run; `.predict(pos)` returns local class indices for
    the given *positions* into the shared `rows_all` array (which holds absolute
    test-memmap row indices). Working in position space lets tree specialists
    share a single gathered feature matrix `X_sub` aligned to `rows_all`."""

    def __init__(self, run_dir: Path, *, binary_dir, rows_all, device=None,
                 batch_size=4096, num_workers=8, X_sub=None):
        self.run_dir = run_dir
        self.cfg = _read_cfg(run_dir)
        self.spec = registry.get(self.cfg["model"])
        self.binary_dir = binary_dir
        self.rows_all = rows_all          # absolute test rows, aligned to X_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._X_sub = X_sub               # (N, F) gathered for rows_all (tree models)
        self.label_space = self.cfg["label_space"]
        self.num_classes = taxonomy.num_classes(self.label_space)
        self._model = None
        self._device = device

    # -- lazy model load --------------------------------------------------- #
    def _load(self):
        if self._model is not None:
            return
        if self.spec.kind == "tree":
            import torch
            self._model = torch.load(self.run_dir / "model.pt", weights_only=False)
        elif self.spec.kind == "nn":
            import torch
            if self.spec.input != "bytes":
                raise NotImplementedError(
                    f"{self.run_dir.name}: nn/{self.spec.input} not supported "
                    "by cascade (no saved standardization params).")
            dev = self._device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self._device = dev
            m = self.spec.builder(num_classes=self.num_classes, input_dim=None,
                                  **self.cfg.get("model_args", {})).to(dev)
            ckpt = torch.load(self.run_dir / "ckpt" / "best.pt", map_location=dev)
            sd = ckpt.get("model", ckpt)
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            m.load_state_dict(sd)
            m.eval()
            self._model = m
        else:
            raise NotImplementedError(self.spec.kind)

    # -- inference --------------------------------------------------------- #
    def predict(self, pos: np.ndarray) -> np.ndarray:
        """pos: positions into rows_all. Returns local class indices."""
        if pos.size == 0:
            return np.empty(0, dtype=np.int64)
        self._load()
        if self.spec.kind == "tree":
            return self._predict_tree(pos)
        return self._predict_nn(pos)

    def _predict_tree(self, pos):
        probs = self._model.predict_proba(self._X_sub[pos])
        classes = getattr(self._model, "classes_", None)
        local = probs.argmax(1)
        if classes is not None:
            local = np.asarray(classes)[local]
        return local.astype(np.int64)

    def _predict_nn(self, pos):
        import torch
        from torch.utils.data import DataLoader
        rows = self.rows_all[pos]
        dummy = np.zeros(rows.shape[0], dtype=np.int64)
        ds = D.make_byte_dataset(self.binary_dir, "test", self.label_space,
                                 indices=rows, labels=dummy)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=True)
        out = []
        dev = self._device
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(dev, non_blocking=True)
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    logits = self._model(xb)
                out.append(logits.float().argmax(-1).cpu().numpy())
        return np.concatenate(out).astype(np.int64)


# --------------------------------------------------------------------------- #
# Cascade
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default=None,
                    help="phase-1 run dir (default: latest tcn_ft_90)")
    ap.add_argument("--spec", nargs="*", default=[],
                    help="per-group overrides, e.g. raw=tcn archive=lgbm")
    ap.add_argument("--binary-dir", default="data/4k_1/binary")
    ap.add_argument("--features-dir", default="data/4k_1/features")
    ap.add_argument("--features", default="stats_hist")
    ap.add_argument("--cap", type=int, default=None,
                    help="subsample N rows per leaf (default: full test set)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip the oracle (true-routing) pass for max speed")
    args = ap.parse_args()

    t0 = time.time()
    recipe = dict(DEFAULT_SPEC)
    for kv in args.spec:
        g, _, m = kv.partition("=")
        recipe[g] = m

    # Resolve run dirs --------------------------------------------------------
    router_dir = Path(args.router) if args.router else latest("tcn_ft_90")
    if router_dir is None or not router_dir.exists():
        raise SystemExit("router run not found (pass --router runs/<dir>)")
    spec_dirs: dict[str, Path] = {}
    for g in taxonomy.GROUP_NAMES:
        d = latest(f"spec_{g}_{recipe[g]}")
        if d is None:
            raise SystemExit(f"no run for spec_{g}_{recipe[g]} "
                             f"(train it or override with --spec {g}=<model>)")
        spec_dirs[g] = d

    print(f"router  : {router_dir.name}")
    for g in taxonomy.GROUP_NAMES:
        print(f"  {g:11s} -> {recipe[g]:5s}  {spec_dirs[g].name}")

    # Test rows + true labels -------------------------------------------------
    _, raw_labels, _ = D.open_split(args.binary_dir, "test")
    raw_labels = raw_labels.astype(np.int64)
    if args.cap is not None:
        rng = np.random.default_rng(args.seed)
        keep = []
        for c in range(taxonomy.NUM_LEAVES):
            idx = np.nonzero(raw_labels == c)[0]
            if idx.shape[0] > args.cap:
                idx = rng.choice(idx, size=args.cap, replace=False)
            keep.append(idx)
        rows_all = np.sort(np.concatenate(keep))
    else:
        rows_all = np.arange(raw_labels.shape[0])
    y_leaf = raw_labels[rows_all]
    y_group = np.asarray(taxonomy.LEAF_TO_GROUP, dtype=np.int64)[y_leaf]
    N = rows_all.shape[0]
    print(f"\ntest rows: {N}")

    # Shared feature matrix for all tree specialists — gathered for rows_all
    # only (positions align with rows_all), so a capped run never touches the
    # full 768k-row cache.
    need_tree = any(registry.get(recipe[g]).kind == "tree"
                    for g in taxonomy.GROUP_NAMES)
    X_sub = None
    if need_tree:
        print("loading test features ...")
        X_sub, _ = Feat.load_features(args.features_dir, "test", args.features,
                                      rows=rows_all)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_pred(run_dir):
        return Predictor(run_dir, binary_dir=args.binary_dir, rows_all=rows_all,
                         device=device, batch_size=args.batch_size,
                         num_workers=args.num_workers, X_sub=X_sub)

    # Phase-1 routing ---------------------------------------------------------
    print("routing (phase-1) ...")
    router = make_pred(router_dir)
    g_hat = router.predict(np.arange(N))             # predicted group per row
    coarse_acc = float((g_hat == y_group).mean())

    # Phase-2: predicted-routing (cascade) and true-routing (oracle).
    # Run each specialist ONCE over the union of the rows it must judge for
    # either pass, then scatter into both — halves NN specialist inference.
    gleaves = {g: np.asarray(taxonomy.GROUP_LEAVES[g]) for g in taxonomy.GROUP_NAMES}

    pred_cascade = np.full(N, -1, dtype=np.int64)
    pred_oracle = np.full(N, -1, dtype=np.int64)
    per_group = {}
    for gi, g in enumerate(taxonomy.GROUP_NAMES):
        gl = gleaves[g]
        m_cas = np.nonzero(g_hat == gi)[0]                # router sent here
        m_orc = np.array([], dtype=np.int64) if args.no_oracle \
            else np.nonzero(y_group == gi)[0]             # truly belongs here
        u = np.union1d(m_cas, m_orc)
        if u.size:
            t = time.time()
            loc = gl[make_pred(spec_dirs[g]).predict(u)]
            pred_cascade[m_cas] = loc[np.searchsorted(u, m_cas)]
            if m_orc.size:
                pred_oracle[m_orc] = loc[np.searchsorted(u, m_orc)]
            print(f"  {g:11s} {u.size:7d} rows  {time.time()-t:5.1f}s")
        true_here = y_group == gi
        routed_here = g_hat == gi
        per_group[g] = {
            "n_true": int(true_here.sum()),
            "router_recall": float((routed_here & true_here).sum() / max(1, true_here.sum())),
            "router_prec": float((routed_here & true_here).sum() / max(1, routed_here.sum())),
            "oracle_leaf_acc": (None if args.no_oracle else
                                float((pred_oracle[true_here] == y_leaf[true_here]).mean())),
            "cascade_leaf_acc": float((pred_cascade[true_here] == y_leaf[true_here]).mean()),
        }

    cascade_acc = float((pred_cascade == y_leaf).mean())
    oracle_acc = None if args.no_oracle else float((pred_oracle == y_leaf).mean())

    # Report ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("END-TO-END CASCADE  (test split, correct 75-class labels)")
    print("=" * 60)
    print(f"  Phase-1 coarse-11 accuracy   : {coarse_acc:.4f}")
    if oracle_acc is not None:
        print(f"  Oracle leaf acc (true route) : {oracle_acc:.4f}")
    print(f"  Full cascade leaf acc (75-way): {cascade_acc:.4f}")
    if oracle_acc is not None:
        print(f"  routing loss (oracle-cascade): {oracle_acc - cascade_acc:.4f}")
    print("\n  per-group (leaf acc on each group's TRUE members):")
    print(f"    {'group':11s} {'n':>7} {'r_rec':>6} {'r_prec':>7} "
          f"{'oracle':>7} {'cascade':>8} {'drop':>6}")
    print("    " + "-" * 56)
    for g in sorted(taxonomy.GROUP_NAMES,
                    key=lambda g: per_group[g]["cascade_leaf_acc"]):
        p = per_group[g]
        orc = p["oracle_leaf_acc"]
        oc = f"{orc:7.3f}" if orc is not None else f"{'-':>7}"
        dr = f"{orc-p['cascade_leaf_acc']:6.3f}" if orc is not None else f"{'-':>6}"
        print(f"    {g:11s} {p['n_true']:7d} {p['router_recall']:6.3f} "
              f"{p['router_prec']:7.3f} {oc} {p['cascade_leaf_acc']:8.3f} {dr}")
    print(f"\n  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
