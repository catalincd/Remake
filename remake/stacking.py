"""Stacking / blending ensemble over saved base-model predictions.

Each base run saved val_probs/val_true and test_probs/test_true (same label
space, same deterministic val/test ordering). We fit a meta-learner on the
*validation* probabilities (held out from base training) and evaluate on test.
Base errors are partly orthogonal (trees nail entropy/histogram; byte models nail
sequence rhythm), so the blend is the most likely route past 90%.

Also supports a no-fit 'mean' blend (simple probability averaging).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import metrics
from .logging_utils import RunLogger


def _load(run_dir: Path, split: str):
    p = run_dir / f"{split}_probs.npy"
    t = run_dir / f"{split}_true.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run that model first")
    return np.load(p), np.load(t)


def _stack_features(run_dirs, split):
    probs, truth = [], None
    for rd in run_dirs:
        pr, tr = _load(Path(rd), split)
        probs.append(pr)
        if truth is None:
            truth = tr
        elif not np.array_equal(truth, tr):
            raise ValueError(f"label ordering mismatch in {rd} for {split} split; "
                             "ensure same label_space/seed/caps across base runs")
    return np.concatenate(probs, axis=1), truth


def run(run_dirs, label_space, meta="logreg", name="stack", log_root="runs") -> str:
    run_dirs = [Path(r) for r in run_dirs]
    Xva, yva = _stack_features(run_dirs, "val")
    Xte, yte = _stack_features(run_dirs, "test")

    cfg = {"name": name, "model": f"stack_{meta}", "label_space": label_space,
           "base_runs": [str(r) for r in run_dirs], "meta": meta}
    logger = RunLogger(cfg, log_root=log_root)
    print(f"[stack] {name} bases={len(run_dirs)} meta={meta} "
          f"Xval={Xva.shape} Xtest={Xte.shape}")

    if meta == "mean":
        C = yva.max() + 1
        k = Xte.shape[1] // C
        test_probs = Xte.reshape(Xte.shape[0], k, C).mean(1)
    else:
        if meta == "logreg":
            from sklearn.linear_model import LogisticRegression
            # sklearn >=1.5 dropped multi_class (always multinomial for >2 classes)
            clf = LogisticRegression(max_iter=2000, C=1.0)
        elif meta == "lgbm":
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                     learning_rate=0.05, verbose=-1)
        else:
            raise ValueError(f"unknown meta '{meta}'")
        clf.fit(Xva, yva)
        test_probs = clf.predict_proba(Xte)
        classes = getattr(clf, "classes_", None)
        if classes is not None:
            C = int(yva.max() + 1)
            full = np.zeros((test_probs.shape[0], C), dtype=np.float32)
            full[:, classes.astype(int)] = test_probs
            test_probs = full
        logger.save_blob("meta_model.pt", clf)

    logger.save_predictions("test", test_probs, yte)
    summ = metrics.summarize(yte, test_probs.argmax(1), label_space)
    logger.save_summary("test", summ)
    extra = f" coarse={summ.get('coarse_accuracy'):.4f}" if "coarse_accuracy" in summ else ""
    print(f"  [stack test] acc={summ['accuracy']:.4f}{extra}")
    rep = metrics.confusion_report(summ, label_space)
    if rep:
        print(rep)
        logger.save_text("confusion_test.txt", rep)
    logger.set_best({"epoch": 0, "val_acc": summ["accuracy"]})
    logger.finalize("complete", extra={"test_accuracy": summ["accuracy"]})
    return str(logger.dir)
