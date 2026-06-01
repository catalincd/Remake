"""Train/evaluate tree models on cached features.

Mirrors the NN trainer's outputs (run dir, JSON log, TensorBoard scalars, saved
predicted probabilities + metric summaries) so tree and NN runs are directly
comparable and both feed the stacking ensemble. For LightGBM/XGBoost the
per-iteration validation curve is logged to TensorBoard (the boosting "sliders").
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import data as D
from . import features as Feat
from . import metrics
from . import taxonomy
from .config import Config
from .logging_utils import RunLogger
from .registry import ModelSpec


def _load_xy(cfg: Config, split, cap):
    idx, y = D.split_indices(cfg.data.binary_dir, split, cfg.label_space, cap, cfg.data.seed)
    X, names = Feat.load_features(cfg.data.features_dir, split, cfg.features, rows=idx)
    return X, y, names


def train(cfg: Config, spec: ModelSpec) -> str:
    num_classes = taxonomy.num_classes(cfg.label_space)
    t0 = time.time()

    Xtr, ytr, names = _load_xy(cfg, "train", cfg.data.max_per_class)
    Xva, yva, _ = _load_xy(cfg, "val", cfg.data.val_max_per_class)

    log_cfg = cfg.to_dict()
    log_cfg.update({"model_kind": spec.kind, "model_input": spec.input,
                    "num_classes": num_classes, "n_features": Xtr.shape[1],
                    "n_train": int(Xtr.shape[0])})
    logger = RunLogger(log_cfg, log_root=cfg.log_root)
    print(f"[tree] {cfg.name} model={cfg.model} X={Xtr.shape} classes={num_classes}")

    model = spec.builder(num_classes=num_classes, **cfg.model_args)

    # Fit (with eval curve where the backend supports it)
    fit_log = {}
    try:
        if cfg.model == "lgbm":
            import lightgbm as lgb
            rec = {}
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="multi_error",
                      callbacks=[lgb.record_evaluation(rec),
                                 lgb.log_evaluation(period=50)])
            curve = 1.0 - np.asarray(rec["valid_0"]["multi_error"])
            for i, v in enumerate(curve):
                logger.scalars(i, {"val/acc": float(v)})
            fit_log["val_curve"] = curve.tolist()
        elif cfg.model == "xgb":
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            res = model.evals_result()
            if res:
                key = list(res.keys())[0]
                metric = list(res[key].keys())[0]
                for i, v in enumerate(res[key][metric]):
                    logger.scalars(i, {f"val/{metric}": float(v)})
                fit_log["val_curve_metric"] = metric
        else:
            model.fit(Xtr, ytr)
    except Exception:
        model.fit(Xtr, ytr)

    # Feature importances (where available)
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            order = np.argsort(imp)[::-1][:40]
            fit_log["top_features"] = [(names[i], float(imp[i])) for i in order]
    except Exception:
        pass

    logger.save_blob("model.pt", model)

    # Evaluate on val + test, dump probs + metrics
    best = {}
    for split, cap in (("val", cfg.data.val_max_per_class),
                       ("test", cfg.data.val_max_per_class)):
        X, y, _ = _load_xy(cfg, split, cap)
        probs = model.predict_proba(X)
        # align column order to 0..C-1 if estimator exposes classes_
        classes = getattr(model, "classes_", None)
        if classes is not None and not np.array_equal(classes, np.arange(num_classes)):
            full = np.zeros((probs.shape[0], num_classes), dtype=np.float32)
            full[:, classes.astype(int)] = probs
            probs = full
        logger.save_predictions(split, probs, y)
        summ = metrics.summarize(y, probs.argmax(1), cfg.label_space)
        logger.save_summary(split, summ)
        if split == "val":
            best = {"epoch": 0, "val_acc": summ["accuracy"]}
        extra = f" coarse={summ.get('coarse_accuracy'):.4f}" if "coarse_accuracy" in summ else ""
        print(f"  [{split}] acc={summ['accuracy']:.4f}{extra}")
        if split == "test":
            rep = metrics.confusion_report(summ, cfg.label_space)
            if rep:
                print(rep)
                logger.save_text("confusion_test.txt", rep)

    logger.set_best(best)
    logger.finalize("complete", extra={"fit": fit_log,
                                       "fit_time_s": time.time() - t0})
    return str(logger.dir)
