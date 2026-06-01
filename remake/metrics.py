"""Evaluation metrics: overall/per-class accuracy, confusion matrix, and for
the flat-75 space a collapse to the 11 coarse groups (so we can read both the
fine number and the group-level number from a single set of predictions).
"""
from __future__ import annotations

import numpy as np

from . import taxonomy


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> np.ndarray:
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return cm


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1).clip(min=1)
    return cm.diagonal() / denom


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=0).clip(min=1)
    return cm.diagonal() / denom


def summarize(y_true: np.ndarray, y_pred: np.ndarray, label_space: str) -> dict:
    """Full metric bundle for one set of predictions."""
    names = taxonomy.class_names(label_space)
    C = len(names)
    cm = confusion_matrix(y_true, y_pred, C)
    rec = per_class_recall(cm)
    prec = per_class_precision(cm)
    acc = float((y_true == y_pred).mean())

    out = {
        "accuracy": acc,
        "macro_recall": float(rec.mean()),
        "n": int(y_true.shape[0]),
        "per_class": {
            names[i]: {"recall": float(rec[i]), "precision": float(prec[i]),
                       "support": int(cm[i].sum())}
            for i in range(C)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": names,
    }

    # When predicting all 75 leaves, also report the coarse-11 collapse.
    if label_space == "flat75":
        lut = np.asarray(taxonomy.LEAF_TO_GROUP)
        gt, gp = lut[y_true], lut[y_pred]
        gC = taxonomy.NUM_GROUPS
        gcm = confusion_matrix(gt, gp, gC)
        out["coarse_accuracy"] = float((gt == gp).mean())
        out["coarse_per_class"] = {
            taxonomy.GROUP_NAMES[i]: {
                "recall": float(per_class_recall(gcm)[i]),
                "precision": float(per_class_precision(gcm)[i]),
                "support": int(gcm[i].sum()),
            } for i in range(gC)
        }
        out["coarse_confusion_matrix"] = gcm.tolist()
    return out


def worst_classes(summary: dict, k: int = 10) -> list[tuple[str, float]]:
    pc = summary["per_class"]
    ranked = sorted(((n, d["recall"]) for n, d in pc.items()), key=lambda kv: kv[1])
    return ranked[:k]


def format_confusion(cm, names, title: str | None = None, max_classes: int = 20):
    """Old-project-style text table: row=true, col=pred, percentages.

    Diagonal cells show recall in [ ]; a precision row is printed at the foot.
    Cells < 0.5% are left blank for readability. Returns None if there are too
    many classes to print legibly (caller should fall back to a collapse).
    """
    cm = np.asarray(cm, dtype=np.int64)
    C = len(names)
    if C == 0 or C > max_classes:
        return None
    rec = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    prec = cm.diagonal() / np.clip(cm.sum(0), 1, None)
    abbr = [str(n)[:6] for n in names]
    cw = 7
    lw = max([8] + [len(str(n)) for n in names])
    out = []
    if title:
        out.append(f"  {title}")
    out.append(" " * (lw + 3) + "".join(f"{a:>{cw}}" for a in abbr))
    out.append("  " + " " * lw + "-" * (cw * C + 1))
    for i in range(C):
        cells = []
        for j in range(C):
            v = rec[i, j] * 100
            if i == j:
                cells.append(f"[{v:3.0f}%]".rjust(cw))
            elif v >= 0.5:
                cells.append(f"{v:.0f}%".rjust(cw))
            else:
                cells.append(" " * cw)
        out.append(f"  {str(names[i]):>{lw}} " + "".join(cells))
    out.append("  " + " " * lw + "-" * (cw * C + 1))
    out.append(f"  {'prec':>{lw}} " + "".join(f"{prec[j]*100:.0f}%".rjust(cw)
                                               for j in range(C)))
    return "\n".join(out)


def confusion_report(summary: dict, label_space: str) -> str | None:
    """Pick the right matrix to print: the native one, or (for flat75) the
    coarse-11 collapse since a 75x75 text table is unreadable."""
    if label_space == "flat75":
        cm = summary.get("coarse_confusion_matrix")
        names = taxonomy.GROUP_NAMES
        title = ("confusion matrix — coarse-11 collapse (row=true, col=pred); "
                 "full 75x75 saved in metrics_*.json")
    else:
        cm = summary.get("confusion_matrix")
        names = summary.get("class_names")
        title = "confusion matrix (row=true, col=pred; [diag]=recall, prec at foot)"
    if cm is None or names is None:
        return None
    return format_confusion(cm, names, title=title)
