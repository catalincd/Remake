"""Command-line entrypoint: `python -m remake.cli <command> ...`.

Commands
--------
  list                          list all registered models
  features  --groups stats,hist build/refresh the feature cache
  train     --config FILE       train one model (NN or tree, auto-dispatched)
  eval      --run DIR           print saved metrics for a finished run
  stack     --runs A B C ...    fit a meta-learner over base-run predictions

Quick experiments without editing YAML: `--set key.path=value` (repeatable),
e.g. `--set train.epochs=5 data.max_per_class=3000 name=lgbm_smoke`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from . import models  # noqa: F401  (registers all models on import)
from . import registry
from .config import Config


def _apply_overrides(d: dict, overrides: list[str]) -> dict:
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return d


def cmd_list(_args):
    for name, spec in sorted(registry.all_specs().items()):
        print(f"  {name:14s} [{spec.kind:4s}/{spec.input:8s}] {spec.description}")


def cmd_features(args):
    from . import features as Feat
    groups = args.groups.split(",")
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for sp in splits:
        Feat.build_cache(args.binary_dir, args.features_dir, sp, groups,
                         chunk=args.chunk)


def cmd_train(args):
    with open(args.config) as fh:
        raw = yaml.safe_load(fh)
    raw = _apply_overrides(raw, args.set)
    cfg = Config.from_dict(raw)
    spec = registry.get(cfg.model)
    if spec.kind == "nn":
        from . import trainer
        run_dir = trainer.train(cfg, spec)
    else:
        from . import tree_trainer
        run_dir = tree_trainer.train(cfg, spec)
    print("RUN_DIR:", run_dir)


def cmd_eval(args):
    run = Path(args.run)
    m = run / f"metrics_{args.split}.json"
    if not m.exists():
        raise SystemExit(f"no metrics at {m}")
    summ = json.loads(m.read_text())
    print(f"== {run.name} [{args.split}] ==")
    print(f"accuracy      : {summ['accuracy']:.4f}")
    if "coarse_accuracy" in summ:
        print(f"coarse acc    : {summ['coarse_accuracy']:.4f}")
    print("worst classes :")
    pc = sorted(summ["per_class"].items(), key=lambda kv: kv[1]["recall"])[:10]
    for n, d in pc:
        print(f"   {n:10s} recall={d['recall']:.3f} prec={d['precision']:.3f} "
              f"n={d['support']}")
    from . import metrics, taxonomy
    if "coarse_confusion_matrix" in summ:
        rep = metrics.format_confusion(
            summ["coarse_confusion_matrix"], taxonomy.GROUP_NAMES,
            title="confusion — coarse-11 collapse (full 75x75 in metrics_*.json)")
    else:
        rep = metrics.format_confusion(
            summ["confusion_matrix"], summ["class_names"],
            title="confusion matrix (row=true, col=pred; [diag]=recall, prec at foot)")
    if rep:
        print(rep)


def cmd_stack(args):
    from . import stacking
    stacking.run(args.runs, args.label_space, meta=args.meta, name=args.name)


def main():
    ap = argparse.ArgumentParser(prog="remake")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    pf = sub.add_parser("features")
    pf.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    pf.add_argument("--groups", default="stats,hist")
    pf.add_argument("--binary-dir", default="data/4k_1/binary")
    pf.add_argument("--features-dir", default="data/4k_1/features")
    pf.add_argument("--chunk", type=int, default=8192)
    pf.set_defaults(func=cmd_features)

    pt = sub.add_parser("train")
    pt.add_argument("--config", required=True)
    pt.add_argument("--set", nargs="*", default=[], help="dotted overrides key=val")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval")
    pe.add_argument("--run", required=True)
    pe.add_argument("--split", default="test", choices=["val", "test"])
    pe.set_defaults(func=cmd_eval)

    ps = sub.add_parser("stack")
    ps.add_argument("--runs", nargs="+", required=True)
    ps.add_argument("--label-space", default="coarse11")
    ps.add_argument("--meta", default="logreg", choices=["logreg", "lgbm", "mean"])
    ps.add_argument("--name", default="stack")
    ps.set_defaults(func=cmd_stack)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
