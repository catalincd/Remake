"""Unified NN training loop (byte models and the feature-MLP).

Logs everything per epoch (loss/acc/lr/grad-norm/epoch-time/GPU-mem) to
TensorBoard + JSON, checkpoints every epoch, keeps the best by val accuracy, and
at the end dumps val/test predicted probabilities (for stacking) plus full
metric summaries with confusion matrices.
"""
from __future__ import annotations

import math
import time

try:
    from tqdm import tqdm
except Exception:  # tqdm optional — fall back to a no-op
    tqdm = None

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data as D
from . import features as Feat
from . import metrics
from . import taxonomy
from .config import Config
from .logging_utils import RunLogger
from .registry import ModelSpec


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model, lr, wd):
    """AdamW with no weight decay on norms / biases / embeddings."""
    decay, no_decay = [], []
    no_decay_mods = (nn.LayerNorm, nn.BatchNorm1d, nn.Embedding)
    nd_ids = {id(p) for m in model.modules() if isinstance(m, no_decay_mods)
              for p in m.parameters(recurse=False)}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (id(p) in nd_ids or n.endswith(".bias")) else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999))


def cosine_warmup(step, total, warmup, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def _make_loaders(cfg: Config, spec: ModelSpec, num_classes: int):
    d = cfg.data
    train_idx, train_y = D.split_indices(d.binary_dir, "train", cfg.label_space,
                                         d.max_per_class, d.seed)
    val_idx, val_y = D.split_indices(d.binary_dir, "val", cfg.label_space,
                                     d.val_max_per_class, d.seed)
    input_dim, std_params = None, None
    if spec.input == "bytes":
        train_ds = D.make_byte_dataset(d.binary_dir, "train", cfg.label_space,
                                       indices=train_idx, labels=train_y,
                                       augment=cfg.train.augment,
                                       gbflip_sigma=cfg.train.gbflip_sigma,
                                       gbflip_max_rate=cfg.train.gbflip_max_rate,
                                       seed=cfg.train.seed)
        val_ds = D.make_byte_dataset(d.binary_dir, "val", cfg.label_space,
                                     indices=val_idx, labels=val_y)
    else:  # features — z-score with train statistics (tabular best practice)
        Xtr, names = Feat.load_features(d.features_dir, "train", cfg.features, rows=train_idx)
        Xva, _ = Feat.load_features(d.features_dir, "val", cfg.features, rows=val_idx)
        mean, std = Feat.standardize_fit(Xtr)
        std_params = (mean, std)
        input_dim = Xtr.shape[1]
        train_ds = D.make_feature_dataset(Feat.standardize_apply(Xtr, mean, std), train_y)
        val_ds = D.make_feature_dataset(Feat.standardize_apply(Xva, mean, std), val_y)
    nw = cfg.train.num_workers
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size * 2, shuffle=False,
                        num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    return train_ds, val_dl, input_dim, std_params, nw


def _train_loader(train_ds, batch_size, nw, persistent):
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      num_workers=nw, pin_memory=True, drop_last=True,
                      persistent_workers=persistent and nw > 0)


@torch.no_grad()
def _predict(model, dl, device, num_classes, desc="eval"):
    model.eval()
    probs, ys = [], []
    correct = seen = 0
    it = tqdm(dl, desc=f"  {desc}", unit="batch", leave=False,
              bar_format="{l_bar}{bar:30}{r_bar}") if tqdm is not None else dl
    for xb, yb in it:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(xb)
        p = torch.softmax(logits.float(), -1)
        probs.append(p.cpu().numpy())
        ys.append(yb.numpy())
        if tqdm is not None:
            correct += (p.argmax(-1).cpu() == yb).sum().item()
            seen += yb.size(0)
            it.set_postfix(acc=f"{correct/seen:.3f}", refresh=False)
    return np.concatenate(probs), np.concatenate(ys)


def train(cfg: Config, spec: ModelSpec) -> str:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = _device()
    num_classes = taxonomy.num_classes(cfg.label_space)

    train_ds, val_dl, input_dim, std_params, nw = _make_loaders(cfg, spec, num_classes)
    # Per-epoch batch-size jitter (regulariser): bs ~ base ± batch_size_jitter.
    # When jittering we rebuild the train loader each epoch, so don't keep
    # workers persistent (they'd leak across rebuilds).
    bs_jitter = int(cfg.train.batch_size_jitter)
    train_dl = _train_loader(train_ds, cfg.train.batch_size, nw, persistent=bs_jitter == 0)
    model = spec.builder(num_classes=num_classes, input_dim=input_dim,
                         **cfg.model_args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if cfg.train.compile:
        model = torch.compile(model)

    log_cfg = cfg.to_dict()
    log_cfg["params"] = int(n_params)
    log_cfg["model_kind"] = spec.kind
    log_cfg["model_input"] = spec.input
    log_cfg["num_classes"] = num_classes
    logger = RunLogger(log_cfg, log_root=cfg.log_root)
    print(f"[train] {cfg.name}  model={cfg.model}  params={n_params/1e6:.2f}M  "
          f"classes={num_classes}  device={device}")

    opt = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    # Optional confusion penalty: discourage routing wrong samples into a "sink"
    # class (here `archive`, which every high-entropy class bleeds into).
    conf_lambda = float(cfg.train.confusion_lambda)
    conf_idx = None
    if conf_lambda > 0:
        names = taxonomy.class_names(cfg.label_space)
        if cfg.train.confusion_target not in names:
            raise ValueError(f"confusion_target {cfg.train.confusion_target!r} "
                             f"not in {cfg.label_space} classes: {names}")
        conf_idx = names.index(cfg.train.confusion_target)
        print(f"[train] confusion penalty λ={conf_lambda} on "
              f"'{cfg.train.confusion_target}' (idx {conf_idx})")

    steps_per_epoch = len(train_dl) // cfg.train.grad_accum
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup = int(cfg.train.warmup_pct * total_steps)

    ema = None
    if cfg.train.ema_decay > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    best_acc, best_epoch, since_best, gstep = -1.0, 0, 0, 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0 = time.time()
        if bs_jitter > 0:
            bs = max(1, cfg.train.batch_size + int(np.random.randint(-bs_jitter, bs_jitter + 1)))
            train_dl = _train_loader(train_ds, bs, nw, persistent=False)
            print(f"  [epoch {epoch:02d}] batch_size={bs}")
        run_loss, run_correct, run_n, gnorm = 0.0, 0, 0, 0.0
        opt.zero_grad(set_to_none=True)
        desc = f"epoch {epoch:>{len(str(cfg.train.epochs))}}/{cfg.train.epochs} [train]"
        pbar = (tqdm(total=len(train_dl), desc=desc, unit="batch",
                     bar_format="{l_bar}{bar:30}{r_bar}") if tqdm is not None else None)
        for bi, (xb, yb) in enumerate(train_dl):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(xb)
                loss = crit(logits, yb)
                if conf_idx is not None:
                    # mean prob mass placed on the sink class over samples whose
                    # true label is NOT the sink — pushes the model off the lazy default.
                    p_sink = torch.softmax(logits.float(), -1)[:, conf_idx]
                    mask = (yb != conf_idx).float()
                    loss = loss + conf_lambda * (p_sink * mask).sum() / mask.sum().clamp_min(1.0)
                loss = loss / cfg.train.grad_accum
            scaler.scale(loss).backward()
            if (bi + 1) % cfg.train.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = cosine_warmup(gstep, total_steps, warmup,
                                            cfg.train.lr, cfg.train.min_lr)
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg.train.grad_clip).item()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    sd = model.state_dict()
                    for k in ema:
                        if ema[k].dtype.is_floating_point:
                            ema[k].mul_(cfg.train.ema_decay).add_(
                                sd[k].detach(), alpha=1 - cfg.train.ema_decay)
                        else:
                            ema[k].copy_(sd[k])
                gstep += 1
            run_loss += loss.item() * cfg.train.grad_accum * yb.size(0)
            run_correct += (logits.argmax(-1) == yb).sum().item()
            run_n += yb.size(0)
            if pbar is not None:
                pbar.set_postfix(loss=f"{run_loss/run_n:.4f}",
                                 acc=f"{run_correct/run_n:.3f}",
                                 lr=f"{opt.param_groups[0]['lr']:.2e}",
                                 gnorm=f"{gnorm:.2f}", refresh=False)
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        # ---- eval ----
        if ema is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema)
        val_probs, val_true = _predict(model, val_dl, device, num_classes, desc="val")
        val_pred = val_probs.argmax(1)
        val_acc = float((val_pred == val_true).mean())
        # probs are already softmaxed -> NLL on log-probs (don't re-log_softmax)
        val_loss = float(nn.functional.nll_loss(
            torch.from_numpy(val_probs).clamp_min(1e-9).log(),
            torch.from_numpy(val_true)).item())

        gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
        rec = {
            "epoch": epoch,
            "train_loss": run_loss / run_n,
            "train_acc": run_correct / run_n,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": opt.param_groups[0]["lr"],
            "grad_norm": gnorm,
            "elapsed_s": time.time() - t0,
            "gpu_mem_mb": gpu_mb,
        }
        logger.epoch(rec)
        print(f"  e{epoch:02d} train_acc={rec['train_acc']:.4f} "
              f"val_acc={val_acc:.4f} loss={val_loss:.4f} "
              f"{rec['elapsed_s']:.0f}s gpu={gpu_mb:.0f}MB")
        rep = metrics.confusion_report(
            metrics.summarize(val_true, val_pred, cfg.label_space), cfg.label_space)
        if rep:
            print(rep)

        is_best = val_acc > best_acc
        state = {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                 "config": log_cfg}
        logger.save_checkpoint(state, epoch, is_best=is_best)
        if is_best:
            best_acc, best_epoch, since_best = val_acc, epoch, 0
            logger.set_best({"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss})
        else:
            since_best += 1
        if ema is not None:
            model.load_state_dict(backup)
        if cfg.train.early_stop_patience and since_best >= cfg.train.early_stop_patience:
            print(f"  early stop at epoch {epoch} (best={best_acc:.4f}@{best_epoch})")
            break

    # ---- final: load best, score val+test, dump probs + metrics ----
    best_path = logger.ckpt_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    _finalize(cfg, spec, model, device, num_classes, logger, std_params)
    logger.finalize("complete", extra={"best_val_acc": best_acc, "best_epoch": best_epoch})
    return str(logger.dir)


def _finalize(cfg, spec, model, device, num_classes, logger, std_params):
    d = cfg.data
    for split, cap in (("val", d.val_max_per_class), ("test", d.val_max_per_class)):
        idx, y = D.split_indices(d.binary_dir, split, cfg.label_space, cap, d.seed)
        if spec.input == "bytes":
            ds = D.make_byte_dataset(d.binary_dir, split, cfg.label_space,
                                     indices=idx, labels=y)
        else:
            X, _ = Feat.load_features(d.features_dir, split, cfg.features, rows=idx)
            if std_params is not None:
                X = Feat.standardize_apply(X, *std_params)
            ds = D.make_feature_dataset(X, y)
        dl = DataLoader(ds, batch_size=cfg.train.batch_size * 2, shuffle=False,
                        num_workers=cfg.train.num_workers, pin_memory=True)
        probs, true = _predict(model, dl, device, num_classes, desc=split)
        logger.save_predictions(split, probs, true)
        summ = metrics.summarize(true, probs.argmax(1), cfg.label_space)
        logger.save_summary(split, summ)
        extra = f" coarse={summ.get('coarse_accuracy'):.4f}" if "coarse_accuracy" in summ else ""
        print(f"  [{split}] acc={summ['accuracy']:.4f}{extra}")
        if split == "test":
            rep = metrics.confusion_report(summ, cfg.label_space)
            if rep:
                print(rep)
                logger.save_text("confusion_test.txt", rep)
