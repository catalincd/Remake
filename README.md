# Remake — FFT-75 file-fragment classification zoo

Modular harness to try many architectures (gradient-boosted trees, byte CNN+BiGRU,
TCN, Transformer, Mamba/SSM, byte-transition GNN, feature-MLP, stacking) against
**FFT-75 Scenario #1, 4096-byte** fragments — all behind one data/feature/train/
log/eval pipeline, on a single RX 6750 XT (ROCm).

> **Why this exists:** the previous pipeline hit a hard **87.8%** wall because a
> label-conversion bug scrambled the FiFTy labels (the real text formats were
> collapsed into "jpg"; "text" was actually compressed archives). See
> [DESIGN.md](DESIGN.md) §0. The data is now fixed; this harness benchmarks
> architectures on **correct** labels toward the **≥90% both-phases** target.

## Layout
```
remake/        package: taxonomy, config, data, features, metrics, logging,
               registry, trainer (NN), tree_trainer, stacking, models/
configs/       one YAML per experiment
scripts/       relabel_from_npz.py (the data fix), _smoke.sh
data/4k_1/     corrected dataset (labels) + symlinked fragments + feature cache
runs/          per-run outputs (TensorBoard + JSON + checkpoints + probs)
```

## Quick start
```bash
# 1. open the ROCm container (mounts the workspace so symlinks resolve, sets gfx override)
./run_docker_torch.sh

# --- everything below runs inside the container ---
./run.sh setup                 # install deps into ./.pydeps (once)
./run.sh data                  # (re)build corrected dataset from 4k_1/*.npz  [already done]
./run.sh features              # build feature cache (stats,hist) for all splits
./run.sh list                  # list all models
```

## Train a model
`./run.sh train <config> [--set dotted.key=value ...]`
```bash
./run.sh train configs/lgbm_coarse.yaml             # LightGBM, coarse-11 (start here)
./run.sh train configs/cnn_bigru_coarse.yaml        # reference CNN+BiGRU on correct labels
./run.sh train configs/mamba_coarse.yaml            # pure-torch SSM
./run.sh train configs/gnn_coarse.yaml              # byte-transition GNN
./run.sh train configs/tcn_coarse.yaml
./run.sh train configs/transformer_coarse.yaml
./run.sh train configs/xgb_coarse.yaml
./run.sh train configs/rf_coarse.yaml
./run.sh train configs/feature_mlp_coarse.yaml

# overrides (no new YAML needed):
./run.sh train configs/lgbm_coarse.yaml --set label_space=flat75 name=lgbm_flat75
./run.sh train configs/cnn_bigru_coarse.yaml --set data.max_per_class=null   # full 6.14M
```

## Sweep everything
```bash
./run.sh zoo-smoke     # tiny subset, 1-2 epochs — proves every model runs
./run.sh zoo-coarse    # full coarse-11 sweep of all models
```

## Phase-2 specialists (the old bottleneck)
```bash
./run.sh features  # ensure cache exists
./run.sh train configs/lgbm_specialist_text.yaml
./run.sh train configs/lgbm_specialist_archive.yaml
# any group: --set label_space=specialist:audio name=lgbm_spec_audio
```

## Stacking ensemble (orthogonal models → past 90%)
```bash
# train base models with IDENTICAL label_space + caps + seed so val/test rows align, then:
./run.sh stack --runs runs/lgbm_coarse_XXXX runs/cnn_bigru_coarse_YYYY runs/mamba_coarse_ZZZZ \
               --label-space coarse11 --meta logreg --name stack_coarse
# --meta lgbm | mean  also available
```

## Inspect results
```bash
./run.sh eval runs/lgbm_coarse_XXXX test      # accuracy + worst classes from JSON
./run.sh board                                # TensorBoard on runs/ (the "sliders") -> :6006
```
Each `runs/<name>_<ts>/` holds `training_log.json`, TensorBoard events,
`ckpt/epoch_*.pt` (+`best.pt`), `metrics_{val,test}.json`, and
`{val,test}_probs.npy` for stacking and offline graphing.

## NCD features (optional, slow)
```bash
./run.sh features stats,hist,ncd            # lzma is ~ms/fragment; be patient
./run.sh train configs/ncd_lgbm_coarse.yaml
```

## Notes
- Subsample caps in configs keep runs tractable on the 6750 XT; set
  `data.max_per_class=null` for full data once a model looks promising.
- Mamba & GNN are pure-PyTorch (ROCm-safe); no `mamba-ssm`/`torch-geometric`.
- `flat75` runs also report the coarse-11 collapse, so one run gives both numbers.
