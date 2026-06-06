# Phase 2 — per-group leaf specialists

Each of the 11 coarse groups gets a **specialist** that classifies among only that
group's fine leaf types (the `specialist:<group>` label space). Reuses the whole
`remake/` zoo — no new models, just per-group configs + orchestration.

```
phase2/
├── gen_configs.py   # write per-group configs (lgbm always; transformer/tcn optional, warm-started)
├── train_all.sh     # train one model family across groups, then summarise
├── summarize.py     # isolated per-group accuracy table (assumes perfect routing)
└── configs/         # generated per-group configs
```

## 1. LightGBM specialists (all 11 groups) — start here
```bash
python3 phase2/gen_configs.py          # writes configs/spec_<group>_lgbm.yaml
bash    phase2/train_all.sh lgbm       # trains all 11 (CPU, minutes each), prints a table
# subset:  bash phase2/train_all.sh lgbm raw archive
```

## 2. Transformer / TCN specialists (afterwards, warm-started from phase-1)
Point at a finished phase-1 run of the **same architecture** (its byte encoder
transfers; the head is reinitialised):
```bash
python3 phase2/gen_configs.py \
    --transformer-from runs/transformer_large_XXXX \
    --tcn-from         runs/tcn_full_XXXX
bash phase2/train_all.sh transformer raw archive   # GPU; warm-start = fast convergence
bash phase2/train_all.sh tcn        raw archive
```
Warm-start relies on the `init_from` shape-tolerant loader in the trainer (loads
matching encoder weights, skips the mismatched head). If no `--*-from` is given,
NN specialists are written to train fresh.

## 3. Read results
`train_all.sh` calls `summarize.py` at the end; rerun it anytime:
```bash
python3 phase2/summarize.py --model lgbm
```
Each specialist run also prints its own leaf confusion matrix and saves the usual
`runs/spec_<group>_<model>_<ts>/` (training_log.json, confusion_test.txt,
model.pt / ckpt, val+test probs).

## Notes
- **Isolated accuracy** here assumes perfect phase-1 routing — it's the ceiling.
  The end-to-end cascade (phase-1 routing → specialist) is a later add (the real
  headline number, like the old project's 79.4%).
- `raw` (11 camera RAW formats) and `archive` (13 compressed formats) are the
  hard groups; `gen_configs.py` gives them extra capacity / epochs.
