# Remake — FFT-75 File-Fragment Classifier, Design & Model Catalogue

A modular harness for trying many architectures against FFT-75 **Scenario #1,
4096-byte** fragments. Every model is a plugin behind one shared data /
feature / training / logging / evaluation pipeline, so we can train any of them
with one command and graph everything afterwards.

---

## 0. The bug that motivated the rebuild (read this first)

The previous project plateaued at **exactly 87.8%** coarse accuracy with `text`
and `archive` bleeding into each other (44.7% bidirectional confusion). That was
**not** an architecture problem — it was a label-corruption bug in
`convert_npz_to_binary.py`:

1. The FiFTy NPZ stores integer labels in FiFTy's **official 75-class order**
   (`tags.txt`). The converter assumed they matched a hand-written 58-class
   order, so it **permuted every label** — e.g. FiFTy `30:DMG`→"txt",
   `36:RAR`→"zip", `55:MD`→"ps".
2. FiFTy classes 58–74 (`TEX, JSON, HTML, XML, LOG, CSV, AIFF, FLAC, M4A, MP3,
   OGG, WAV, WMA, PCAP, TTF, DWG, SQLITE`) were `>= 58` and got **collapsed into
   class 0** ("jpg"), which ended up with 17.8× the samples of any other class.

Net effect: the real human-readable text formats vanished into "jpg", and what
was labelled `text` was actually compressed archives — so the model was asked to
separate two indistinguishable piles of high-entropy bytes (→ the 50/50,
44.7%-confusion, 87.8%-ceiling signature).

**Proof.** Per-class content fingerprints matched FiFTy's order exactly (idx 30
entropy 7.94 = DMG; idx 55 printable 0.97 = MD; idx 14 all-`0xFF` = BMP white
runs; …). The fragment bytes were never corrupted and stayed in NPZ row order
(`bin_label[i] == buggy_remap(npz_y[i])` for all 768k val rows, fragment bytes
identical). So the fix is just to rewrite labels from the NPZ `y` and keep the
fragments (`scripts/relabel_from_npz.py`).

After the fix, an entropy threshold alone nearly separates text (4.3–5.4 b/B,
80–100% printable) from archive (7.85–7.95 b/B, 37% printable). The bleed is gone.

---

## 1. Dataset & taxonomy

- **4096-byte** fragments, **75 leaf classes** in FiFTy's canonical order
  (`tags.txt`), 81 920 train / 10 240 val / 10 240 test per class
  (6.14M / 768k / 768k total).
- Stored as raw `uint8` binary (`*_fragments.bin` + `*_labels.bin` + meta),
  fragments symlinked from the verified-correct bytes; labels rewritten from the
  NPZ source of truth.
- `remake/taxonomy.py` is the single source of truth and defines three **label
  spaces**:
  - `flat75` — the 75 leaves (also reports the coarse-11 collapse automatically);
  - `coarse11` — FiFTy's official **11 tag groups** (Raw, Bitmap, Vector, Video,
    Archive, Executable, Office, Published, **Text**, Audio, Other) — the
    byte-sensible Phase-1 grouping (Scenario #2);
  - `specialist:<group>` — within-group fine classification (Phase 2).

On grouping: we **start** with FiFTy's official 11 tags (per your decision) and
will revisit with the corrected confusion matrix — if a merge/split clearly
helps cascade separability, `taxonomy.py` is the one place to change it.

---

## 2. Features (`remake/features.py`)

Position-free statistics, cached once to disk and reused by all feature models.

| group | dim | contents |
|---|---|---|
| `stats` | ~60 | Shannon entropy, **zlib compression ratio**, byte-class ratios (printable / null / high / whitespace / alpha / digit / control), mean/std, distinct-byte ratio, peak freq, mean |Δ|, lag-1 autocorrelation, null-run, ~24 structural-char densities (`{}[]<>/=&"',;|…`), ~10 structural bigrams (`</`, `="`, `,\n`, …) |
| `hist` | 256 | normalised byte histogram (compression-algorithm fingerprint) |
| `ncd` | 5 | multi-compressor ratios (zlib/bz2/lzma) + deltas — **slow (lzma)**, opt-in |

Default tree/feature set: `stats_hist` (~316 dims). Cache lives in
`data/4k_1/features/<split>.<group>.npy`.

---

## 3. Model catalogue

Common interface (`remake/registry.py`): each model declares `kind`
(`nn`/`tree`) and `input` (`bytes`/`features`). NN models return logits `(B, C)`;
tree models expose `.fit/.predict_proba`. **ROCm note:** the SSM and GNN are
pure-PyTorch (no CUDA-only `mamba-ssm`/`torch-geometric` kernels), so they run on
the RX 6750 XT.

### Tree models — `input=features`, CPU, fastest
- **`lgbm` (LightGBM)** — gradient-boosted histograms. The expected front-runner:
  entropy + the histogram fingerprint separate the archive cluster almost
  linearly, and GBDTs mix those features with no GPU. Logs the per-boost
  validation curve.
- **`xgb` (XGBoost)** — second GBDT for ensemble diversity.
- **`rf` / `extratrees`** — bagged-tree baselines (sklearn).
- **`catboost`** — optional GBDT (if installed).
- **NCD model** — `lgbm` on `stats_ncd` (`configs/ncd_lgbm_coarse.yaml`); the
  normalized-compression-distance view of compression families.

### Sequence NNs — `input=bytes`, GPU
- **`cnn_bigru`** — multi-scale byte CNN (k=9,27) + 2-layer BiGRU + attention
  pool. The **reference** (the architecture that hit 87.8%), re-measured on
  *correct* labels.
- **`tcn`** — dilated temporal conv net; large receptive field via dilation
  (1→128), no recurrence, fully parallel (fast on ROCm).
- **`transformer`** — strided-conv stem downsamples 4096→256 tokens, then a small
  pre-norm Transformer encoder. The "transformer-on-bytes" line, kept small.
- **`mamba`** — pure-PyTorch selective SSM (S6): input-dependent Δ/B/C, sequential
  scan over the 256 downsampled tokens, linear-time. Alternative to the BiGRU.

### Graph — `input=bytes`, GPU
- **`gnn`** — each fragment is a 256-node byte graph; edges are the per-fragment
  byte-transition (bigram) matrix. Dense message passing (forward+reverse edges)
  with shared learnable byte embeddings, mass-weighted readout. Captures
  transition structure independent of absolute position; complementary to the
  sequence view.

### Hybrid / meta
- **`feature_mlp`** — deep MLP on the engineered features; the differentiable
  twin of the trees, ensembles well, trains in seconds/epoch.
- **Stacking (`remake/stacking.py`)** — fit a meta-learner (logreg / lgbm) on the
  base models' **validation** probabilities, evaluate on test. Because tree and
  byte models make partly orthogonal errors, the blend is the most likely route
  past 90%. A no-fit `mean` blend is also supported.

---

## 4. Training, logging, evaluation

- **NN trainer** (`trainer.py`): AdamW (no decay on norms/embeds/bias), cosine LR
  with warmup, AMP, grad-clip, optional grad-accum / EMA / GBFlip augmentation,
  early stopping.
- **Tree trainer** (`tree_trainer.py`): fit on cached features with an eval set;
  logs the boosting curve and top feature importances.
- **"Save everything, graph later"** (`logging_utils.py`): every run gets
  `runs/<name>_<ts>/` with
  - `training_log.json` — config + per-epoch records + best + final metrics
    (rewritten every epoch, crash-safe);
  - **TensorBoard** scalars — the live "sliders" (loss/acc/lr/grad-norm/epoch-
    time/GPU-mem; boosting curve for trees);
  - `ckpt/epoch_XXXX.pt` **every epoch** + `best.pt` symlink ("every generation");
  - `metrics_{val,test}.json` — overall + per-class recall/precision + full
    confusion matrix (+ coarse-11 collapse for flat75);
  - `{val,test}_probs.npy` + `{val,test}_true.npy` — for stacking & offline plots.
- **Metrics** (`metrics.py`): accuracy, macro-recall, per-class, confusion, and
  the coarse collapse so one flat-75 run yields both the fine and group numbers.

---

## 5. Targets & plan

Target: **≥90% on both phases.** Phase 1 = coarse-11; Phase 2 = specialists,
with `text` and `archive` (the old bottleneck) the ones to watch. Sequencing:

1. **Now:** corrected data + feature cache.
2. Coarse-11 sweep of every model (cheap trees first), pick winners by val acc.
3. Stack the top orthogonal models → coarse-11 ≥90% check.
4. Build Phase-2 specialists (start with `text`, `archive`), stack there too.
5. Revisit the grouping against the corrected confusion matrix if useful.
