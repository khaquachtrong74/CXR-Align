# CXR-Align

Multimodal image–text alignment on chest X-rays (MIMIC-CXR): a two-tower contrastive
retriever trained from **frozen, pre-trained encoders**.

This was the modelling core of my undergraduate thesis project. The thesis conclusion — that
aligning frozen modality-specific encoders from scratch is far weaker than using a
natively multimodal pre-trained model — is the empirical basis for a later decision, so the
code is kept public.

---

## What this is (and what it is not)

**Is:** a full, runnable pipeline for contrastive image–report retrieval on MIMIC-CXR.
Pre-compute frozen embeddings once, then train only the lightweight alignment heads on top.
Pre-computing is what makes the whole thing fit on free-tier GPUs.

**Is not:** a new foundation model, and not a diagnostic system. Nothing here produces
clinical output, and nothing here should be used for diagnosis.

---

## Architecture

```
                       ┌──────────────────────────────┐
 X-ray image ────────► │ ResNet-50 (torchxrayvision,  │──► 2048-d ─┐
 (≤ 2 views / study)   │ CheXpert weights, FROZEN)    │           │
                       └──────────────────────────────┘           │
                                                                  ├─► View Attention
                                                                  │   Pooling (max 2 views)
                       ┌──────────────────────────────┐           │
 Report text ────────► │ Bio_ClinicalBERT (FROZEN)    │──► 768-d ─┘
 (clinical + augmented)└──────────────────────────────┘
                                                                  │
                                            ┌─────────────────────┴─────────────────────┐
                                            │  Projection → 256-d  (h)                  │
                                            │  Contrastive head    (z, MLP)             │
                                            └───────────────────────────────────────────┘
                                                     │                      │
                                            InfoNCE on z            retrieval uses h
```

Two design points worth naming:

- **`h` vs `z` are separate.** `h` is the output of the modality-specific adapters; `z` is the
  contrastive head output built on top of `h`. The loss is computed on `z`, so the projection
  head's geometry does not have to be the retrieval geometry. ⚠️ **In the current code the
  epoch-level retrieval logging still measures `z`, not `h`** (`_log_retrieval` in
  `src/train.py`), while `configs/model/projection.yaml` states retrieval uses `h`. The config
  comment and the implementation disagree — fix before quoting any retrieval number as
  "retrieval performance".
- **View attention pooling.** A study holds 1–2 views; a learned attention pool reduces them
  to one embedding per study, with masking so padded views never contribute.

### Loss

`GlobalInfoNCE` — symmetric cross-entropy over a scaled similarity matrix, with a
**learnable temperature** (`α` term), clamped to `[0.01, 0.1]` for stability:

```
logits = z_img @ z_txt.T / clamp(exp(log_temp), min_temp, max_temp)
loss   = 0.5 * (CE(logits, labels) + CE(logits.T, labels))
```

The temperature is configured as `shared` or `separate` (independent values for the
cross-modal and intra-modal terms) — see `configs/model/loss.yaml`.

---

## Data

MIMIC-CXR, split **by subject** so no patient appears on both sides of a split:

| Split | Studies | 1 view | 2 views | 3+ views (capped at 2) |
|---|---:|---:|---:|---:|
| Train | 126,536 | 57,201 | 57,403 | 11,932 |
| Val | 15,778 | 6,971 | 7,188 | 1,619 |
| Test | 15,363 | 6,640 | 7,316 | 1,407 |
| **Total** | **157,677** | **70,812** | **71,907** | **14,958** |

Reports are used in two forms per study (`text` and `text_augment`). Studies with 3+ views are
capped at 2 — the documented ceiling of this pipeline.

MIMIC-CXR is **credentialed** data: it requires PhysioNet training and a signed data use
agreement, and it may not be redistributed. This repository contains **no data**. Point
`configs/data/paths_*.yaml` at your own copy.

---

## Layout

```
configs/          Hydra config tree (data / model / train)
src/
  dataset.py              cached-embedding study dataset + collate
  model.py                pooling, projection heads, GlobalInfoNCE, encoder builders
  precompute_embeddings.py  frozen-encoder pass → on-disk cache
  train.py                MedAlignModel + Trainer (AMP, DataParallel, periodic retrieval eval)
  split.py, utils.py      subject-level splitting helpers
  visualize_utils.py      retrieval metrics (R@k, median rank) + embedding/heatmap logging
  finalize_cache.py       cache manifest finalisation
scripts/          setup_env.sh · precompute.sh · precompute_parallel.sh · train.sh
tests/            16 test functions: tensor shapes, variable view counts, temperature
                  clamping, independent gradients under separate temperatures, DataParallel,
                  batch independence
tracking.md       embedding-cache size estimation
```

---

## Running it

```bash
bash scripts/setup_env.sh                  # uv-based venv
# edit configs/data/paths_local.yaml  → point at your MIMIC-CXR + metadata
bash scripts/precompute.sh                 # frozen encoders → embedding cache

# Metadata must expose: subject_id, study_id, image paths, views, text, text_augment.
# The exact CSV schema is the one contract this repo assumes — see src/dataset.py.
python -m pytest tests/ -q
bash scripts/train.sh
```

Embedding cache for the full dataset is **≈3.0 GB** uncompressed (2048-d images, 768-d text,
two text vectors per study) — the per-split arithmetic is in `tracking.md`.

`precompute_parallel.sh` shards the cache build across GPUs for Kaggle's 2× T4 sessions.

---

## Compute

Trained on **free-tier Kaggle**: 2× NVIDIA T4 (16 GB each), with AMP and `DataParallel`.
Pre-computing frozen embeddings is the reason this fits — the gradient-bearing part is only
the pooling, projection, and contrastive heads.

---

## Results

Headline numbers (I2T and T2I `Recall@k` on the subject-held-out test split) are reported in
the thesis write-up, not in this repository. The repository holds code and tests only — no
result artifacts, no checkpoints, no logs.

> The thesis' central finding concerns the approach itself: training an alignment on top of
> frozen modality-specific encoders (ResNet-50 + Bio_ClinicalBERT) lands far below a natively
> multimodal pre-trained model on the same retrieval task. That measurement is why the
> follow-up work starts from a pre-trained medical VLM instead of training alignment from
> scratch. See the thesis for the numbers and the caveats attached to them.

---

## Notes / limitations

- **2D only.** No 3D CT/MRI.
- **≤ 2 views per study** by construction; the pooling mask is built for this ceiling.
- **Retrieval metrics are computed on a hard-coded 1,000-sample cap**
  (`max_samples = 1000` inside `_log_retrieval`), against a validation split of 15,778 studies.
  Any `R@k` produced by training-time logging is therefore a **1,000-sample** number and must
  be quoted with that qualifier attached. It also runs on `z`, not `h` (see above).
- Reports are English; the encoders are English. No translation layer here.

## References

- MIMIC-CXR — Johnson et al., *Scientific Data* (2019)
- CheXpert / torchxrayvision — Irvin et al. (2019); Cohen et al.
- Bio_ClinicalBERT — Alsentzer et al. (2019)
- InfoNCE / CLIP-style contrastive objective — van den Oord et al. (2018); Radford et al. (2021)

## License

Code released under the MIT License. Data is **not** included and remains governed by the
PhysioNet MIMIC-CXR data use agreement.
