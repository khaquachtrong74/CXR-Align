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
  │ ResNet-50, grayscale [B,1,H,W]│──► 2048-d ─┐
  │ torchxrayvision, FROZEN       │           │
  └──────────────────────────────┘           ├─► View Attention Pooling
  ┌──────────────────────────────┐           │     (masked, ≤ 2 views)
  │ Bio_ClinicalBERT, FROZEN      │──► 768-d ─┤
  │ clinical report  +  augmented │           │
  └──────────────────────────────┘           │
                                             │
              ┌──────────────────────────────┴──────────────────────────────┐
              │  ProjectionHeadWithNorm → 256-d                              │
              │    h_img · h_clin · h_aug      (retrieval-side adapters)      │
              │  ContrastiveProjectionHead → z (MLP, dropout 0.0)            │
              │    z_img · z_clin · z_aug      (loss-side heads, 3 separate)  │
              └─────────────────────────────────────────────────────────────┘
```

### Loss

Two `GlobalInfoNCE` instances with **independent learnable temperatures** — one for the
image↔text term, one for the text↔text term — combined into a weighted sum:

```python
L_text     = intra_loss_fn(z_clin, z_aug)          # clinical vs augmented report
L_img_clin = cross_loss_fn(z_img, z_clin)
L_img_aug  = cross_loss_fn(z_img, z_aug)
L_img_txt  = 0.5 * (L_img_clin + L_img_aug)
total_loss = 0.3 * L_text + 0.7 * L_img_txt
```

So the image branch is matched against **both** text variants, and the two text variants are
additionally pulled together. The weights `0.3 / 0.7` are **constructor defaults, not
config** — `configs/model/loss.yaml` only carries the temperatures. Changing the balance
means editing `src/train.py`.

`GlobalInfoNCE` is symmetric cross-entropy over a scaled similarity matrix:

```python
logits = z1 @ z2.T / clamp(exp(log_temp), min_temp, max_temp)
loss   = 0.5 * (CE(logits, labels) + CE(logits.T, labels))
```

The temperature is a learned parameter clamped to `[0.01, 0.1]`, so the model can sharpen or
soften the contrastive distribution but cannot diverge. Two instances means `cross` and
`intra` temperatures move independently — this is what `temperature_mode: separate` selects
in `configs/model/loss.yaml`.

### `h` vs `z`

Each branch has an adapter (`ProjectionHeadWithNorm` → 256-d `h`) and a contrastive head
(`ContrastiveProjectionHead` → `z`). The loss is computed on `z`.

> ⚠️ **Implementation inconsistency, verified in the code.** The
> config comment in `configs/model/projection.yaml` states *"Retrieval uses h, the output of
> the modality-specific adapters"*, but `_log_retrieval()` in `src/train.py` collects
> `z_img` / `z_txt` and runs `compute_retrieval_metrics` on those. **The config comment and
> the implementation disagree.** Quote nothing about retrieval geometry until this is
> resolved one way or the other.

## Data

MIMIC-CXR, split **by subject** so no patient appears on both sides of a split:

| Split | Studies | 1 view | 2 views | 3+ views (capped at 2) |
|---|---:|---:|---:|---:|
| Train | 126,536 | 57,201 | 57,403 | 11,932 |
| Val | 15,778 | 6,971 | 7,188 | 1,619 |
| Test | 15,363 | 6,640 | 7,316 | 1,407 |
| **Total** | **157,677** | **70,812** | **71,907** | **14,958** |

Two text variants per study are **required** by the loss, not optional: `text` and
`text_augment` become `clinical_global` and `augment_global`. A study with more than 2 views
is capped at 2 — the ceiling the pooling mask is built for.

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
python -m pytest tests/ -q
bash scripts/train.sh
```

### The metadata contract

The one thing you must match is the studies CSV named in `configs/data/paths_*.yaml`
(`metadata_file`). Per study it needs:

| Field | Used as |
|---|---|
| `subject_id` | split key — splits are built **by subject** — and identity in the batch |
| `study_id` | identity in the batch |
| image paths | resolved against `image_root` |
| views | chosen deterministically, capped at `max_views_per_study` |
| `text` | clinical report → `clinical_global` |
| `text_augment` | augmented report → `augment_global` |

`precompute_embeddings.py` then writes a cache that `CachedStudyDataset` reads:
`<split>_metadata.json` with a `studies[]` list carrying `img_file` / `txt_file`, over
`images/<split>/*.npz` (`global_feat`, `view_ids`) and `texts/<split>/*.npz`
(`clinical`, `augment`). Both text arrays must be present — the loss consumes both.

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
- torchxrayvision (weights `resnet50-res512-all`) — Cohen et al., *Scientific Data* (2022)
- Bio_ClinicalBERT — Alsentzer et al. (2019)
- InfoNCE / CLIP-style contrastive objective — van den Oord et al. (2018); Radford et al. (2021)

## License

Code released under the MIT License. Data is **not** included and remains governed by the
PhysioNet MIMIC-CXR data use agreement.
