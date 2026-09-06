# Precompute Embedding Storage Estimation

## Dataset Statistics (from splits/summary/)

| Split | Studies | 1 view | 2 views | 3+ views (capped at 2) |
|-------|---------|--------|---------|------------------------|
| Train | 126,536 | 57,201 | 57,403  | 11,932 |
| Val   | 15,778  | 6,971  | 7,188   | 1,619 |
| Test  | 15,363  | 6,640  | 7,316   | 1,407 |
| **Total** | **157,677** | **70,812** | **71,907** | **14,958** |

## Embedding Specifications

- **Image (ResNet50):** 2048 dim, float32 (4 bytes), max 2 views/study
- **Text (BioClinicalBERT):** 768 dim, float32 (4 bytes), 2 vectors (clinical + augment)

## Storage Calculation (NPZ uncompressed)

### Image embeddings
- 1-view studies: 70,812 × 1 × 2048 × 4 bytes = **580 MB**
- 2-view studies: 71,907 × 2 × 2048 × 4 bytes = **1,178 MB**
- 3+ view studies (capped): 14,958 × 2 × 2048 × 4 bytes = **245 MB**
- **Total image:** ~2,003 MB

### Text embeddings
- All studies: 157,677 × 2 × 768 × 4 bytes = **969 MB**

### Overhead
- NPZ headers: ~200 bytes × 2 files × 157,677 = **63 MB**
- Manifests + metadata JSON: **~10 MB**

### Grand Total (uncompressed)
**~3.0 GB**

## Compressed Estimate
- NPZ uses `np.savez` (no compression). If switched to `np.savez_compressed`: ~40-50% reduction
- Full cache zip/tar.gz: **~1.5-1.8 GB**

## Recommendation
Change `_save_npz_atomic` in `src/dataset.py` to use `np.savez_compressed` for ~40-50% disk savings per file.