#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load config from .env
[ -f "$PROJECT_ROOT/.env" ] && source "$PROJECT_ROOT/.env"

PYTHON="${PYTHON:-$VENV_DIR/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python"

SPLIT="${SPLIT:-train}"
GPU_INDEX="${GPU_INDEX:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
SKIP_FINALIZE="${SKIP_FINALIZE:-0}"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/precompute_${SPLIT}_gpu${GPU_INDEX}.log"

echo "============================================================"
echo "medalign-pipeline: Precompute Embeddings"
echo "============================================================"
echo "Split: $SPLIT | GPU: $GPU_INDEX / $NUM_GPUS | Cache: $CACHE_DIR"
echo "ImgBS: $IMG_BATCH_SIZE | TxtBS: $TXT_BATCH_SIZE | SubjBS: $SUBJECT_BATCH_SIZE | Workers: $NUM_WORKERS"

exec > >(tee -a "$LOG_FILE") 2>&1

"$PYTHON" src/precompute_embeddings.py \
  --data-root "$DATA_ROOT" \
  --image-root "$IMAGE_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --split "$SPLIT" \
  --gpu-index "$GPU_INDEX" \
  --num-gpus "$NUM_GPUS" \
  --img-batch-size "$IMG_BATCH_SIZE" \
  --txt-batch-size "$TXT_BATCH_SIZE" \
  --subject-batch-size "$SUBJECT_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-views-per-study "$MAX_VIEWS" \
  --checkpoint-path "/kaggle/temp/checkpoint/pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt"

if [ "$SKIP_FINALIZE" -eq 0 ] && [ "$NUM_GPUS" -eq 1 ]; then
  "$PYTHON" src/finalize_cache.py --cache-dir "$CACHE_DIR" --split "$SPLIT" --expected-shards 1
fi

echo "Precompute done for split=$SPLIT on GPU $GPU_INDEX"