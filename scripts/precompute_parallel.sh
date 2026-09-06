#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load config from .env
[ -f "$PROJECT_ROOT/.env" ] && source "$PROJECT_ROOT/.env"

SPLIT="${1:-train}"
NUM_GPUS="${2:-2}"

export CACHE_DIR VENV_DIR PYTHON DATA_ROOT IMAGE_ROOT \
       IMG_BATCH_SIZE TXT_BATCH_SIZE SUBJECT_BATCH_SIZE \
       NUM_WORKERS MAX_VIEWS SPLIT NUM_GPUS

echo "============================================================"
echo "medalign-pipeline: Precompute Embeddings (Parallel)"
echo "============================================================"
echo "Split: $SPLIT | GPUs: $NUM_GPUS | Cache: $CACHE_DIR"

mkdir -p "$CACHE_DIR/logs"

PIDS=()
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching GPU $gpu_idx / $NUM_GPUS..."
    GPU_INDEX=$gpu_idx SKIP_FINALIZE=1 \
    bash "$SCRIPT_DIR/precompute.sh" \
        >"$CACHE_DIR/logs/precompute_${SPLIT}_gpu${gpu_idx}.log" 2>&1 &
    PIDS+=("$!")
done

echo "All launched. Waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "GPU $i finished."
    else
        echo "GPU $i failed. See $CACHE_DIR/logs/precompute_${SPLIT}_gpu${i}.log" >&2
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi

# Finalize once all shards done
"$PYTHON" src/finalize_cache.py --cache-dir "$CACHE_DIR" --split "$SPLIT" --expected-shards "$NUM_GPUS"

echo "============================================================"
echo "Precompute done for split=$SPLIT on all $NUM_GPUS GPUs"
echo "============================================================"