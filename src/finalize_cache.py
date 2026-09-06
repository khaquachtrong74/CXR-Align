import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import finalize_split_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge precompute shard manifests into a training cache manifest."
    )
    parser.add_argument("--cache-dir", required=True, help="Embedding cache directory")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=1,
        help="Number of completed precompute shards to merge",
    )
    args = parser.parse_args()

    stats = finalize_split_cache(
        cache_dir=Path(args.cache_dir),
        split=args.split,
        expected_shards=args.expected_shards,
    )
    print(
        f"Finalized {args.split}: {stats['total_studies']} studies, "
        f"{stats['cache_size_mb']:.1f} MB"
    )


if __name__ == "__main__":
    main()
