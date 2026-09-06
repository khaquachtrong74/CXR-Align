import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import save_study_cache, write_manifest
from src.model import (
    build_resnet50_xrv,
    build_bioclinicalbert_encoder,
    get_resnet50_transform,
)
from src.split import load_metadata, select_views_deterministic, filter_studies_by_split


class PrecomputeImageDataset(Dataset):
    def __init__(
        self,
        studies: List[Dict[str, Any]],
        image_root: Path,
        image_transform,
        max_views_per_study: int = 2,
    ):
        self.studies = studies
        self.image_root = image_root
        self.image_transform = image_transform
        self.max_views_per_study = max_views_per_study

        self.image_items = []
        self.study_meta = {}

        for study_idx, study in enumerate(studies):
            subject_id = study["subject_id"]
            study_id = study["study_id"]
            image_paths = study["image_paths"]
            views = study["views"]

            selected_paths, selected_views = select_views_deterministic(
                image_paths, views, max_views_per_study
            )

            if not selected_paths:
                continue

            study_key = (subject_id, study_id)
            for idx, img_path in enumerate(selected_paths):
                self.image_items.append({
                    "path": self.image_root / img_path,
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "view_idx": idx,
                })

            self.study_meta[study_key] = {
                "subject_id": subject_id,
                "study_id": study_id,
                "num_views": len(selected_paths),
                "view_ids": selected_views,
                "clinical_text": study.get("text", ""),
                "augment_text": study.get("text_augment", ""),
            }

    def __len__(self) -> int:
        return len(self.image_items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.image_items[idx]
        try:
            img = Image.open(item["path"]).convert("L")
            img = self.image_transform(img)
            if img.shape[0] != 1:
                img = img.mean(dim=0, keepdim=True)
        except Exception as e:
            print(f"Warning: Failed to load {item['path']}: {e}")
            img = torch.zeros(1, 512, 512, dtype=torch.float32)

        return {
            "image": img,
            "subject_id": item["subject_id"],
            "study_id": item["study_id"],
            "view_idx": item["view_idx"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "subject_ids": [b["subject_id"] for b in batch],
        "study_ids": [b["study_id"] for b in batch],
        "view_indices": [b["view_idx"] for b in batch],
    }


def compute_studies(
    studies: List[Dict[str, Any]],
    image_encoder: nn.Module,
    text_encoder: nn.Module,
    tokenizer,
    image_transform,
    image_root: Path,
    device: torch.device,
    max_views_per_study: int = 2,
    img_batch_size: int = 256,
    txt_batch_size: int = 512,
    num_workers: int = 4,
    cache_dir: Path = None,
    split: str = "train",
) -> int:

    dataset = PrecomputeImageDataset(
        studies,
        image_root,
        image_transform,
        max_views_per_study,
    )

    if len(dataset) == 0:
        return 0

    loader = DataLoader(
        dataset,
        batch_size=img_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )

    print(
        f"  Encoding {len(dataset.study_meta)} studies ({len(dataset)} images) "
        f"with {num_workers} workers..."
    )
    image_encoder.eval()
    study_features = {}
    processed = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="Images", mininterval=1.0, disable=not sys.stderr.isatty())
        start_time = time.time()
        for batch in progress:
            images = batch["images"].to(device, non_blocking=True)
            subject_ids = batch["subject_ids"]
            study_ids = batch["study_ids"]
            view_indices = batch["view_indices"]

            img_global = image_encoder(images)
            img_global = img_global.cpu().numpy()

            for i, (sid, study_id, v_idx) in enumerate(
                zip(subject_ids, study_ids, view_indices)
            ):
                key = (sid, study_id)
                if key not in study_features:
                    n_views = dataset.study_meta[key]["num_views"]
                    study_features[key] = np.zeros(
                        (n_views, img_global.shape[1]), dtype=np.float32
                    )
                study_features[key][v_idx] = img_global[i]

            for key, feats in study_features.items():
                if not np.any(feats == 0):
                    meta = dataset.study_meta[key]
                    clinical_texts = [meta["clinical_text"]]
                    augment_texts = [meta["augment_text"]]

                    clinical_global = text_encoder.encode_texts(
                        clinical_texts, tokenizer, device, batch_size=txt_batch_size
                    )[0].numpy().astype(np.float32)
                    augment_global = text_encoder.encode_texts(
                        augment_texts, tokenizer, device, batch_size=txt_batch_size
                    )[0].numpy().astype(np.float32)

                    save_study_cache(
                        cache_dir, split,
                        meta["subject_id"], meta["study_id"],
                        feats, meta["view_ids"],
                        clinical_global, augment_global
                    )
                    processed += 1
                    del study_features[key]

            elapsed_minutes = max((time.time() - start_time) / 60, 1e-6)
            progress.set_postfix(processed=processed, rate=f"{processed / elapsed_minutes:.1f} studies/min")

    return processed


def process_shard(
    shard_studies: List[Dict],
    shard_idx: int,
    total_shards: int,
    args,
    device: torch.device,
):
    print(f"[Shard {shard_idx+1}/{total_shards}] Processing {len(shard_studies)} studies on {device}")

    image_encoder = build_resnet50_xrv(device=device)
    text_encoder, tokenizer = build_bioclinicalbert_encoder(
        model_name=args.text_model,
        max_length=args.max_text_length,
        device=device,
    )
    image_transform = get_resnet50_transform(
        input_size=args.input_size,
        mean=args.normalize_mean,
        std=args.normalize_std,
    )
    image_root = Path(args.image_root)
    cache_dir = Path(args.cache_dir)

    start_time = time.time()
    to_process = []
    skipped = 0
    for study in shard_studies:
        key = f"{study['subject_id']}_{study['study_id']}"
        img_file = cache_dir / "images" / args.split / f"{key}.npz"
        txt_file = cache_dir / "texts" / args.split / f"{key}.npz"
        if img_file.exists() and txt_file.exists():
            skipped += 1
        else:
            to_process.append(study)

    processed = compute_studies(
        to_process,
        image_encoder,
        text_encoder,
        tokenizer,
        image_transform,
        image_root,
        device,
        max_views_per_study=args.max_views_per_study,
        img_batch_size=args.img_batch_size,
        txt_batch_size=args.txt_batch_size,
        num_workers=args.num_workers,
        cache_dir=cache_dir,
        split=args.split,
    )
    write_manifest(cache_dir, args.split, shard_studies)

    elapsed = time.time() - start_time

    print(f"\n[Shard {shard_idx+1}/{total_shards}] Done in {elapsed:.1f}s")
    print(f"  Processed: {processed}, Skipped (cached): {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings for medalign-pipeline (ResNet50 + BioClinicalBERT)")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory with metadata CSV")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory for images")
    parser.add_argument("--cache-dir", type=str, required=True, help="Directory to save cache")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Data split to process")
    parser.add_argument("--text-model", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="Text encoder model name")
    parser.add_argument("--max-text-length", type=int, default=512, help="Max token length for text encoder")
    parser.add_argument("--input-size", type=int, default=512, help="Input image size for TorchXRayVision ResNet50 (native 512x512)")
    parser.add_argument("--normalize-mean", type=float, nargs='+', default=[0.5], help="Normalize mean for grayscale XRV preprocessing (single value)")
    parser.add_argument("--normalize-std", type=float, nargs='+', default=[0.5], help="Normalize std for grayscale XRV preprocessing (single value)")
    parser.add_argument("--max-views-per-study", type=int, default=2, help="Max views per study")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for this process (0 or 1)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Total number of GPUs (for sharding)")
    parser.add_argument("--img-batch-size", type=int, default=32, help="Batch size for image encoding; keep small on 14-16 GB GPUs")
    parser.add_argument("--txt-batch-size", type=int, default=512, help="Batch size for text encoding")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers for image loading; lower to reduce memory spikes")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to ResNet50 checkpoint (.pt)")

    args = parser.parse_args()

    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        if args.gpu_index >= num_available_gpus:
            raise ValueError(
                f"Invalid --gpu-index {args.gpu_index}: only {num_available_gpus} GPU(s) available (0-{num_available_gpus-1}). "
                f"Valid indices: {list(range(num_available_gpus))}"
            )
        if args.gpu_index >= args.num_gpus:
            raise ValueError(
                f"Invalid --gpu-index {args.gpu_index}: must be < --num-gpus {args.num_gpus}. "
                f"Use --gpu-index 0 to {args.num_gpus-1}"
            )

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)} ({torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB)")

    metadata_file = Path(args.data_root) / "metadata" / "version-with-views" / "mimic_cxr_clean_studies_with_views.csv"
    if not metadata_file.exists():
        metadata_file = Path(args.data_root) / "mimic_cxr_clean_studies_with_views.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    print(f"Loading metadata from {metadata_file}...")
    all_studies = load_metadata(metadata_file)

    split_file = Path(args.data_root) / "splits" / f"{args.split}.csv"
    all_studies = filter_studies_by_split(all_studies, split_file)
    print(f"Filtered to {len(all_studies)} studies for {args.split} split")

    total_studies = len(all_studies)
    print(f"Total studies in {args.split}: {total_studies}")

    if args.num_gpus > 1:
        shard_size = (total_studies + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_index * shard_size
        end = min(start + shard_size, total_studies)
        shard_studies = all_studies[start:end]
        print(f"Shard {args.gpu_index+1}/{args.num_gpus}: studies [{start}:{end}] ({len(shard_studies)} studies)")
    else:
        shard_studies = all_studies

    process_shard(shard_studies, args.gpu_index, args.num_gpus, args, device)

    print("\nPrecomputation complete!")
    print("To run on 2 T4s in parallel:")
    print(f"  GPU 0: python precompute_embeddings.py ... --gpu-index 0 --num-gpus 2")
    print(f"  GPU 1: python precompute_embeddings.py ... --gpu-index 1 --num-gpus 2")
    print("\nTo persist cache on Kaggle:")
    print("  1. Save /kaggle/working/cache as a Kaggle Dataset")
    print("  2. In next session, add the dataset and mount at /kaggle/input/cache")
    print("  3. Use --cache-dir /kaggle/input/cache")


if __name__ == "__main__":
    main()