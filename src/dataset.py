import json
import os
from datetime import datetime, timezone
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union


class CachedStudyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: Path,
        split: str,
        max_views_per_study: int = 2,
        metadata_file: Optional[Path] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.max_views_per_study = max_views_per_study

        self.image_dir = self.cache_dir / "images" / split
        self.text_dir = self.cache_dir / "texts" / split

        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
        self.studies = self.metadata["studies"]

        print(f"Loaded {len(self.studies)} studies from cache for {split}")


    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        study_meta = self.studies[idx]
        subject_id = study_meta["subject_id"]
        study_id = study_meta["study_id"]

        img_data = np.load(self.cache_dir / study_meta["img_file"], allow_pickle=True)
        txt_data = np.load(self.cache_dir / study_meta["txt_file"], allow_pickle=True)

        image_global = img_data["global_feat"]
        view_ids = img_data["view_ids"].tolist()
        clinical_global = txt_data["clinical"]
        augment_global = txt_data["augment"]

        V_actual = image_global.shape[0]
        V_pad = self.max_views_per_study
        D_img = image_global.shape[-1]
        D_txt = clinical_global.shape[-1]

        image_global_padded = np.zeros((V_pad, D_img), dtype=np.float32)
        view_mask = np.zeros(V_pad, dtype=bool)

        v_fill = min(V_actual, V_pad)
        image_global_padded[:v_fill] = image_global[:v_fill]
        view_mask[:v_fill] = True

        return {
            "image_global": torch.from_numpy(image_global_padded),
            "view_mask": torch.from_numpy(view_mask),
            "clinical_global": torch.from_numpy(clinical_global),
            "augment_global": torch.from_numpy(augment_global),
            "subject_id": subject_id,
            "study_id": study_id,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "image_global": torch.stack([b["image_global"] for b in batch], dim=0),
        "view_mask": torch.stack([b["view_mask"] for b in batch], dim=0),
        "clinical_global": torch.stack([b["clinical_global"] for b in batch], dim=0),
        "augment_global": torch.stack([b["augment_global"] for b in batch], dim=0),
        "subject_ids": [b["subject_id"] for b in batch],
        "study_ids": [b["study_id"] for b in batch],
    }


def save_study_cache(
    cache_dir: Path,
    split: str,
    subject_id: str,
    study_id: str,
    image_global: np.ndarray,
    view_ids: List[str],
    clinical_global: np.ndarray,
    augment_global: np.ndarray,
) -> None:
    image_dir = cache_dir / "images" / split
    text_dir = cache_dir / "texts" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    key = f"{subject_id}_{study_id}"
    np.savez(image_dir / f"{key}.npz",
             global_feat=image_global.astype(np.float32),
             view_ids=np.array(view_ids, dtype=object))
    np.savez(text_dir / f"{key}.npz",
             clinical=clinical_global.astype(np.float32),
             augment=augment_global.astype(np.float32))


def write_manifest(
    cache_dir: Path,
    split: str,
    studies: List[Dict[str, Any]],
) -> None:
    cache_dir = Path(cache_dir)
    image_dir = cache_dir / "images" / split
    text_dir = cache_dir / "texts" / split

    entries = []
    for study in studies:
        subject_id = study["subject_id"]
        study_id = study["study_id"]
        key = f"{subject_id}_{study_id}"
        if not (image_dir / f"{key}.npz").exists() or not (text_dir / f"{key}.npz").exists():
            continue
        entries.append({
            "subject_id": subject_id,
            "study_id": study_id,
            "img_file": f"images/{split}/{key}.npz",
            "txt_file": f"texts/{split}/{key}.npz",
        })

    metadata = {
        "studies": entries,
        "total_studies": len(entries),
        "cache_size_bytes": sum(
            (cache_dir / e["img_file"]).stat().st_size + (cache_dir / e["txt_file"]).stat().st_size
            for e in entries
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_dir / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2))