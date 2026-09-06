import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import Counter


def parse_list_field(value: Union[str, List]) -> List:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def load_metadata(metadata_file: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(metadata_file)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        row['image_paths'] = parse_list_field(row['image_paths'])
        row['views'] = parse_list_field(row['views'])
        row['num_images'] = int(row['num_images'])

    return rows


def select_views_deterministic(
    image_paths: List[str],
    views: List[str],
    max_views_per_study: int = 2,
) -> tuple[List[str], List[str]]:
    view_priority = {"PA": 0, "AP": 1, "LATERAL": 2, "LL": 3, "UNKNOWN": 99}
    view_indices = list(range(len(image_paths)))
    view_indices.sort(key=lambda i: view_priority.get(views[i], 99))
    selected_indices = view_indices[:max_views_per_study]
    selected_paths = [image_paths[i] for i in selected_indices]
    selected_views = [views[i] for i in selected_indices]
    return selected_paths, selected_views


def create_splits_by_subject(
    metadata: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    subject_ids = list(set(row['subject_id'] for row in metadata))
    random.seed(seed)
    random.shuffle(subject_ids)

    n_total = len(subject_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_subjects = set(subject_ids[:n_train])
    val_subjects = set(subject_ids[n_train:n_train + n_val])
    test_subjects = set(subject_ids[n_train + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for row in metadata:
        sid = row['subject_id']
        if sid in train_subjects:
            splits["train"].append(row)
        elif sid in val_subjects:
            splits["val"].append(row)
        else:
            splits["test"].append(row)

    return splits


def save_splits(
    splits: Dict[str, List[Dict[str, Any]]],
    splits_dir: Union[str, Path],
    metadata_fields: List[str],
) -> None:
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in splits.items():
        out_path = splits_path / f"{split_name}.csv"
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metadata_fields)
            writer.writeheader()
            for row in split_rows:
                row_copy = row.copy()
                row_copy['image_paths'] = str(row_copy['image_paths'])
                row_copy['views'] = str(row_copy['views'])
                writer.writerow(row_copy)

        subj_path = splits_path / f"{split_name}_subjects.json"
        subjects = sorted(set(row['subject_id'] for row in split_rows))
        with open(subj_path, 'w') as f:
            json.dump(subjects, f)

        n_subjects = len(subjects)
        n_studies = len(split_rows)
        total_studies = sum(len(v) for v in splits.values())
        print(f"  {split_name}: {n_subjects} subjects, {n_studies} studies ({n_studies/total_studies*100:.1f}%)")


def get_split_stats(
    metadata: List[Dict[str, Any]],
    splits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    stats = {}
    total_studies = len(metadata)
    total_subjects = len(set(row['subject_id'] for row in metadata))

    for split_name, split_rows in splits.items():
        subjects = set(row['subject_id'] for row in split_rows)
        studies = len(split_rows)
        views = Counter()
        for row in split_rows:
            views.update(row['views'])

        stats[split_name] = {
            "subjects": len(subjects),
            "studies": studies,
            "studies_pct": studies / total_studies * 100,
            "views": dict(views),
        }

    stats["total"] = {"subjects": total_subjects, "studies": total_studies}
    return stats


def filter_studies_by_split(
    all_studies: List[Dict[str, Any]],
    split_file: Union[str, Path],
) -> List[Dict[str, Any]]:
    split_path = Path(split_file)
    if not split_path.exists():
        return all_studies

    split_studies = load_metadata(split_path)
    split_subject_ids = set(s["subject_id"] for s in split_studies)
    return [s for s in all_studies if s["subject_id"] in split_subject_ids]