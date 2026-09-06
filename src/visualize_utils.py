import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


def _import_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


def _import_dim_reduction():
    from sklearn.manifold import TSNE
    from umap import UMAP
    return TSNE, UMAP


def _import_pandas():
    import pandas as pd
    return pd


class Visualizer:
    def __init__(
        self,
        log_dir: Union[str, Path],
        use_tensorboard: bool = True,
        plot_dir: Optional[Union[str, Path]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.plot_dir = Path(plot_dir) if plot_dir else self.log_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        self.step = 0
        self.epoch = 0
        self.gpu_memory_log = []

        print(f"Visualizer initialized: log_dir={self.log_dir}, tensorboard={self.use_tensorboard}")

    def set_step(self, step: int, epoch: int = 0):
        self.step = step
        self.epoch = epoch

    def log_loss(
        self,
        loss_dict: Dict[str, float],
        phase: str = "train",
    ):
        prefix = f"{phase}/"

        if self.use_tensorboard and self.writer:
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"{prefix}{k}", v, self.step)

        log_entry = {
            "step": self.step,
            "epoch": self.epoch,
            "phase": phase,
            **loss_dict,
        }
        with open(self.log_dir / "loss_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_learning_rate(self, lr: float, group_name: str = "default"):
        if self.use_tensorboard and self.writer:
            self.writer.add_scalar(f"lr/{group_name}", lr, self.step)

    def log_temperature(self, temp: float):
        if self.use_tensorboard and self.writer:
            self.writer.add_scalar("temperature", temp, self.step)

    def log_retrieval_metrics(
        self,
        metrics: Dict[str, float],
        phase: str = "val",
    ):
        prefix = f"{phase}/retrieval/"

        if self.use_tensorboard and self.writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"{prefix}{k}", v, self.epoch)

        log_entry = {
            "epoch": self.epoch,
            "phase": phase,
            **metrics,
        }
        with open(self.log_dir / "retrieval_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_similarity_heatmap(
        self,
        sim_matrix: torch.Tensor,
        title: str = "similarity",
        epoch: int = 0,
    ):
        try:
            plt, _ = _import_matplotlib()
        except ImportError:
            return

        sim_np = sim_matrix.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(sim_np, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"{title} (epoch {epoch})")
        ax.set_xlabel("Text index")
        ax.set_ylabel("Image index")
        plt.colorbar(im, ax=ax)

        save_path = self.plot_dir / f"{title}_epoch{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.use_tensorboard and self.writer:
            self.writer.add_figure(f"similarity/{title}", fig, epoch)

    def log_embedding_viz(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_names: Optional[List[str]] = None,
        epoch: int = 0,
        method: str = "tsne",
    ):
        try:
            plt, _ = _import_matplotlib()
            TSNE, UMAP = _import_dim_reduction()
        except ImportError:
            warnings.warn("sklearn/umap/matplotlib not available, skipping embedding viz")
            return

        N = image_embeds.shape[0]
        if N > 2000:
            idx = torch.randperm(N)[:2000]
            image_embeds = image_embeds[idx]
            text_embeds = text_embeds[idx]
            if labels is not None:
                labels = labels[idx]
            N = 2000

        all_embeds = torch.cat([image_embeds, text_embeds], dim=0).cpu().numpy()
        modalities = np.array(['image'] * N + ['text'] * N)

        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, N-1))
        else:
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, N-1))

        embeds_2d = reducer.fit_transform(all_embeds)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i, (mod, color) in enumerate([('image', 'blue'), ('text', 'orange')]):
            mask = modalities == mod
            axes[0].scatter(
                embeds_2d[mask, 0], embeds_2d[mask, 1],
                c=color, label=mod, alpha=0.6, s=10
            )
        axes[0].set_title(f"{method.upper()} by Modality (epoch {epoch})")
        axes[0].legend()
        axes[0].set_xlabel("Dim 1")
        axes[0].set_ylabel("Dim 2")

        if labels is not None:
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            labels_np = np.concatenate([labels_np, labels_np])
            unique_labels = np.unique(labels_np)

            for lbl in unique_labels:
                mask = labels_np == lbl
                name = label_names[lbl] if label_names and lbl < len(label_names) else f"Class {lbl}"
                axes[1].scatter(
                    embeds_2d[mask, 0], embeds_2d[mask, 1],
                    label=name, alpha=0.6, s=10
                )
            axes[1].set_title(f"{method.upper()} by Pathology (epoch {epoch})")
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[1].set_title("No labels available")

        axes[1].set_xlabel("Dim 1")
        axes[1].set_ylabel("Dim 2")

        plt.tight_layout()
        save_path = self.plot_dir / f"embedding_{method}_epoch{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.use_tensorboard and self.writer:
            self.writer.add_figure(f"embeddings/{method}", fig, epoch)

    def log_gpu_memory(self, step: int = None):
        if not torch.cuda.is_available():
            return

        step = step or self.step
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        self.gpu_memory_log.append({
            "step": step,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
        })

        if self.use_tensorboard and self.writer:
            self.writer.add_scalar("gpu/allocated_gb", allocated, step)
            self.writer.add_scalar("gpu/reserved_gb", reserved, step)
            self.writer.add_scalar("gpu/max_allocated_gb", max_allocated, step)

        with open(self.log_dir / "gpu_memory_log.jsonl", "a") as f:
            f.write(json.dumps(self.gpu_memory_log[-1]) + "\n")

        return allocated, reserved, max_allocated

    def reset_peak_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _format_metric(value: Any) -> str:
        """Render numeric metrics consistently without failing on metadata."""
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            return f"{float(value):.4f}"
        return str(value)

    def log_epoch_summary(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float,
        epoch: int = None,
    ):
        epoch = epoch or self.epoch

        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*60}")
        print(f"Time: {epoch_time:.1f}s")
        print(f"\nTrain:")
        for k, v in train_metrics.items():
            print(f"  {k}: {self._format_metric(v)}")
        print(f"\nVal:")
        for k, v in val_metrics.items():
            print(f"  {k}: {self._format_metric(v)}")

        if torch.cuda.is_available():
            alloc, reserv, max_alloc = self.log_gpu_memory()
            print(f"\nGPU Memory: alloc={alloc:.2f}GB, reserv={reserv:.2f}GB, peak={max_alloc:.2f}GB")

        print(f"{'='*60}\n")

        summary = {
            "epoch": epoch,
            "time": epoch_time,
            "train": train_metrics,
            "val": val_metrics,
        }
        if torch.cuda.is_available():
            summary["gpu"] = self.gpu_memory_log[-1] if self.gpu_memory_log else {}

        with open(self.log_dir / "epoch_summaries.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")

    def close(self):
        if self.writer:
            self.writer.close()


def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5, 10),
    batch_size: int = 1024,
) -> Dict[str, float]:
    N, D = image_embeds.shape
    assert text_embeds.shape == (N, D)
    assert image_embeds.device == text_embeds.device

    device = image_embeds.device
    i2t_recalls = {k: 0 for k in k_values}
    t2i_recalls = {k: 0 for k in k_values}
    i2t_ranks = []
    t2i_ranks = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        img_batch = image_embeds[start:end]

        sim = torch.matmul(img_batch, text_embeds.T)

        for i in range(img_batch.size(0)):
            global_idx = start + i
            row = sim[i]
            sorted_idx = row.argsort(descending=True)
            rank = (sorted_idx == global_idx).nonzero(as_tuple=True)[0].item() + 1
            i2t_ranks.append(rank)

            for k in k_values:
                if rank <= k:
                    i2t_recalls[k] += 1

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        txt_batch = text_embeds[start:end]

        sim = torch.matmul(txt_batch, image_embeds.T)

        for j in range(txt_batch.size(0)):
            global_idx = start + j
            col = sim[j]
            sorted_idx = col.argsort(descending=True)
            rank = (sorted_idx == global_idx).nonzero(as_tuple=True)[0].item() + 1
            t2i_ranks.append(rank)

            for k in k_values:
                if rank <= k:
                    t2i_recalls[k] += 1

    for k in k_values:
        i2t_recalls[k] /= N
        t2i_recalls[k] /= N

    i2t_ranks = torch.tensor(i2t_ranks, dtype=torch.float, device=device)
    t2i_ranks = torch.tensor(t2i_ranks, dtype=torch.float, device=device)

    i2t_mrr = (1.0 / i2t_ranks).mean().item()
    t2i_mrr = (1.0 / t2i_ranks).mean().item()

    def _precision_at_k(ranks: torch.Tensor, k: int) -> float:
        return (ranks <= k).float().mean().item() / k

    return {
        "I2T_R@1": i2t_recalls.get(1, 0.0),
        "I2T_R@5": i2t_recalls.get(5, 0.0),
        "I2T_R@10": i2t_recalls.get(10, 0.0),
        "T2I_R@1": t2i_recalls.get(1, 0.0),
        "T2I_R@5": t2i_recalls.get(5, 0.0),
        "T2I_R@10": t2i_recalls.get(10, 0.0),
        "I2T_P@1": _precision_at_k(i2t_ranks, 1),
        "I2T_P@5": _precision_at_k(i2t_ranks, 5),
        "I2T_P@10": _precision_at_k(i2t_ranks, 10),
        "T2I_P@1": _precision_at_k(t2i_ranks, 1),
        "T2I_P@5": _precision_at_k(t2i_ranks, 5),
        "T2I_P@10": _precision_at_k(t2i_ranks, 10),
        "I2T_MRR": i2t_mrr,
        "T2I_MRR": t2i_mrr,
        "I2T_mAP": i2t_mrr,
        "T2I_mAP": t2i_mrr,
        "I2T_median_rank": float(i2t_ranks.median().item()),
        "T2I_median_rank": float(t2i_ranks.median().item()),
        "I2T_mean_rank": float(i2t_ranks.mean().item()),
        "T2I_mean_rank": float(t2i_ranks.mean().item()),
    }


def plot_similarity_heatmap(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    save_path: Union[str, Path],
    title: str = "Cosine Similarity Matrix",
    max_size: int = 128,
):
    try:
        plt, _ = _import_matplotlib()
    except ImportError:
        return

    N = image_embeds.shape[0]
    if N > max_size:
        idx = torch.randperm(N)[:max_size]
        image_embeds = image_embeds[idx]
        text_embeds = text_embeds[idx]
        N = max_size

    sim = torch.matmul(image_embeds, text_embeds.T).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Text index")
    ax.set_ylabel("Image index")
    plt.colorbar(im, ax=ax)

    ax.plot(range(N), range(N), 'w-', linewidth=0.5)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tsne_umap(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    save_path: Union[str, Path],
    labels: Optional[torch.Tensor] = None,
    label_names: Optional[List[str]] = None,
    method: str = "tsne",
    title: str = None,
    max_points: int = 2000,
):
    try:
        plt, _ = _import_matplotlib()
        TSNE, UMAP = _import_dim_reduction()
    except ImportError:
        return

    N = image_embeds.shape[0]
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        image_embeds = image_embeds[idx]
        text_embeds = text_embeds[idx]
        if labels is not None:
            labels = labels[idx]
        N = max_points

    all_embeds = torch.cat([image_embeds, text_embeds], dim=0).cpu().numpy()
    modalities = np.array(['image'] * N + ['text'] * N)

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, N-1))
    else:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, N-1))

    embeds_2d = reducer.fit_transform(all_embeds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for mod, color in [('image', 'blue'), ('text', 'orange')]:
        mask = modalities == mod
        axes[0].scatter(embeds_2d[mask, 0], embeds_2d[mask, 1], c=color, label=mod, alpha=0.6, s=10)
    axes[0].set_title(f"{method.upper()} by Modality")
    axes[0].legend()
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")

    if labels is not None:
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        labels_np = np.concatenate([labels_np, labels_np])
        unique_labels = np.unique(labels_np)

        for lbl in unique_labels:
            mask = labels_np == lbl
            name = label_names[lbl] if label_names and lbl < len(label_names) else f"Class {lbl}"
            axes[1].scatter(embeds_2d[mask, 0], embeds_2d[mask, 1], label=name, alpha=0.6, s=10)
        axes[1].set_title(f"{method.upper()} by Pathology")
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[1].set_title("No labels available")

    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(
    log_file: Union[str, Path],
    save_path: Union[str, Path],
    metrics: List[str] = None,
):
    try:
        plt, _ = _import_matplotlib()
        pd = _import_pandas()
    except ImportError:
        return

    data = []
    with open(log_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    if not data:
        return

    df = pd.DataFrame(data)

    if metrics is None:
        metrics = [c for c in df.columns if c not in ['step', 'epoch', 'phase']]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[i, 0]
        for phase in ['train', 'val', 'train_epoch']:
            phase_data = df[df['phase'] == phase]
            if len(phase_data) > 0 and metric in phase_data.columns:
                ax.plot(phase_data['step'], phase_data[metric], label=phase, alpha=0.7)
        ax.set_title(metric)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_epoch_summary(
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    epoch_time: float,
    gpu_memory: Optional[Dict[str, float]] = None,
):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch} SUMMARY  |  Time: {epoch_time:.1f}s")
    print(f"{'='*70}")

    print(f"\n{'TRAIN':^30} | {'VAL':^30}")
    print(f"{'-'*30}-+-{'-'*30}")

    all_keys = set(train_metrics.keys()) | set(val_metrics.keys())
    for k in sorted(all_keys):
        tr = train_metrics.get(k, float('nan'))
        vl = val_metrics.get(k, float('nan'))
        print(f"  {k:<28} | {tr:>6.4f} | {vl:>6.4f}")

    if gpu_memory:
        print(f"\nGPU Memory:")
        print(f"  Allocated: {gpu_memory.get('allocated_gb', 0):.2f} GB")
        print(f"  Reserved:  {gpu_memory.get('reserved_gb', 0):.2f} GB")
        print(f"  Peak:      {gpu_memory.get('max_allocated_gb', 0):.2f} GB")

    print(f"{'='*70}\n")


def log_gpu_memory(
    step: int = 0,
    log_file: Optional[Union[str, Path]] = None,
) -> Tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    if log_file:
        entry = {
            "step": step,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return allocated, reserved, max_allocated


__all__ = [
    "Visualizer",
    "compute_retrieval_metrics",
    "plot_similarity_heatmap",
    "plot_tsne_umap",
    "plot_loss_curves",
    "print_epoch_summary",
    "log_gpu_memory",
]