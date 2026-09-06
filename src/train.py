import argparse
import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import CachedStudyDataset, collate_fn
from src.model import (
    ProjectionHeadWithNorm,
    ContrastiveProjectionHead,
    GlobalInfoNCE,
    ViewAttentionPooling,
)
from src.visualize_utils import compute_retrieval_metrics, Visualizer

VISUALIZATION_AVAILABLE = True


class MedAlignModel(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 2048,
        text_embed_dim: int = 768,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        pool_hidden_dim: int = 128,
        pool_dropout: float = 0.1,
        temperature_cross: float = 0.07,
        temperature_intra: float = 0.07,
        min_temp: float = 0.01,
        max_temp: float = 0.1,
        max_views_per_study: int = 2,
        contrastive_head_hidden_dim: Optional[int] = None,
        contrastive_head_dropout: float = 0.0,
        loss_weight_text: float = 0.3,
        loss_weight_img_txt: float = 0.7,
    ):
        super().__init__()

        self.max_views_per_study = max_views_per_study

        self.image_pooling = ViewAttentionPooling(
            embed_dim=image_embed_dim,
            hidden_dim=pool_hidden_dim,
            dropout=pool_dropout,
        )
        self.text_pooling_clin = nn.Identity()
        self.text_pooling_aug = nn.Identity()

        self.image_proj = ProjectionHeadWithNorm(
            input_dim=image_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            dropout=dropout,
        )
        self.text_proj_clin = ProjectionHeadWithNorm(
            input_dim=text_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            dropout=dropout,
        )
        self.text_proj_aug = ProjectionHeadWithNorm(
            input_dim=text_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            dropout=dropout,
        )

        contrastive_head_hidden_dim = (
            hidden_dim if contrastive_head_hidden_dim is None else contrastive_head_hidden_dim
        )
        self.image_contrastive_head = ContrastiveProjectionHead(
            embedding_dim=latent_dim,
            hidden_dim=contrastive_head_hidden_dim,
            dropout=contrastive_head_dropout,
        )
        self.clinical_contrastive_head = ContrastiveProjectionHead(
            embedding_dim=latent_dim,
            hidden_dim=contrastive_head_hidden_dim,
            dropout=contrastive_head_dropout,
        )
        self.augment_contrastive_head = ContrastiveProjectionHead(
            embedding_dim=latent_dim,
            hidden_dim=contrastive_head_hidden_dim,
            dropout=contrastive_head_dropout,
        )

        self.cross_loss_fn = GlobalInfoNCE(
            temperature=temperature_cross,
            min_temp=min_temp,
            max_temp=max_temp,
        )
        self.intra_loss_fn = GlobalInfoNCE(
            temperature=temperature_intra,
            min_temp=min_temp,
            max_temp=max_temp,
        )

        self.loss_weight_text = loss_weight_text
        self.loss_weight_img_txt = loss_weight_img_txt

    def _encode_representations(
        self,
        image_global: torch.Tensor,
        view_mask: torch.Tensor,
        clinical_global: torch.Tensor,
        augment_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_img_study = self.image_pooling(image_global, view_mask)
        h_img = self.image_proj(h_img_study)
        h_clin = self.text_proj_clin(clinical_global)
        h_aug = self.text_proj_aug(augment_global)
        return h_img, h_clin, h_aug

    def _project_for_contrastive(
        self,
        h_img: torch.Tensor,
        h_clin: torch.Tensor,
        h_aug: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_img = self.image_contrastive_head(h_img)
        z_clin = self.clinical_contrastive_head(h_clin)
        z_aug = self.augment_contrastive_head(h_aug)
        return z_img, z_clin, z_aug

    def forward(
        self,
        image_global: torch.Tensor,
        view_mask: torch.Tensor,
        clinical_global: torch.Tensor,
        augment_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_img, h_clin, h_aug = self._encode_representations(
            image_global, view_mask, clinical_global, augment_global
        )
        z_img, z_clin, z_aug = self._project_for_contrastive(h_img, h_clin, h_aug)

        L_text, _ = self.intra_loss_fn(z_clin, z_aug)

        L_img_clin, _ = self.cross_loss_fn(z_img, z_clin)
        L_img_aug, _ = self.cross_loss_fn(z_img, z_aug)
        L_img_txt = 0.5 * (L_img_clin + L_img_aug)

        total_loss = self.loss_weight_text * L_text + self.loss_weight_img_txt * L_img_txt

        tau_cross = self.cross_loss_fn.temperature.detach()
        tau_intra = self.intra_loss_fn.temperature.detach()

        loss_dict = {
            "total_loss": total_loss.detach(),
            "L_text": L_text.detach(),
            "L_img_clin": L_img_clin.detach(),
            "L_img_aug": L_img_aug.detach(),
            "L_img_txt": L_img_txt.detach(),
            "temperature": tau_cross,
            "temperature_cross": tau_cross,
            "temperature_intra": tau_intra,
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def forward_embeddings(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_global = batch["image_global"]
        view_mask = batch["view_mask"]
        clinical_global = batch["clinical_global"]
        augment_global = batch["augment_global"]

        h_img, h_clin, h_aug = self._encode_representations(
            image_global,
            view_mask,
            clinical_global,
            augment_global,
        )

        z_img, z_clin, z_aug = self._project_for_contrastive(
            h_img,
            h_clin,
            h_aug,
        )

        return h_img, h_clin, z_img, z_clin

    @torch.no_grad()
    def forward_contrastive_embeddings(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image_global = batch["image_global"]
        view_mask = batch["view_mask"]
        clinical_global = batch["clinical_global"]
        augment_global = batch["augment_global"]

        h_img, h_clin, h_aug = self._encode_representations(
            image_global, view_mask, clinical_global, augment_global
        )
        z_img, z_clin, _ = self._project_for_contrastive(h_img, h_clin, h_aug)
        return z_img, z_clin


class Trainer:
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self.cache_dir = Path("/kaggle/input/datasets/nghiaquchtrng/output-precompute/_output_/cache")
        self.train_metadata_file = Path("/kaggle/input/datasets/nghiaquchtrng/metadata-output-precompute/train_metadata.json")
        self.val_metadata_file = Path("/kaggle/input/datasets/nghiaquchtrng/metadata-output-precompute/val_metadata.json")

        train_ds = CachedStudyDataset(
            cache_dir=self.cache_dir,
            split="train",
            max_views_per_study=self.args.max_views_per_study,
            metadata_file=self.train_metadata_file,
        )
        val_ds = CachedStudyDataset(
            cache_dir=self.cache_dir,
            split="val",
            max_views_per_study=self.args.max_views_per_study,
            metadata_file=self.val_metadata_file,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.args.num_workers > 0,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.args.num_workers > 0,
        )

        sample = train_ds[0]
        self.image_embed_dim = sample["image_global"].shape[-1]
        self.text_embed_dim = sample["clinical_global"].shape[-1]

        print(f"Image embed dim: {self.image_embed_dim}, Text embed dim: {self.text_embed_dim}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"Effective batch size (studies/step): {self.args.batch_size}")
        if self.args.batch_size < self.args.min_batch_size_warning:
            print(f"WARNING: Batch size {self.args.batch_size} < {self.args.min_batch_size_warning}, consider increasing for better contrastive learning")

        self._init_model()
        self._init_optimizer()
        self._init_logging()

        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

    def _init_model(self):
        self.model = MedAlignModel(
            image_embed_dim=self.image_embed_dim,
            text_embed_dim=self.text_embed_dim,
            latent_dim=self.args.latent_dim,
            hidden_dim=self.args.hidden_dim,
            dropout=self.args.dropout,
            pool_hidden_dim=self.args.pool_hidden_dim,
            pool_dropout=self.args.pool_dropout,
            temperature_cross=self.args.temperature_cross,
            temperature_intra=self.args.temperature_intra,
            min_temp=self.args.min_temperature,
            max_temp=self.args.max_temperature,
            max_views_per_study=self.args.max_views_per_study,
            contrastive_head_hidden_dim=self.args.contrastive_head_hidden_dim,
            contrastive_head_dropout=self.args.contrastive_head_dropout,
            loss_weight_text=self.args.loss_weight_text,
            loss_weight_img_txt=self.args.loss_weight_img_txt,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
        print(
            "Contrastive head: mlp "
            f"(hidden={self.args.contrastive_head_hidden_dim}, "
            f"dropout={self.args.contrastive_head_dropout})"
        )
        print(
            "Temperature: separate "
            f"(cross={self.args.temperature_cross}, "
            f"intra={self.args.temperature_intra})"
        )

    def _init_optimizer(self):
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

        self.sched = CosineAnnealingWarmRestarts(
            self.optim,
            T_0=self.args.T_0,
            T_mult=self.args.T_mult,
            eta_min=self.args.eta_min,
        )

        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

    def _init_logging(self):
        self.log_dir = Path(self.args.log_dir)
        self.ckpt_dir = Path(self.args.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "train_log.jsonl"
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2)

        self.visualizer = None
        if VISUALIZATION_AVAILABLE and self.args.enable_visualization:
            self.visualizer = Visualizer(
                log_dir=str(self.log_dir),
                plot_dir=str(self.log_dir / "plots"),
                use_tensorboard=self.args.use_tensorboard,
            )
            print("Visualization enabled (TensorBoard + matplotlib)")

        self.retrieval_every = self.args.retrieval_every_n_epochs
        self.embedding_every = self.args.embedding_every_n_epochs
        self.heatmap_every = self.args.heatmap_every_n_epochs
        self.gpu_log_every = self.args.gpu_log_every_n_steps
        self.label_names = self.args.label_names
        self._val_embeds_cache = None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()

        image_global = batch["image_global"].to(self.device, non_blocking=True)
        view_mask = batch["view_mask"].to(self.device, non_blocking=True)
        clinical_global = batch["clinical_global"].to(self.device, non_blocking=True)
        augment_global = batch["augment_global"].to(self.device, non_blocking=True)

        if self.scaler:
            with torch.cuda.amp.autocast():
                total_loss, loss_dict = self.model(
                    image_global, view_mask,
                    clinical_global, augment_global,
                )
            total_loss = total_loss.mean() / self.args.grad_accum_steps
            self.scaler.scale(total_loss).backward()

            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and v.numel() > 1:
                    loss_dict[k] = v.mean().item()

            if (self.step + 1) % self.args.grad_accum_steps == 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad()
        else:
            total_loss, loss_dict = self.model(
                image_global, view_mask,
                clinical_global, augment_global,
            )
            total_loss = total_loss.mean() / self.args.grad_accum_steps

            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and v.numel() > 1:
                    loss_dict[k] = v.mean().item()
            total_loss.backward()

            if (self.step + 1) % self.args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optim.step()
                self.sched.step()
                self.optim.zero_grad()

        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        loss_dict["lr"] = self.optim.param_groups[0]["lr"]

        if self.visualizer:
            lr = self.optim.param_groups[0]["lr"]
            temp = loss_dict.get("temperature", None)
            self.visualizer.log_loss(loss_dict, "train")
            self.visualizer.log_learning_rate(lr)
            if temp is not None:
                self.visualizer.log_temperature(temp)
            if self.step % self.gpu_log_every == 0:
                self.visualizer.log_gpu_memory()

        return loss_dict

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        totals = {}
        n = 0
        for batch in tqdm(self.val_loader, desc="Val", leave=False, disable=not sys.stderr.isatty()):
            image_global = batch["image_global"].to(self.device, non_blocking=True)
            view_mask = batch["view_mask"].to(self.device, non_blocking=True)
            clinical_global = batch["clinical_global"].to(self.device, non_blocking=True)
            augment_global = batch["augment_global"].to(self.device, non_blocking=True)

            _, loss_dict = self.model(
                image_global, view_mask,
                clinical_global, augment_global,
            )

            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() > 1:
                        v = v.mean().item()
                    else:
                        v = v.item()
                totals[k] = totals.get(k, 0.0) + v
            n += 1

        return {k: v / n for k, v in totals.items()}

    def save_checkpoint(self, is_best: bool = False, val_loss: Optional[float] = None):
        model_state = self.model.state_dict()

        ckpt = {
            "epoch": self.epoch,
            "step": self.step,
            "model": model_state,
            "optim": self.optim.state_dict(),
            "sched": self.sched.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "best_val_loss": self.best_val_loss,
            "args": vars(self.args),
        }

        latest_path = self.ckpt_dir / "latest.pt"
        torch.save(ckpt, latest_path)

        epoch_label = self.epoch + 1
        detailed_name = (
            f"checkpoint_epoch_{epoch_label:02d}_"
            f"step_{self.step:06d}"
            f"{f'_val_{val_loss:.4f}' if val_loss is not None else ''}.pt"
        )
        torch.save(ckpt, self.ckpt_dir / detailed_name)

        if is_best:
            best_name = (
                f"best_epoch_{epoch_label:02d}_"
                f"step_{self.step:06d}_"
                f"val_{self.best_val_loss:.4f}.pt"
            )
            torch.save(ckpt, self.ckpt_dir / best_name)

            best_path = self.ckpt_dir / "best.pt"
            torch.save(ckpt, best_path)

    def log_metrics(self, metrics: Dict[str, float], phase: str):
        log_entry = {**metrics, "step": self.step, "epoch": self.epoch, "phase": phase}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if phase == "train":
            print(f"  Step {self.step}: loss={metrics.get('total_loss', 0):.4f} "
                  f"L_text={metrics.get('L_text', 0):.4f} L_img_txt={metrics.get('L_img_txt', 0):.4f} "
                  f"tau_cross={metrics.get('temperature_cross', metrics.get('temperature', 0)):.4f} "
                  f"tau_intra={metrics.get('temperature_intra', metrics.get('temperature', 0)):.4f} "
                  f"lr={metrics.get('lr', 0):.2e}")
        else:
            print(f"  Val: loss={metrics.get('total_loss', 0):.4f} "
                  f"L_text={metrics.get('L_text', 0):.4f} L_img_txt={metrics.get('L_img_txt', 0):.4f} "
                  f"tau_cross={metrics.get('temperature_cross', metrics.get('temperature', 0)):.4f} "
                  f"tau_intra={metrics.get('temperature_intra', metrics.get('temperature', 0)):.4f}")

    @torch.no_grad()
    def _log_retrieval(self, epoch: int):
        if not self.visualizer:
            return

        self.model.eval()
        max_samples = 1000
        all_img = []
        all_txt = []

        for batch in self.val_loader:
            if len(all_img) >= max_samples:
                break

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            h_img, h_txt, z_img, z_txt = self.model.forward_embeddings(batch)

            all_img.append(z_img.cpu())
            all_txt.append(z_txt.cpu())

        if not all_img:
            return

        image_embeds = torch.cat(all_img, dim=0)[:max_samples]
        text_embeds = torch.cat(all_txt, dim=0)[:max_samples]

        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        metrics = compute_retrieval_metrics(image_embeds, text_embeds)
        self.visualizer.log_retrieval_metrics(metrics, "val")

        self._val_embeds_cache = (image_embeds, text_embeds)

        print(f"  Retrieval: I2T R@1={metrics['I2T_R@1']:.4f}, T2I R@1={metrics['T2I_R@1']:.4f}")

    @torch.no_grad()
    def _log_embeddings(self, epoch: int):
        if not self.visualizer or self._val_embeds_cache is None:
            return

        image_embeds, text_embeds = self._val_embeds_cache

        labels = None
        for batch in self.val_loader:
            if 'labels' in batch:
                labels = batch['labels'][:len(image_embeds)]
                break

        from src.visualize_utils import plot_tsne_umap
        plot_tsne_umap(
            image_embeds, text_embeds,
            save_path=self.visualizer.plot_dir / f"tsne_epoch{epoch}.png",
            labels=labels,
            label_names=self.label_names,
            method="tsne",
            title=f"t-SNE Embeddings (epoch {epoch})",
        )
        plot_tsne_umap(
            image_embeds, text_embeds,
            save_path=self.visualizer.plot_dir / f"umap_epoch{epoch}.png",
            labels=labels,
            label_names=self.label_names,
            method="umap",
            title=f"UMAP Embeddings (epoch {epoch})",
        )

    @torch.no_grad()
    def _log_heatmap(self, epoch: int):
        if not self.visualizer or self._val_embeds_cache is None:
            return

        image_embeds, text_embeds = self._val_embeds_cache

        from src.visualize_utils import plot_similarity_heatmap
        plot_similarity_heatmap(
            image_embeds, text_embeds,
            save_path=self.visualizer.plot_dir / f"similarity_heatmap_epoch{epoch}.png",
            title=f"Cosine Similarity (epoch {epoch})",
        )

    def train(self):

        print(f"\nTraining {self.args.epochs} epochs on {self.device} (AMP={self.args.use_amp})")

        for epoch in range(self.epoch, self.args.epochs):
            self.epoch = epoch
            t0 = time.time()

            train_totals = {}
            n = 0
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.args.epochs}",
                disable=not sys.stderr.isatty(),
            )
            for batch in pbar:
                loss_dict = self.train_step(batch)
                for k, v in loss_dict.items():
                    train_totals[k] = train_totals.get(k, 0.0) + v
                n += 1
                self.step += 1

                if self.step % self.args.log_interval == 0:
                    self.log_metrics(loss_dict, "train")
                pbar.set_postfix(loss=f"{loss_dict.get('total_loss', 0):.4f}")

            avg_train = {k: v / n for k, v in train_totals.items()}
            self.log_metrics(avg_train, "train_epoch")

            val_metrics = self.validate()
            self.log_metrics(val_metrics, "val")

            val_loss = val_metrics.get("total_loss", float('inf'))
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(is_best)

            completed_epoch = epoch + 1
            if self.visualizer:
                self.visualizer.epoch = completed_epoch
                self.visualizer.log_epoch_summary(
                    train_metrics=avg_train,
                    val_metrics=val_metrics,
                    epoch_time=time.time() - t0,
                    epoch=completed_epoch,
                )

            if completed_epoch % self.retrieval_every == 0:
                self._log_retrieval(completed_epoch)

            if completed_epoch % self.embedding_every == 0:
                self._log_embeddings(completed_epoch)

            if completed_epoch % self.heatmap_every == 0:
                self._log_heatmap(completed_epoch)

            epoch_time = time.time() - t0
            print(f"Epoch {epoch+1} done in {epoch_time:.1f}s | "
                  f"Train {avg_train.get('total_loss', 0):.4f} | "
                  f"Val {val_loss:.4f} {'*' if is_best else ''} | "
                  f"LR {self.optim.param_groups[0]['lr']:.2e}")

        print(f"\nDone. Best val loss: {self.best_val_loss:.4f}")

        if self.visualizer:
            self.visualizer.close()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()

                    if not line or line.startswith("#"):
                        continue

                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")

                        os.environ.setdefault(k, v)

        except Exception as e:
            print(f"Warning: failed to load .env: {e}")

    def env_str(name, default=None):
        return os.environ.get(name, default)

    def env_int(name, default):
        value = os.environ.get(name)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError:
            raise ValueError(
                f"Invalid integer value for {name}: {value!r}"
            )

    def env_float(name, default):
        value = os.environ.get(name)
        if value is None:
            return default

        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"Invalid float value for {name}: {value!r}"
            )

    def env_bool(name, default):
        value = os.environ.get(name)

        if value is None:
            return default

        value = value.strip().lower()

        if value in ("1", "true", "yes", "y", "on"):
            return True

        if value in ("0", "false", "no", "n", "off"):
            return False

        raise ValueError(
            f"Invalid boolean value for {name}: {value!r}. "
            f"Expected true/false, 1/0, yes/no, on/off."
        )

    parser = argparse.ArgumentParser(
        description="Train medalign-pipeline alignment on cached embeddings"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=env_str("CACHE_DIR", env_str("CACHE")),
        help="Directory with precomputed embeddings "
             "(or set CACHE_DIR/CACHE in environment/.env)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=env_str(
            "LOG_DIR",
            "logs/medalign-deep-projection-separate-temperature",
        ),
        help="Log directory",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=env_str(
            "CHECKPOINT_DIR",
            "checkpoints/medalign-deep-projection-separate-temperature",
        ),
        help="Checkpoint directory",
    )

    parser.add_argument(
        "--max-views-per-study",
        type=int,
        default=env_int("MAX_VIEWS", 2),
        help="Max views per study",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=env_int("NUM_WORKERS", 4),
        help="DataLoader workers",
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=env_int("LATENT_DIM", 256),
        help="Latent dimension for projections",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=env_int("POOL_HIDDEN_DIM", 512),
        help="Hidden dimension in projection heads",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=env_float("DROPOUT", 0.1),
        help="Dropout in projection heads",
    )

    parser.add_argument(
        "--contrastive-head-hidden-dim",
        type=int,
        default=(
            env_int("CONTRASTIVE_HEAD_HIDDEN_DIM", 512)
            if os.environ.get("CONTRASTIVE_HEAD_HIDDEN_DIM") is not None
            else None
        ),
        help="Hidden dimension of the contrastive MLP "
             "(defaults to --hidden-dim)",
    )

    parser.add_argument(
        "--contrastive-head-dropout",
        type=float,
        default=env_float("CONTRASTIVE_HEAD_DROPOUT", 0.0),
        help="Dropout in the contrastive MLP projection head",
    )

    parser.add_argument(
        "--pool-hidden-dim",
        type=int,
        default=env_int("POOL_HIDDEN_DIM", 512),
        help="Hidden dim for attention pooling",
    )

    parser.add_argument(
        "--pool-dropout",
        type=float,
        default=env_float("POOL_DROPOUT", 0.1),
        help="Dropout in attention pooling",
    )

    parser.add_argument(
        "--temperature-cross",
        type=float,
        default=env_float("TEMPERATURE_CROSS", 0.07),
        help="Initial temperature for image-text pairs",
    )

    parser.add_argument(
        "--temperature-intra",
        type=float,
        default=env_float("TEMPERATURE_INTRA", 0.07),
        help="Initial temperature for clinical-augmented text pairs",
    )

    parser.add_argument(
        "--min-temperature",
        type=float,
        default=env_float("MIN_TEMPERATURE", 0.01),
        help="Min temperature clamp",
    )

    parser.add_argument(
        "--max-temperature",
        type=float,
        default=env_float("MAX_TEMPERATURE", 0.1),
        help="Max temperature clamp",
    )

    parser.add_argument(
        "--loss-weight-text",
        type=float,
        default=env_float("LOSS_WEIGHT_TEXT", 0.3),
        help="Weight for L_text in total loss",
    )

    parser.add_argument(
        "--loss-weight-img-txt",
        type=float,
        default=env_float("LOSS_WEIGHT_IMG_TXT", 0.7),
        help="Weight for L_img_txt in total loss",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=env_float("LR", 1e-4),
        help="Learning rate",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=env_float("WEIGHT_DECAY", 0.01),
        help="Weight decay",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=env_int("EPOCHS", 20),
        help="Number of epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=env_int("BATCH_SIZE", 128),
        help="Batch size (studies per step)",
    )

    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=env_int("GRAD_ACCUM", 1),
        help="Gradient accumulation steps",
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=env_float("MAX_GRAD_NORM", 1.0),
        help="Max gradient norm for clipping",
    )

    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=env_bool("USE_AMP", False),
        help="Use AMP",
    )

    parser.add_argument(
        "--T_0",
        type=int,
        default=env_int("T_0", 10),
        help="CosineAnnealingWarmRestarts T_0",
    )

    parser.add_argument(
        "--T_mult",
        type=int,
        default=env_int("T_MULT", 2),
        help="CosineAnnealingWarmRestarts T_mult",
    )

    parser.add_argument(
        "--eta_min",
        type=float,
        default=env_float("ETA_MIN", 1e-6),
        help="Minimum learning rate",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=env_int("LOG_INTERVAL", 50),
        help="Log interval (steps)",
    )

    parser.add_argument(
        "--min-batch-size-warning",
        type=int,
        default=env_int("MIN_BATCH_SIZE_WARNING", 128),
        help="Warn if batch size below this",
    )

    parser.add_argument(
        "--enable-visualization",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ENABLE_VIZ", True),
        help="Enable visualization",
    )

    parser.add_argument(
        "--use-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=env_bool("USE_TENSORBOARD", True),
        help="Use TensorBoard",
    )

    parser.add_argument(
        "--retrieval-every-n-epochs",
        type=int,
        default=env_int("RETRIEVAL_EVERY", 5),
        help="Compute retrieval metrics every N epochs",
    )

    parser.add_argument(
        "--embedding-every-n-epochs",
        type=int,
        default=env_int("EMBEDDING_EVERY", 10),
        help="Plot embeddings every N epochs",
    )

    parser.add_argument(
        "--heatmap-every-n-epochs",
        type=int,
        default=env_int("HEATMAP_EVERY", 5),
        help="Plot similarity heatmap every N epochs",
    )

    parser.add_argument(
        "--gpu-log-every-n-steps",
        type=int,
        default=env_int("GPU_LOG_EVERY", 100),
        help="Log GPU memory every N steps",
    )

    parser.add_argument(
        "--label-names",
        type=str,
        nargs="*",
        default=None,
        help="Pathology label names for embedding viz",
    )

    args = parser.parse_args()

    if not args.cache_dir:
        parser.error(
            "--cache-dir is required "
            "(or set CACHE_DIR/CACHE in environment/.env)"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        )
        print(
            f"Available GPUs: {torch.cuda.device_count()}"
        )

    print("\n" + "=" * 60)
    print("MedAlign Training Configuration")
    print("=" * 60)

    print(f"Cache dir              : {args.cache_dir}")
    print(f"Log dir                : {args.log_dir}")
    print(f"Checkpoint dir         : {args.checkpoint_dir}")
    print(f"Max views              : {args.max_views_per_study}")
    print(f"Num workers            : {args.num_workers}")

    print(f"Latent dim             : {args.latent_dim}")
    print(f"Hidden dim             : {args.hidden_dim}")
    print(f"Dropout                : {args.dropout}")
    print(f"Pool hidden dim        : {args.pool_hidden_dim}")
    print(f"Pool dropout           : {args.pool_dropout}")

    print(f"Contrastive head: mlp")
    print(
        f"Contrastive head dim   : "
        f"{args.contrastive_head_hidden_dim}"
    )
    print(
        f"Contrastive dropout    : "
        f"{args.contrastive_head_dropout}"
    )

    print(f"Temperature: separate")
    print(f"Temperature cross      : {args.temperature_cross}")
    print(f"Temperature intra      : {args.temperature_intra}")
    print(f"Min/Max temperature    : {args.min_temperature}/{args.max_temperature}")

    print(f"Loss weight text       : {args.loss_weight_text}")
    print(f"Loss weight img-txt    : {args.loss_weight_img_txt}")

    print(f"Learning rate          : {args.lr}")
    print(f"Weight decay           : {args.weight_decay}")
    print(f"Epochs                 : {args.epochs}")
    print(f"Batch size             : {args.batch_size}")
    print(f"Grad accumulation      : {args.grad_accum_steps}")
    print(f"AMP                    : {args.use_amp}")

    print(f"T_0                    : {args.T_0}")
    print(f"T_mult                 : {args.T_mult}")
    print(f"Eta min                : {args.eta_min}")

    print(f"Visualization          : {args.enable_visualization}")
    print(f"TensorBoard            : {args.use_tensorboard}")
    print(f"Retrieval every        : {args.retrieval_every_n_epochs}")
    print(f"Embedding every        : {args.embedding_every_n_epochs}")
    print(f"Heatmap every          : {args.heatmap_every_n_epochs}")
    print(f"GPU log every          : {args.gpu_log_every_n_steps}")

    print("=" * 60 + "\n")

    trainer = Trainer(args, device)
    trainer.train()


if __name__ == "__main__":
    set_seed(1234)
    main()