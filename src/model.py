from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


XRV_SCALE = 2048.0
XRV_SHIFT = -1024.0


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=True),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)

        if mask is not None:
            if mask.shape != scores.shape:
                mask = torch.ones_like(scores, dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~mask, float("-inf"))
            all_masked = (~mask).all(dim=-1, keepdim=True)
            if all_masked.any():
                scores = scores.masked_fill(all_masked, 0.0)

        weights = torch.softmax(scores, dim=-1)

        if mask is not None:
            if mask.shape != weights.shape:
                mask = torch.ones_like(weights, dtype=torch.bool, device=weights.device)
            weights = weights.masked_fill(~mask, 0.0)

        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=-2)
        return pooled


class ViewAttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.view_pool = AttentionPooling(embed_dim, hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.view_pool(x, view_mask)


class ProjectionHeadWithNorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class ContrastiveProjectionHead(nn.Module):
    """
    Projection used only by the contrastive objective.

    The modality-specific ProjectionHeadWithNorm modules produce
    the shared representation h used for retrieval.

    This module maps h -> z for InfoNCE.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if hidden_dim is None:
            hidden_dim = embedding_dim

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive for an MLP projection head")

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), p=2, dim=-1)


class GlobalInfoNCE(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        min_temp: float = 0.01,
        max_temp: float = 0.1,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if min_temp <= 0 or max_temp <= 0:
            raise ValueError("temperature bounds must be positive")

        if min_temp > max_temp:
            raise ValueError("min_temp must be <= max_temp")

        self.log_temp = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        self.min_temp = min_temp
        self.max_temp = max_temp

    @property
    def temperature(self) -> torch.Tensor:
        temperature = torch.exp(self.log_temp)
        return temperature.clamp(min=self.min_temp, max=self.max_temp)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert z1.ndim == 2 and z2.ndim == 2, f"Expected 2D tensors, got {z1.shape} and {z2.shape}"
        assert z1.shape == z2.shape, f"Shape mismatch: {z1.shape} vs {z2.shape}"

        temperature = self.temperature
        logits = torch.matmul(z1, z2.T) / temperature.clamp(min=0.01)
        logits = logits.clamp(min=-50.0, max=50.0)

        labels = torch.arange(
            z1.shape[0],
            device=z1.device,
            dtype=torch.long,
        )

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        loss = 0.5 * (loss_i2t + loss_t2i)

        loss_dict = {
            "loss_i2t": loss_i2t.detach(),
            "loss_t2i": loss_t2i.detach(),
            "temperature": temperature.detach(),
        }

        return loss, loss_dict


def build_resnet50_xrv(
    device: torch.device,
) -> nn.Module:
    import torchxrayvision as xrv

    backbone = xrv.models.ResNet(
        weights="resnet50-res512-all"
    )

    backbone.eval()
    backbone.to(device)

    class FrozenResNet50(nn.Module):
        def __init__(self, backbone: nn.Module):
            super().__init__()
            self.backbone = backbone
            self.feature_dim = 2048

        def train(self, mode: bool = True):
            super().train(False)
            self.backbone.eval()
            return self

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 4:
                raise ValueError(
                    f"Expected image tensor [B,C,H,W], got {x.shape}"
                )

            if x.shape[1] != 1:
                raise ValueError(
                    f"Expected grayscale input [B,1,H,W], got {x.shape}"
                )

            features = self.backbone.features(x)

            if features.ndim != 2 or features.shape[1] != self.feature_dim:
                raise RuntimeError(
                    f"Expected features [B,2048], got {features.shape}"
                )

            return features

    return FrozenResNet50(backbone).to(device)


def build_bioclinicalbert_encoder(
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Any]:

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    class FrozenBioClinicalBERT(nn.Module):
        def __init__(
            self,
            model: nn.Module,
            max_length: int,
        ):
            super().__init__()

            self.model = model
            self.max_length = max_length
            self.hidden_dim = model.config.hidden_size

        @torch.no_grad()
        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            return outputs.last_hidden_state[:, 0, :]

        @torch.no_grad()
        def encode_texts(
            self,
            texts: List[str],
            tokenizer,
            device: torch.device,
            batch_size: int = 32,
        ) -> torch.Tensor:
            self.eval()

            if not texts:
                return torch.empty(0, self.hidden_dim, dtype=torch.float32)

            all_features = []
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start:start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device, non_blocking=True)
                attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
                all_features.append(self.forward(input_ids, attention_mask).cpu())

            return torch.cat(all_features, dim=0)

    encoder = FrozenBioClinicalBERT(
        model=model,
        max_length=max_length,
    )

    return encoder.to(device), tokenizer


def get_resnet50_transform(
    input_size: int = 512,
    mean: Tuple[float, ...] = (0.5,),
    std: Tuple[float, ...] = (0.5,),
):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * XRV_SCALE + XRV_SHIFT),
    ])