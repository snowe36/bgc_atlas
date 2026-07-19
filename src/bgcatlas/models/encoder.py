"""Set encoder: pool variable-length ESM2 protein vectors into a BGC embedding.

Architecture (small, trainable over frozen protein embeddings):
  per-protein MLP → set pooling (attention | mean | deepsets) → BGC embedding
  + optional projection head for contrastive training (SimCLR / SupCon).

Inputs are already-computed ESM2 protein embeddings (see scripts/run_esm_embed.py).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

PoolingName = Literal["attention", "mean", "deepsets"]


class AttentionPooling(nn.Module):
    """Masked attention pooling over protein tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), mask: (B, N) bool — True = valid protein
        logits = self.score(x).squeeze(-1)  # (B, N)
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class DeepSetsPooling(nn.Module):
    """Sum-pool then MLP (DeepSets-style)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        summed = (x * mask.unsqueeze(-1).float()).sum(dim=1)
        counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.mlp(summed / counts)


class BGCSetEncoder(nn.Module):
    """Encode a set of protein ESM embeddings into one BGC vector.

    Parameters
    ----------
    input_dim : int
        Protein embedding dim (1280 for ESM2-650M).
    hidden_dim : int
        Width of the per-protein MLP.
    embed_dim : int
        Output BGC embedding dimension (used for novelty / ablation).
    proj_dim : int
        Contrastive projection-head dimension.
    pooling : {'attention', 'mean', 'deepsets'}
    dropout : float
        Dropout inside the per-protein MLP (training only).
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        proj_dim: int = 128,
        pooling: PoolingName = "attention",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pooling not in {"attention", "mean", "deepsets"}:
            raise ValueError(f"Unknown pooling={pooling!r}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.pooling_name = pooling

        self.protein_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if pooling == "attention":
            self.pool: nn.Module = AttentionPooling(hidden_dim)
        elif pooling == "deepsets":
            self.pool = DeepSetsPooling(hidden_dim)
        else:
            self.pool = nn.Identity()  # mean handled in forward

        self.bgc_head = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def _pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_name == "mean":
            summed = (h * mask.unsqueeze(-1).float()).sum(dim=1)
            counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            return summed / counts
        return self.pool(h, mask)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized BGC embeddings (B, embed_dim)."""
        h = self.protein_mlp(x)
        pooled = self._pool(h, mask)
        z = self.bgc_head(pooled)
        return F.normalize(z, dim=-1)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Contrastive projection head; L2-normalized (B, proj_dim)."""
        return F.normalize(self.projector(z), dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, project: bool = False) -> torch.Tensor:
        z = self.encode(x, mask)
        if project:
            return self.project(z)
        return z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR NT-Xent loss for two views of a batch (each B × D, already normalized)."""
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = z @ z.T / temperature
    # mask self-similarity
    eye = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    # positives: i ↔ i+B
    targets = torch.arange(b, device=z.device)
    targets = torch.cat([targets + b, targets], dim=0)
    return F.cross_entropy(sim, targets)


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al.) over a batch of normalized embeddings.

    Positives = other samples sharing the same integer label. Samples with a
    unique label in the batch contribute 0 (skipped via mask).
    """
    device = z.device
    b = z.shape[0]
    if b < 2:
        return z.new_zeros(())

    sim = z @ z.T / temperature
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.T).fill_diagonal_(False)
    # For numerical stability
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    exp_logits = torch.exp(logits) * (~torch.eye(b, device=device, dtype=torch.bool)).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-9))

    pos_counts = mask_pos.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (mask_pos.float() * log_prob).sum(dim=1) / pos_counts
    # Only average over samples that have ≥1 positive
    has_pos = mask_pos.any(dim=1)
    if not has_pos.any():
        return z.new_zeros(())
    return -mean_log_prob_pos[has_pos].mean()
