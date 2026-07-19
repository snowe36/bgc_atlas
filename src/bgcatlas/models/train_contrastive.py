"""Contrastive BGC set-encoder over cached ESM2 protein vectors.

simclr / supcon; optional ``train_cutoff`` for leakage-safe temporal split.
Writes learned_embeddings.npy, learned_bgc_ids.csv, manifest, and artifacts/bgc_encoder.pt.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from bgcatlas.config import (
    DEFAULT_ENCODER_BATCH_SIZE,
    DEFAULT_ENCODER_EMBED_DIM,
    DEFAULT_ENCODER_EPOCHS,
    DEFAULT_ENCODER_FEAT_DROPOUT,
    DEFAULT_ENCODER_HIDDEN,
    DEFAULT_ENCODER_KEEP_FRAC,
    DEFAULT_ENCODER_LR,
    DEFAULT_ENCODER_MAX_PROTEINS,
    DEFAULT_ENCODER_OBJECTIVE,
    DEFAULT_ENCODER_POOLING,
    DEFAULT_ENCODER_PROJ_DIM,
    DEFAULT_ENCODER_SEED,
    DEFAULT_ENCODER_TEMPERATURE,
    DEFAULT_TEMPORAL_CUTOFF,
)
from bgcatlas.models.encoder import BGCSetEncoder, nt_xent_loss, supcon_loss
from bgcatlas.paths import PROCESSED, REPORTS, ROOT, ensure_dirs

LOG = logging.getLogger(__name__)

ARTIFACTS = ROOT / "artifacts"


@dataclass
class TrainConfig:
    objective: str = DEFAULT_ENCODER_OBJECTIVE
    pooling: str = DEFAULT_ENCODER_POOLING
    hidden_dim: int = DEFAULT_ENCODER_HIDDEN
    embed_dim: int = DEFAULT_ENCODER_EMBED_DIM
    proj_dim: int = DEFAULT_ENCODER_PROJ_DIM
    epochs: int = DEFAULT_ENCODER_EPOCHS
    batch_size: int = DEFAULT_ENCODER_BATCH_SIZE
    lr: float = DEFAULT_ENCODER_LR
    temperature: float = DEFAULT_ENCODER_TEMPERATURE
    keep_frac: float = DEFAULT_ENCODER_KEEP_FRAC
    feat_dropout: float = DEFAULT_ENCODER_FEAT_DROPOUT
    max_proteins: int = DEFAULT_ENCODER_MAX_PROTEINS
    seed: int = DEFAULT_ENCODER_SEED
    train_cutoff: str | None = None  # if set: train only on date_added < cutoff
    device: str | None = None


def pick_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BGCProteinDataset(Dataset):
    """One item = one BGC: padded protein tensor (N, D) + mask + optional class label."""

    def __init__(
        self,
        bgc_ids: list[str],
        protein_embeds: np.ndarray,
        protein_meta: pd.DataFrame,
        class_labels: dict[str, int] | None,
        max_proteins: int,
        keep_frac: float,
        feat_dropout: float,
        augment: bool,
    ) -> None:
        self.bgc_ids = list(bgc_ids)
        self.max_proteins = max_proteins
        self.keep_frac = keep_frac
        self.feat_dropout = feat_dropout
        self.augment = augment
        self.class_labels = class_labels or {}
        self.input_dim = int(protein_embeds.shape[1])

        meta = protein_meta.reset_index(drop=True)
        if "emb_idx" in meta.columns:
            groups: dict[str, list[int]] = {}
            for row in meta.itertuples(index=False):
                groups.setdefault(row.bgc_id, []).append(int(row.emb_idx))
        else:
            groups = {}
            for i, bgc_id in enumerate(meta["bgc_id"].tolist()):
                groups.setdefault(bgc_id, []).append(i)

        self._arrays: list[np.ndarray] = []
        self._labels: list[int] = []
        for bgc_id in self.bgc_ids:
            idxs = groups.get(bgc_id, [])[:max_proteins]
            if not idxs:
                arr = np.zeros((0, self.input_dim), dtype=np.float32)
            else:
                arr = np.asarray(protein_embeds[idxs], dtype=np.float32)
            self._arrays.append(arr)
            self._labels.append(int(self.class_labels.get(bgc_id, -1)))

    def __len__(self) -> int:
        return len(self.bgc_ids)

    def _augment_view(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (padded N×D, mask N) for one stochastic view."""
        n = arr.shape[0]
        if n == 0:
            pad = np.zeros((self.max_proteins, self.input_dim), dtype=np.float32)
            mask = np.zeros(self.max_proteins, dtype=bool)
            return pad, mask

        if self.augment and n > 1:
            keep = max(1, int(round(self.keep_frac * n)))
            keep = min(keep, n)
            sel = np.random.choice(n, size=keep, replace=False)
            view = arr[sel].copy()
        else:
            view = arr.copy()

        if self.augment and self.feat_dropout > 0:
            drop = np.random.rand(*view.shape) < self.feat_dropout
            view = view.copy()
            view[drop] = 0.0

        n_keep = view.shape[0]
        pad = np.zeros((self.max_proteins, self.input_dim), dtype=np.float32)
        mask = np.zeros(self.max_proteins, dtype=bool)
        n_use = min(n_keep, self.max_proteins)
        pad[:n_use] = view[:n_use]
        mask[:n_use] = True
        return pad, mask

    def __getitem__(self, idx: int):
        arr = self._arrays[idx]
        x1, m1 = self._augment_view(arr)
        x2, m2 = self._augment_view(arr)
        label = self._labels[idx]
        n_prot = int(arr.shape[0])
        return {
            "x1": torch.from_numpy(x1),
            "m1": torch.from_numpy(m1),
            "x2": torch.from_numpy(x2),
            "m2": torch.from_numpy(m2),
            "label": torch.tensor(label, dtype=torch.long),
            "n_prot": n_prot,
            "bgc_id": self.bgc_ids[idx],
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "x1": torch.stack([b["x1"] for b in batch]),
        "m1": torch.stack([b["m1"] for b in batch]),
        "x2": torch.stack([b["x2"] for b in batch]),
        "m2": torch.stack([b["m2"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "n_prot": [b["n_prot"] for b in batch],
        "bgc_id": [b["bgc_id"] for b in batch],
    }


def load_protein_cache(
    processed: Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    processed = processed or PROCESSED
    emb_path = processed / "esm_protein_embeddings.npy"
    meta_path = processed / "esm_protein_meta.csv"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Protein cache missing under {processed}. Run scripts/run_esm_embed.py first "
            "(writes esm_protein_embeddings.npy + esm_protein_meta.csv)."
        )
    embeds = np.load(emb_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    return embeds, meta, int(embeds.shape[1])


def _class_label_map(feature_meta: pd.DataFrame) -> dict[str, int]:
    classes = sorted(feature_meta["biosynth_class"].astype(str).unique())
    mapping = {c: i for i, c in enumerate(classes)}
    return {row.bgc_id: mapping[str(row.biosynth_class)] for row in feature_meta.itertuples(index=False)}


def _resolve_bgc_ids(
    protein_meta: pd.DataFrame,
    feature_meta: pd.DataFrame,
    train_cutoff: str | None,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Return (train_ids, all_ids_with_proteins, feature_meta aligned)."""
    all_ids = sorted(protein_meta["bgc_id"].unique().tolist())
    fm = feature_meta[feature_meta["bgc_id"].isin(all_ids)].copy()
    if train_cutoff is None:
        return all_ids, all_ids, fm

    if "date_added" not in fm.columns or fm["date_added"].isna().all():
        raise RuntimeError("train_cutoff set but feature_meta lacks date_added")

    date = fm["date_added"].fillna("9999").astype(str)
    train_mask = date < train_cutoff
    train_ids = sorted(fm.loc[train_mask, "bgc_id"].tolist())
    if len(train_ids) < 8:
        raise RuntimeError(
            f"Too few pre-cutoff BGCs for training ({len(train_ids)}) at cutoff={train_cutoff}"
        )
    LOG.info(
        "Train split @ %s: %d train / %d total BGCs with protein embeddings",
        train_cutoff,
        len(train_ids),
        len(all_ids),
    )
    return train_ids, all_ids, fm


@torch.no_grad()
def embed_all(
    model: BGCSetEncoder,
    bgc_ids: list[str],
    protein_embeds: np.ndarray,
    protein_meta: pd.DataFrame,
    max_proteins: int,
    device: str,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed every BGC (no augmentation). Returns (embeddings, n_proteins)."""
    ds = BGCProteinDataset(
        bgc_ids=bgc_ids,
        protein_embeds=protein_embeds,
        protein_meta=protein_meta,
        class_labels={},
        max_proteins=max_proteins,
        keep_frac=1.0,
        feat_dropout=0.0,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    model.eval()
    chunks: list[np.ndarray] = []
    n_prots: list[int] = []
    for batch in loader:
        x = batch["x1"].to(device)
        m = batch["m1"].to(device)
        z = model.encode(x, m)
        chunks.append(z.cpu().numpy().astype(np.float32))
        n_prots.extend(batch["n_prot"])
    return np.concatenate(chunks, axis=0), np.asarray(n_prots, dtype=np.int64)


def train_encoder(cfg: TrainConfig | None = None, outdir: Path | None = None) -> dict:
    """Train the set encoder and write learned_embeddings.* + checkpoint."""
    cfg = cfg or TrainConfig()
    ensure_dirs()
    outdir = outdir or PROCESSED
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg.seed)
    device = pick_device(cfg.device)
    LOG.info("Training BGC encoder on %s | %s", device, asdict(cfg))

    protein_embeds, protein_meta, input_dim = load_protein_cache()
    feature_meta = pd.read_parquet(PROCESSED / "feature_meta.parquet")
    train_ids, all_ids, fm = _resolve_bgc_ids(protein_meta, feature_meta, cfg.train_cutoff)
    class_labels = _class_label_map(fm)

    train_ds = BGCProteinDataset(
        bgc_ids=train_ids,
        protein_embeds=protein_embeds,
        protein_meta=protein_meta,
        class_labels=class_labels,
        max_proteins=cfg.max_proteins,
        keep_frac=cfg.keep_frac,
        feat_dropout=cfg.feat_dropout,
        augment=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=len(train_ds) > cfg.batch_size,
    )

    model = BGCSetEncoder(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        embed_dim=cfg.embed_dim,
        proj_dim=cfg.proj_dim,
        pooling=cfg.pooling,  # type: ignore[arg-type]
        dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    history: list[dict] = []
    t0 = time.time()
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            x1 = batch["x1"].to(device)
            m1 = batch["m1"].to(device)
            x2 = batch["x2"].to(device)
            m2 = batch["m2"].to(device)
            labels = batch["label"].to(device)

            opt.zero_grad(set_to_none=True)
            if cfg.objective == "simclr":
                p1 = model(x1, m1, project=True)
                p2 = model(x2, m2, project=True)
                loss = nt_xent_loss(p1, p2, temperature=cfg.temperature)
            elif cfg.objective == "supcon":
                p1 = model(x1, m1, project=True)
                p2 = model(x2, m2, project=True)
                z = torch.cat([p1, p2], dim=0)
                y = torch.cat([labels, labels], dim=0)
                # Drop unlabeled (-1)
                keep = y >= 0
                if keep.sum() < 4:
                    loss = z.new_zeros(()).requires_grad_()
                else:
                    loss = supcon_loss(z[keep], y[keep], temperature=cfg.temperature)
            else:
                raise ValueError(f"Unknown objective={cfg.objective!r}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        scheduler.step()
        mean_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": mean_loss, "lr": float(scheduler.get_last_lr()[0])})
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            LOG.info(
                "epoch %03d/%d  loss=%.4f  lr=%.2e", epoch, cfg.epochs, mean_loss, scheduler.get_last_lr()[0]
            )

    elapsed = time.time() - t0

    embeddings, n_prots = embed_all(
        model,
        all_ids,
        protein_embeds,
        protein_meta,
        max_proteins=cfg.max_proteins,
        device=device,
    )

    emb_path = outdir / "learned_embeddings.npy"
    ids_path = outdir / "learned_bgc_ids.csv"
    man_path = outdir / "learned_embed_manifest.json"
    ckpt_path = ARTIFACTS / "bgc_encoder.pt"

    np.save(emb_path, embeddings)
    pd.DataFrame({"bgc_id": all_ids, "n_proteins_embedded": n_prots}).to_csv(ids_path, index=False)

    split_tag = f"pre_{cfg.train_cutoff}" if cfg.train_cutoff else "all"
    representation_label = f"learned_{cfg.objective}_{cfg.pooling}_d{cfg.embed_dim}_{split_tag}"
    manifest = {
        "representation_label": representation_label,
        "objective": cfg.objective,
        "pooling": cfg.pooling,
        "hidden_dim": cfg.hidden_dim,
        "embed_dim": cfg.embed_dim,
        "proj_dim": cfg.proj_dim,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "temperature": cfg.temperature,
        "keep_frac": cfg.keep_frac,
        "feat_dropout": cfg.feat_dropout,
        "max_proteins": cfg.max_proteins,
        "seed": cfg.seed,
        "train_cutoff": cfg.train_cutoff,
        "n_train_bgcs": len(train_ids),
        "n_bgcs": len(all_ids),
        "input_dim": input_dim,
        "device": device,
        "elapsed_s": float(elapsed),
        "final_loss": history[-1]["loss"] if history else None,
        "checkpoint": str(ckpt_path.relative_to(ROOT)) if ckpt_path.exists() or True else None,
    }
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "input_dim": input_dim,
            "manifest": manifest,
        },
        ckpt_path,
    )
    hist_path = REPORTS / "learned_train_history.json"
    hist_path.write_text(json.dumps({"history": history, "manifest": manifest}, indent=2), encoding="utf-8")

    LOG.info(
        "Wrote %s (%s) + %s + checkpoint %s (%.0fs)",
        emb_path.name,
        embeddings.shape,
        man_path.name,
        ckpt_path,
        elapsed,
    )
    return manifest


def run_train_from_args(
    *,
    objective: str = DEFAULT_ENCODER_OBJECTIVE,
    pooling: str = DEFAULT_ENCODER_POOLING,
    embed_dim: int = DEFAULT_ENCODER_EMBED_DIM,
    hidden_dim: int = DEFAULT_ENCODER_HIDDEN,
    epochs: int = DEFAULT_ENCODER_EPOCHS,
    batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    lr: float = DEFAULT_ENCODER_LR,
    temperature: float = DEFAULT_ENCODER_TEMPERATURE,
    keep_frac: float = DEFAULT_ENCODER_KEEP_FRAC,
    feat_dropout: float = DEFAULT_ENCODER_FEAT_DROPOUT,
    seed: int = DEFAULT_ENCODER_SEED,
    train_cutoff: str | None = None,
    prospective: bool = False,
    device: str | None = None,
) -> dict:
    cutoff = train_cutoff
    if prospective and cutoff is None:
        cutoff = DEFAULT_TEMPORAL_CUTOFF
    cfg = TrainConfig(
        objective=objective,
        pooling=pooling,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=temperature,
        keep_frac=keep_frac,
        feat_dropout=feat_dropout,
        seed=seed,
        train_cutoff=cutoff,
        device=device,
    )
    return train_encoder(cfg)
