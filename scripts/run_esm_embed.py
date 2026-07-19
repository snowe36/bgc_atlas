#!/usr/bin/env python
"""Embed MIBiG CDS proteins with ESM2 and pool to per-BGC vectors (GPU step).

Usage:
    uv sync --extra embed
    python scripts/run_esm_embed.py
    python scripts/run_esm_embed.py --from-cache --pooling length_weighted
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bgcatlas.config import (
    DEFAULT_ESM_BATCH_TOKENS,
    DEFAULT_ESM_MAX_AA,
    DEFAULT_ESM_MAX_PROTEINS,
    DEFAULT_ESM_MODEL,
    DEFAULT_ESM_POOLING,
)
from bgcatlas.embed_pool import pool_bgcs, representation_label


def load_model(model_name: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
    except Exception:
        model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    return tok, model


def embed_batch(seqs: list[str], tok, model, device: str, max_aa: int) -> np.ndarray:
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_aa + 2)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    hidden = out.last_hidden_state  # (B, L, D)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return pooled.float().cpu().numpy()


def _write_outputs(
    outdir: Path,
    bgc_ids: list[str],
    bgc_embeds: np.ndarray,
    counts: np.ndarray,
    protein_embeds: np.ndarray | None,
    protein_meta: pd.DataFrame | None,
    manifest: dict,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "esm_embeddings.npy", bgc_embeds)
    pd.DataFrame({"bgc_id": bgc_ids, "n_proteins_embedded": counts}).to_csv(
        outdir / "esm_bgc_ids.csv", index=False
    )
    if protein_embeds is not None and protein_meta is not None:
        np.save(outdir / "esm_protein_embeddings.npy", protein_embeds)
        protein_meta.to_csv(outdir / "esm_protein_meta.csv", index=False)
    with open(outdir / "esm_embed_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {bgc_embeds.shape} BGC embedding matrix -> {outdir / 'esm_embeddings.npy'}")
    print(f"Manifest -> {outdir / 'esm_embed_manifest.json'}")


def run_from_cache(outdir: Path, pooling: str) -> None:
    """Re-pool BGC vectors from cached protein embeddings (CPU, no model load)."""
    prot_path = outdir / "esm_protein_embeddings.npy"
    meta_path = outdir / "esm_protein_meta.csv"
    if not prot_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Protein cache missing under {outdir}. Run a full GPU embed first "
            "(writes esm_protein_embeddings.npy + esm_protein_meta.csv)."
        )
    embeds = np.load(prot_path)
    meta = pd.read_csv(meta_path)
    if len(meta) != len(embeds):
        raise RuntimeError("Protein meta rows != embedding rows; cache is corrupt.")
    prev = {}
    man_path = outdir / "esm_embed_manifest.json"
    if man_path.exists():
        prev = json.loads(man_path.read_text(encoding="utf-8"))

    bgc_ids, bgc_embeds, counts = pool_bgcs(
        embeds,
        meta["bgc_id"].to_numpy(),
        meta["aa_length"].to_numpy(dtype=np.float64),
        pooling=pooling,
    )
    manifest = {
        **prev,
        "pooling": pooling,
        "n_bgcs": int(len(bgc_ids)),
        "n_proteins": int(len(meta)),
        "from_cache": True,
        "representation_label": representation_label(prev.get("model", "esm"), pooling),
    }
    _write_outputs(outdir, bgc_ids, bgc_embeds, counts, None, None, manifest)


def run_embed(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    if args.from_cache:
        run_from_cache(outdir, pooling=args.pooling)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model} pooling={args.pooling} max_aa={args.max_aa}")

    df = pd.read_parquet(args.input)
    df["translation"] = df["translation"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
    df = df[df["translation"].str.len() >= 10].copy()
    if "aa_length" not in df.columns:
        df["aa_length"] = df["translation"].str.len()
    df["aa_length"] = pd.to_numeric(df["aa_length"], errors="coerce").fillna(0).astype(int)
    df["translation"] = df["translation"].str.slice(0, args.max_aa)

    df = (
        df.sort_values(["bgc_id", "aa_length"], ascending=[True, False])
        .groupby("bgc_id")
        .head(args.max_proteins_per_bgc)
        .reset_index(drop=True)
    )
    print(f"Embedding {len(df)} protein sequences across {df['bgc_id'].nunique()} BGCs")

    tok, model = load_model(args.model, device)
    order = df["translation"].str.len().sort_values(ascending=False).index
    df_sorted = df.loc[order].reset_index(drop=True)

    all_embeds = np.zeros((len(df_sorted), model.config.hidden_size), dtype=np.float32)
    batch_idx: list[int] = []
    batch_seqs: list[str] = []
    t0 = time.time()
    n_done = 0
    n_batches = 0

    def flush():
        nonlocal batch_idx, batch_seqs, n_done, n_batches
        if not batch_seqs:
            return
        bt0 = time.time()
        emb = embed_batch(batch_seqs, tok, model, device, max_aa=args.max_aa)
        for local_i, global_i in enumerate(batch_idx):
            all_embeds[global_i] = emb[local_i]
        n_done += len(batch_seqs)
        n_batches += 1
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (len(df_sorted) - n_done) / rate if rate > 0 else float("nan")
        print(
            f"  batch {n_batches}: +{len(batch_seqs)} seqs (maxlen={max(len(s) for s in batch_seqs)}) "
            f"in {time.time() - bt0:.2f}s | {n_done}/{len(df_sorted)} done | "
            f"{rate:.1f} seq/s | elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
            flush=True,
        )
        batch_idx, batch_seqs = [], []

    for i, seq in enumerate(df_sorted["translation"].tolist()):
        projected = (max(len(s) for s in batch_seqs + [seq]) + 2) * (len(batch_seqs) + 1)
        if batch_seqs and projected > args.batch_tokens:
            flush()
        batch_idx.append(i)
        batch_seqs.append(seq)
    flush()
    elapsed = time.time() - t0
    print(f"Done embedding proteins in {elapsed:.0f}s")

    bgc_ids, bgc_embeds, counts = pool_bgcs(
        all_embeds,
        df_sorted["bgc_id"].to_numpy(),
        df_sorted["aa_length"].to_numpy(dtype=np.float64),
        pooling=args.pooling,
    )

    locus = (
        df_sorted["locus_tag"].astype(str)
        if "locus_tag" in df_sorted.columns
        else pd.Series([""] * len(df_sorted))
    )
    protein_meta = pd.DataFrame(
        {
            "bgc_id": df_sorted["bgc_id"].to_numpy(),
            "locus_tag": locus.to_numpy(),
            "aa_length": df_sorted["aa_length"].to_numpy(),
            "emb_idx": np.arange(len(df_sorted)),
        }
    )
    manifest = {
        "model": args.model,
        "pooling": args.pooling,
        "max_aa": args.max_aa,
        "max_proteins_per_bgc": args.max_proteins_per_bgc,
        "batch_tokens": args.batch_tokens,
        "device": device,
        "n_bgcs": int(len(bgc_ids)),
        "n_proteins": int(len(df_sorted)),
        "hidden_size": int(all_embeds.shape[1]),
        "elapsed_s": float(elapsed),
        "from_cache": False,
        "representation_label": representation_label(args.model, args.pooling),
    }
    _write_outputs(outdir, bgc_ids, bgc_embeds, counts, all_embeds, protein_meta, manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description="ESM2 protein → BGC embeddings (V2)")
    ap.add_argument("--input", default="data/processed/mibig_proteins.parquet")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--model", default=DEFAULT_ESM_MODEL)
    ap.add_argument(
        "--pooling",
        default=DEFAULT_ESM_POOLING,
        choices=["mean", "length_weighted"],
        help="How to aggregate protein vectors into a BGC vector",
    )
    ap.add_argument("--max-aa", type=int, default=DEFAULT_ESM_MAX_AA)
    ap.add_argument("--max-proteins-per-bgc", type=int, default=DEFAULT_ESM_MAX_PROTEINS)
    ap.add_argument("--batch-tokens", type=int, default=DEFAULT_ESM_BATCH_TOKENS)
    ap.add_argument(
        "--from-cache",
        action="store_true",
        help="Re-pool from esm_protein_embeddings.npy (no GPU / model download)",
    )
    args = ap.parse_args()
    run_embed(args)


if __name__ == "__main__":
    main()
