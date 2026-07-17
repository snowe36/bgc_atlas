#!/usr/bin/env python
"""Embed MIBiG CDS protein sequences with ESM2 and pool to per-BGC vectors.

Standalone by design: this is the one GPU-dependent step in the pipeline, meant
to run on a rented GPU pod (see README "GPU embeddings" section), not as part
of the CPU-only local pipeline. It reads `mibig_proteins.parquet` (bgc_id,
locus_tag, translation, ...) and writes a dense per-BGC embedding matrix.

Usage:
    python run_esm_embed.py \
        --input data/processed/mibig_proteins.parquet \
        --outdir data/processed \
        --model facebook/esm2_t33_650M_UR50D
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MAX_AA = 700  # domain-composition signal lives in the first ~700 aa; caps worst-case batch cost


def load_model(model_name: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
    except Exception:
        model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    return tok, model


def embed_batch(seqs: list[str], tok, model, device: str) -> np.ndarray:
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_AA + 2)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    hidden = out.last_hidden_state  # (B, L, D)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    # drop BOS/EOS by zeroing first/last valid token per sequence is fiddly;
    # mean over all non-pad tokens (incl. BOS/EOS) is standard practice and stable.
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return pooled.float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/mibig_proteins.parquet")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--batch-tokens", type=int, default=6000, help="approx tokens per batch")
    ap.add_argument("--max-proteins-per-bgc", type=int, default=60)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model}")

    df = pd.read_parquet(args.input)
    df["translation"] = df["translation"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
    df = df[df["translation"].str.len() >= 10].copy()
    df["translation"] = df["translation"].str.slice(0, MAX_AA)

    # cap proteins/BGC so a handful of mega-clusters don't dominate wall-clock
    df = (
        df.sort_values(["bgc_id", "aa_length"], ascending=[True, False])
        .groupby("bgc_id")
        .head(args.max_proteins_per_bgc)
        .reset_index(drop=True)
    )
    print(f"Embedding {len(df)} protein sequences across {df['bgc_id'].nunique()} BGCs")

    tok, model = load_model(args.model, device)

    # length-sorted dynamic batching to minimize padding waste
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
        emb = embed_batch(batch_seqs, tok, model, device)
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
    print(f"Done embedding proteins in {time.time() - t0:.0f}s")

    df_sorted["_emb_idx"] = range(len(df_sorted))
    prot_dim = all_embeds.shape[1]

    bgc_ids = sorted(df_sorted["bgc_id"].unique())
    bgc_embeds = np.zeros((len(bgc_ids), prot_dim), dtype=np.float32)
    id_to_pos = {b: i for i, b in enumerate(bgc_ids)}
    sums = np.zeros((len(bgc_ids), prot_dim), dtype=np.float64)
    counts = np.zeros(len(bgc_ids), dtype=np.int64)
    for row_idx, bgc_id in zip(df_sorted["_emb_idx"].to_numpy(), df_sorted["bgc_id"].to_numpy()):
        pos = id_to_pos[bgc_id]
        sums[pos] += all_embeds[row_idx]
        counts[pos] += 1
    bgc_embeds = (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "esm_embeddings.npy", bgc_embeds)
    pd.DataFrame({"bgc_id": bgc_ids, "n_proteins_embedded": counts}).to_csv(
        outdir / "esm_bgc_ids.csv", index=False
    )
    print(f"Wrote {bgc_embeds.shape} embedding matrix -> {outdir / 'esm_embeddings.npy'}")


if __name__ == "__main__":
    main()
