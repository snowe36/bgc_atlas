"""Build BGC architecture feature matrices from domain annotations."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from npdiscovery.paths import PROCESSED, ensure_dirs

LOG = logging.getLogger(__name__)

N_HASH_DIMS = 256
MIN_DOMAIN_FREQ = 3


def _hash_token(token: str, n_dims: int = N_HASH_DIMS) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_dims


def build_feature_matrix(
    bgcs: pd.DataFrame,
    domains: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (bgc metadata aligned, X dense matrix, feature names)."""
    dom = domains[domains["feature_type"] == "domain"].copy()
    dom = dom[dom["domain_id"].astype(str).str.len() > 0]

    # domain counts per BGC
    counts = (
        dom.groupby(["bgc_id", "domain_id"]).size().unstack(fill_value=0)
        if len(dom)
        else pd.DataFrame(index=bgcs["bgc_id"])
    )

    # drop rare domains
    if len(counts.columns):
        freq = (counts > 0).sum(axis=0)
        keep = freq[freq >= MIN_DOMAIN_FREQ].index
        counts = counts.reindex(columns=keep, fill_value=0)

    # ordered architecture hashed bigrams
    arch_hash = np.zeros((len(bgcs), N_HASH_DIMS), dtype=np.float32)
    id_to_idx = {b: i for i, b in enumerate(bgcs["bgc_id"].tolist())}

    ordered = (
        dom.sort_values(["bgc_id", "gene_order"])
        .groupby("bgc_id")["domain_id"]
        .apply(list)
        if len(dom)
        else pd.Series(dtype=object)
    )
    for bgc_id, seq in ordered.items():
        if bgc_id not in id_to_idx:
            continue
        i = id_to_idx[bgc_id]
        for tok in seq:
            arch_hash[i, _hash_token(f"uni::{tok}")] += 1.0
        for a, b in zip(seq, seq[1:]):
            arch_hash[i, _hash_token(f"bi::{a}::{b}")] += 1.0

    # size / composition numeric features
    size_cols = []
    size_mat = []
    for col, default in [
        ("n_genes", 0),
        ("cluster_nt_length", 0),
        ("mean_aa_length", 0.0),
        ("total_aa_length", 0),
        ("n_domain_annotations", 0),
        ("n_compounds", 0),
    ]:
        if col in bgcs.columns:
            vals = pd.to_numeric(bgcs[col], errors="coerce").fillna(default).to_numpy(
                dtype=np.float32
            )
        else:
            vals = np.full(len(bgcs), float(default), dtype=np.float32)
        size_cols.append(f"size::{col}")
        size_mat.append(vals)
    size_arr = np.column_stack(size_mat) if size_mat else np.zeros((len(bgcs), 0))

    # align domain counts to bgc order
    counts = counts.reindex(bgcs["bgc_id"].tolist(), fill_value=0)
    dom_names = [f"dom::{c}" for c in counts.columns.astype(str)]
    hash_names = [f"arch_hash::{i}" for i in range(N_HASH_DIMS)]

    X = np.hstack(
        [
            counts.to_numpy(dtype=np.float32),
            arch_hash,
            size_arr,
        ]
    )
    feature_names = dom_names + hash_names + size_cols
    LOG.info(
        "Feature matrix: %d BGCs × %d features (%d domains, %d hash, %d size)",
        X.shape[0],
        X.shape[1],
        len(dom_names),
        N_HASH_DIMS,
        len(size_cols),
    )
    return bgcs.reset_index(drop=True), X, feature_names


def run_featurize() -> None:
    ensure_dirs()
    bgcs = pd.read_parquet(PROCESSED / "mibig_bgcs.parquet")
    domains = pd.read_parquet(PROCESSED / "mibig_domains.parquet")

    # keep BGCs with sequence evidence
    if "n_genes" in bgcs.columns:
        bgcs = bgcs[bgcs["n_genes"].fillna(0) > 0].copy()
    LOG.info("Featurizing %d BGCs with gene annotations", len(bgcs))

    meta, X, feature_names = build_feature_matrix(bgcs, domains)

    np.save(PROCESSED / "feature_matrix.npy", X)
    meta.to_parquet(PROCESSED / "feature_meta.parquet", index=False)
    meta.to_csv(PROCESSED / "feature_meta.csv", index=False)
    pd.Series(feature_names, name="feature").to_csv(
        PROCESSED / "feature_names.csv", index=False
    )
    LOG.info("Wrote features to %s", PROCESSED)
