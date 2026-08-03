"""Conditional capability expectations: P(capacity | architecture neighborhood)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from bgcatlas.analyses.capability_gap.config import CAPABILITIES, NEIGHBORHOOD_K, SIZE_MAX
from bgcatlas.analyses.myxo_grammar.cohort import assign_clade
from bgcatlas.analyses.myxo_grammar.tokens import NOISE_DOMAINS


def domain_sets_by_bgc(domains: pd.DataFrame) -> dict[str, set[str]]:
    dom = domains[domains["feature_type"] == "domain"].copy()
    dom = dom[dom["domain_id"].astype(str).str.len() > 0]
    dom = dom[~dom["domain_id"].isin(NOISE_DOMAINS)]
    out: dict[str, set[str]] = {}
    for bgc_id, g in dom.groupby("bgc_id"):
        out[str(bgc_id)] = set(map(str, g["domain_id"].tolist()))
    return out


def has_capacity(dom_set: set[str], capacity: str) -> bool:
    return bool(dom_set & CAPABILITIES[capacity])


def has_any(dom_set: set[str], options: set[str]) -> bool:
    return bool(dom_set & options)


def build_inventory(
    ranking: pd.DataFrame,
    domains: pd.DataFrame,
    *,
    size_max: int | None = SIZE_MAX,
    mask_capacity: str | None = None,
    mask_bgc_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Per-BGC inventory of capabilities (optionally masking a capacity on some BGCs)."""
    dom_sets = domain_sets_by_bgc(domains)
    df = ranking[["bgc_id", "organism", "biosynth_class", "novelty", "n_genes"]].copy()
    df["bgc_id"] = df["bgc_id"].astype(str)
    df["n_genes"] = pd.to_numeric(df["n_genes"], errors="coerce")
    if size_max is not None:
        df = df[df["n_genes"] <= size_max].copy()
    df["clade"] = df["organism"].map(assign_clade)

    rows = []
    for r in df.itertuples(index=False):
        bid = str(r.bgc_id)
        dset = set(dom_sets.get(bid, set()))
        if mask_capacity and mask_bgc_ids and bid in mask_bgc_ids:
            dset = dset - CAPABILITIES[mask_capacity]
        row = {
            "bgc_id": bid,
            "organism": r.organism,
            "biosynth_class": r.biosynth_class,
            "novelty": float(r.novelty),
            "n_genes": float(r.n_genes),
            "clade": r.clade,
        }
        for cap in CAPABILITIES:
            row[f"has_{cap}"] = int(has_capacity(dset, cap))
        rows.append(row)
    return pd.DataFrame(rows)


def attach_neighborhood_expectations(
    inventory: pd.DataFrame,
    coords: pd.DataFrame,
    *,
    k: int = NEIGHBORHOOD_K,
    capacities: list[str] | None = None,
) -> pd.DataFrame:
    """Estimate P(capacity | architecture neighborhood) via atlas kNN.

    Neighborhood is conditional on local architecture geometry (atlas coords),
    not a global prior like \"PKSs usually have an ER.\"
    """
    caps = capacities or list(CAPABILITIES)
    xy = coords[["bgc_id", "dim1", "dim2"]].copy()
    xy["bgc_id"] = xy["bgc_id"].astype(str)
    df = inventory.merge(xy, on="bgc_id", how="inner")
    if len(df) < k + 2:
        raise ValueError(f"Need >{k} BGCs with coords; got {len(df)}")

    X = df[["dim1", "dim2"]].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(df)), metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X)

    out = df.copy()
    for cap in caps:
        col = f"has_{cap}"
        probs = []
        for i in range(len(df)):
            neigh = idx[i, 1:]  # drop self
            probs.append(float(df.iloc[neigh][col].mean()))
        out[f"p_{cap}"] = probs
        out[f"gap_{cap}"] = np.maximum(0.0, out[f"p_{cap}"] - out[col])
    return out
