"""Score biosynthetic novelty via leave-one-out kNN distance and density."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from bgcatlas.atlas.run import _annotate_offframe, _robust_limits
from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)


def score_novelty(Z: np.ndarray, k: int = 5) -> dict[str, np.ndarray]:
    """Leave-one-out kNN novelty in embedding space Z."""
    k_eff = min(k, max(1, len(Z) - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(Z)
    dists, idxs = nn.kneighbors(Z)
    # column 0 is self (distance ~0); use 1..k
    neighbor_dists = dists[:, 1 : k_eff + 1]
    neighbor_idxs = idxs[:, 1 : k_eff + 1]
    knn_mean = neighbor_dists.mean(axis=1)
    knn_kth = neighbor_dists[:, -1]
    nearest_idx = neighbor_idxs[:, 0]
    nearest_dist = neighbor_dists[:, 0]

    # rank-normalize to [0,1]
    def _rank01(x: np.ndarray) -> np.ndarray:
        order = x.argsort().argsort().astype(np.float64)
        if len(x) <= 1:
            return np.zeros_like(x, dtype=np.float64)
        return order / (len(x) - 1)

    novelty = 0.5 * _rank01(knn_mean) + 0.5 * _rank01(knn_kth)
    return {
        "knn_mean_dist": knn_mean,
        "knn_kth_dist": knn_kth,
        "nearest_dist": nearest_dist,
        "nearest_idx": nearest_idx.astype(int),
        "novelty": novelty,
    }


def run_novelty(k: int = 5) -> pd.DataFrame:
    ensure_dirs()
    Z = np.load(PROCESSED / "pca_embedding.npy")
    meta = pd.read_parquet(PROCESSED / "feature_meta.parquet")
    atlas = pd.read_parquet(PROCESSED / "atlas_coords.parquet")

    scores = score_novelty(Z, k=k)
    out = meta.copy()
    out["knn_mean_dist"] = scores["knn_mean_dist"]
    out["knn_kth_dist"] = scores["knn_kth_dist"]
    out["nearest_dist"] = scores["nearest_dist"]
    out["novelty"] = scores["novelty"]
    nearest_idx = scores["nearest_idx"]
    out["nearest_mibig"] = meta["bgc_id"].iloc[nearest_idx].to_numpy()
    out["neighbor_class"] = meta["biosynth_class"].iloc[nearest_idx].to_numpy()
    out = out.sort_values("novelty", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    cols = [
        "rank",
        "bgc_id",
        "organism",
        "biosynth_class",
        "novelty",
        "knn_mean_dist",
        "nearest_dist",
        "nearest_mibig",
        "neighbor_class",
        "compounds",
        "n_genes",
    ]
    cols = [c for c in cols if c in out.columns]
    ranking = out[cols]
    REPORTS.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(REPORTS / "novelty_ranking.csv", index=False)
    out.to_parquet(PROCESSED / "novelty_scores.parquet", index=False)
    LOG.info("Top-5 novel BGCs:\n%s", ranking.head(5).to_string(index=False))

    # overlay on atlas
    plot = atlas.merge(
        out[["bgc_id", "novelty"]], on="bgc_id", how="left"
    )
    thr = plot["novelty"].quantile(0.9)
    plot["high_novelty"] = plot["novelty"] >= thr

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot[~plot["high_novelty"]],
        x="dim1",
        y="dim2",
        color="lightgray",
        s=14,
        ax=ax,
        label="known neighborhood",
    )
    sns.scatterplot(
        data=plot[plot["high_novelty"]],
        x="dim1",
        y="dim2",
        color="crimson",
        s=28,
        ax=ax,
        label="top-decile novelty",
    )
    ax.set_title("Unexplored regions of biosynthetic space")
    ax.set_xlabel("embed-1")
    ax.set_ylabel("embed-2")
    ax.legend()
    coords = plot[["dim1", "dim2"]].dropna().to_numpy()
    xlim, ylim = _robust_limits(coords[:, 0]), _robust_limits(coords[:, 1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _annotate_offframe(ax, coords, xlim, ylim)
    fig.tight_layout()
    fig.savefig(FIGURES / "novelty_overlay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=out,
        x="biosynth_class",
        y="novelty",
        hue="biosynth_class",
        legend=False,
        ax=ax,
    )
    ax.set_title("Novelty by biosynth class")
    ax.set_xlabel("class")
    ax.set_ylabel("novelty score")
    fig.tight_layout()
    fig.savefig(FIGURES / "novelty_by_class.png", dpi=150)
    plt.close(fig)

    LOG.info("Wrote novelty ranking → %s", REPORTS / "novelty_ranking.csv")
    return ranking
