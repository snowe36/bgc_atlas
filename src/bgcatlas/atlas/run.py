"""Map biosynthetic space with PCA (UMAP if available)."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bgcatlas.paths import FIGURES, PROCESSED, ensure_dirs

LOG = logging.getLogger(__name__)

CLASS_ORDER = ["NRPS", "PKS", "RiPP", "terpene", "hybrid", "other"]


def _embed_2d(X: np.ndarray) -> tuple[np.ndarray, str]:
    """Return 2D coords and method name. Prefer UMAP; fall back to PCA."""
    Xs = StandardScaler(with_mean=False).fit_transform(X)
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="euclidean", random_state=42)
        coords = reducer.fit_transform(Xs)
        return coords, "UMAP"
    except Exception as exc:  # noqa: BLE001
        LOG.warning("UMAP unavailable (%s); using PCA for 2D map", exc)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Xs)
        return coords, "PCA"


def run_atlas() -> pd.DataFrame:
    ensure_dirs()
    X = np.load(PROCESSED / "feature_matrix.npy")
    meta = pd.read_parquet(PROCESSED / "feature_meta.parquet")

    # high-D PCA for downstream novelty (saved here for reuse)
    Xs = StandardScaler(with_mean=False).fit_transform(X)
    n_comp = min(50, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    Z = pca.fit_transform(Xs)
    np.save(PROCESSED / "pca_embedding.npy", Z)
    LOG.info(
        "PCA %d-D explains %.1f%% variance",
        n_comp,
        100 * pca.explained_variance_ratio_.sum(),
    )

    coords, method = _embed_2d(X)
    atlas = meta.copy()
    atlas["dim1"] = coords[:, 0]
    atlas["dim2"] = coords[:, 1]
    atlas["embed_method"] = method
    atlas.to_parquet(PROCESSED / "atlas_coords.parquet", index=False)
    atlas.to_csv(PROCESSED / "atlas_coords.csv", index=False)

    # class-colored scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = atlas.copy()
    plot_df["biosynth_class"] = pd.Categorical(
        plot_df["biosynth_class"], categories=CLASS_ORDER, ordered=True
    )
    sns.scatterplot(
        data=plot_df,
        x="dim1",
        y="dim2",
        hue="biosynth_class",
        s=18,
        alpha=0.75,
        ax=ax,
        palette="tab10",
    )
    ax.set_title(f"Biosynthetic space ({method})")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "atlas_by_class.png", dpi=150)
    plt.close(fig)

    # density by class facets
    g = sns.FacetGrid(
        plot_df.dropna(subset=["biosynth_class"]),
        col="biosynth_class",
        col_wrap=3,
        sharex=True,
        sharey=True,
        height=2.6,
    )
    g.map_dataframe(sns.scatterplot, x="dim1", y="dim2", s=10, alpha=0.6, color="steelblue")
    g.figure.suptitle(f"Per-class neighborhoods ({method})", y=1.02)
    g.figure.savefig(FIGURES / "atlas_class_facets.png", dpi=150, bbox_inches="tight")
    plt.close(g.figure)

    LOG.info("Wrote atlas figures to %s", FIGURES)
    return atlas
