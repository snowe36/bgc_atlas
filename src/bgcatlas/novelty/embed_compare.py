"""How much does the novelty ranking change if we score in ESM2 space instead
of (or in addition to) the hashed-architecture space?

Reuses the same leave-one-out kNN novelty definition as `novelty/run.py`, just
applied to three different PCA embeddings, then compares the resulting
rankings via rank correlation and top-decile overlap. This is a robustness
check on the *headline* novelty ranking: if it changes wildly with the
representation, the ranking is representation-specific, not a property of
the BGCs themselves.
"""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bgcatlas.config import DEFAULT_NOVELTY_K, PCA_N_COMPONENTS
from bgcatlas.models.ablation import _load_aligned
from bgcatlas.novelty.run import score_novelty
from bgcatlas.paths import FIGURES, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)


def _pca_novelty(X: np.ndarray, standardize_with_mean: bool, k: int = DEFAULT_NOVELTY_K) -> np.ndarray:
    Xs = StandardScaler(with_mean=standardize_with_mean).fit_transform(X)
    n_comp = min(PCA_N_COMPONENTS, Xs.shape[0] - 1, Xs.shape[1])
    Z = PCA(n_components=n_comp, random_state=42).fit_transform(Xs)
    return score_novelty(Z, k=k)["novelty"]


def _class_stratified_disagreement(
    out: pd.DataFrame, top_frac: float
) -> tuple[list[dict], pd.DataFrame]:
    """Per-class Spearman + top-decile overlap for hashed vs ESM2 novelty."""
    rows: list[dict] = []
    for cls, sub in out.groupby("biosynth_class"):
        if len(sub) < 10:
            continue
        rho, pval = spearmanr(sub["novelty_hashed"], sub["novelty_esm"])
        n_top = max(1, int(round(top_frac * len(sub))))
        top_h = set(sub.nlargest(n_top, "novelty_hashed")["bgc_id"])
        top_e = set(sub.nlargest(n_top, "novelty_esm")["bgc_id"])
        union = top_h | top_e
        jaccard = len(top_h & top_e) / len(union) if union else 0.0
        rows.append(
            {
                "biosynth_class": cls,
                "n": int(len(sub)),
                "spearman_hashed_vs_esm": float(rho),
                "spearman_pvalue": float(pval),
                "top_decile_jaccard": float(jaccard),
                "mean_novelty_hashed": float(sub["novelty_hashed"].mean()),
                "mean_novelty_esm": float(sub["novelty_esm"].mean()),
            }
        )
    by_class = pd.DataFrame(rows).sort_values("spearman_hashed_vs_esm")
    return rows, by_class


def run_novelty_representation_comparison(
    k: int = DEFAULT_NOVELTY_K, top_frac: float = 0.1
) -> dict:
    ensure_dirs()
    meta, X_hash, X_esm, esm_label = _load_aligned()
    X_combined = np.hstack(
        [
            StandardScaler(with_mean=False).fit_transform(X_hash),
            StandardScaler().fit_transform(X_esm),
        ]
    )

    novelty_hash = _pca_novelty(X_hash, standardize_with_mean=False, k=k)
    novelty_esm = _pca_novelty(X_esm, standardize_with_mean=True, k=k)
    novelty_combined = _pca_novelty(X_combined, standardize_with_mean=False, k=k)

    out = meta[["bgc_id", "organism", "biosynth_class"]].copy()
    out["novelty_hashed"] = novelty_hash
    out["novelty_esm"] = novelty_esm
    out["novelty_combined"] = novelty_combined

    n_top = max(1, int(round(top_frac * len(out))))

    def _top_set(col: str) -> set[str]:
        return set(out.nlargest(n_top, col)["bgc_id"])

    top_hash, top_esm, top_combined = _top_set("novelty_hashed"), _top_set("novelty_esm"), _top_set("novelty_combined")

    def _jaccard(a: set, b: set) -> float:
        return len(a & b) / len(a | b) if (a | b) else 0.0

    rho_hash_esm, p_hash_esm = spearmanr(novelty_hash, novelty_esm)
    rho_hash_combined, p_hash_combined = spearmanr(novelty_hash, novelty_combined)
    by_class_rows, by_class_df = _class_stratified_disagreement(out, top_frac=top_frac)

    audit = {
        "n_bgcs": int(len(out)),
        "k": k,
        "top_frac": top_frac,
        "n_top": n_top,
        "esm_representation": esm_label,
        "spearman_hashed_vs_esm": float(rho_hash_esm),
        "spearman_hashed_vs_esm_pvalue": float(p_hash_esm),
        "spearman_hashed_vs_combined": float(rho_hash_combined),
        "spearman_hashed_vs_combined_pvalue": float(p_hash_combined),
        "top_decile_jaccard_hashed_vs_esm": _jaccard(top_hash, top_esm),
        "top_decile_jaccard_hashed_vs_combined": _jaccard(top_hash, top_combined),
        "top_decile_jaccard_esm_vs_combined": _jaccard(top_esm, top_combined),
        "by_class": by_class_rows,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "novelty_representation_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    out.sort_values("novelty_combined", ascending=False).to_csv(
        REPORTS / "novelty_representation_comparison.csv", index=False
    )
    by_class_df.to_csv(REPORTS / "novelty_disagreement_by_class.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sns.scatterplot(data=out, x="novelty_hashed", y="novelty_esm", hue="biosynth_class", s=14, alpha=0.6, ax=axes[0])
    axes[0].set_title(f"Hashed vs. ESM2 novelty (Spearman ρ={rho_hash_esm:.2f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    sns.scatterplot(
        data=out, x="novelty_hashed", y="novelty_combined", hue="biosynth_class", s=14, alpha=0.6, ax=axes[1]
    )
    axes[1].set_title(f"Hashed vs. combined novelty (Spearman ρ={rho_hash_combined:.2f})")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    for ax in axes:
        ax.set_xlabel("novelty (hashed architecture)")
        ax.legend(fontsize=7, title_fontsize=7)
    axes[0].set_ylabel("novelty (ESM2)")
    axes[1].set_ylabel("novelty (combined)")
    fig.tight_layout()
    fig.savefig(FIGURES / "novelty_representation_comparison.png", dpi=150)
    plt.close(fig)

    if len(by_class_df):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        plot_df = by_class_df.sort_values("spearman_hashed_vs_esm")
        sns.barplot(
            data=plot_df,
            x="biosynth_class",
            y="spearman_hashed_vs_esm",
            hue="biosynth_class",
            legend=False,
            ax=ax,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Spearman ρ (hashed vs ESM2 novelty)")
        ax.set_xlabel("")
        ax.set_title("Class-stratified novelty ranking disagreement")
        fig.tight_layout()
        fig.savefig(FIGURES / "novelty_disagreement_by_class.png", dpi=150)
        plt.close(fig)

    LOG.info(
        "Novelty representation comparison: hashed-vs-ESM rho=%.3f, top-decile Jaccard=%.3f",
        rho_hash_esm,
        audit["top_decile_jaccard_hashed_vs_esm"],
    )
    return audit
