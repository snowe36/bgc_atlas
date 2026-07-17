"""Validate novelty discovery strategy and write audit report."""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)


def run_validate() -> dict:
    ensure_dirs()
    meta = pd.read_parquet(PROCESSED / "feature_meta.parquet")
    scores = pd.read_parquet(PROCESSED / "novelty_scores.parquet")
    names = pd.read_csv(PROCESSED / "feature_names.csv")["feature"].tolist()

    # Integrity: class labels must not appear as features
    forbidden = [n for n in names if "biosynth_class" in n.lower() or n.startswith("class::")]
    class_leak = len(forbidden) > 0

    # Size outliers among top novelty (possible contig/genome mis-parses)
    top = scores.nsmallest(50, "rank") if "rank" in scores.columns else scores.nlargest(50, "novelty")
    size_flag = top["n_genes"] > meta["n_genes"].quantile(0.99)
    n_size_outliers = int(size_flag.sum())

    # Stratified novelty summary
    by_class = (
        scores.groupby("biosynth_class")["novelty"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    # Same-class neighbor rate among top-decile novelty
    thr = scores["novelty"].quantile(0.9)
    high = scores[scores["novelty"] >= thr]
    same_class = (high["biosynth_class"] == high["neighbor_class"]).mean()

    # Correlation of novelty with n_genes (should not be the whole story)
    corr_genes = float(scores["novelty"].corr(scores["n_genes"]))

    audit = {
        "n_bgcs": int(len(scores)),
        "class_label_leak_in_features": class_leak,
        "forbidden_features": forbidden,
        "top50_size_outliers": n_size_outliers,
        "top_decile_same_class_neighbor_rate": float(same_class),
        "novelty_n_genes_spearman": corr_genes,
        "novelty_by_class": by_class.to_dict(orient="records"),
        "checks_passed": (not class_leak) and (n_size_outliers < 25),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "validation_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    by_class.to_csv(REPORTS / "novelty_by_class.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=by_class, x="biosynth_class", y="mean", hue="biosynth_class", legend=False, ax=ax)
    ax.set_title("Mean novelty by class (validation)")
    ax.set_ylabel("mean novelty")
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_novelty_by_class.png", dpi=150)
    plt.close(fig)

    LOG.info("Validation audit: %s", json.dumps({k: audit[k] for k in audit if k != "novelty_by_class"}))
    return audit
