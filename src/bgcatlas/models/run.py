"""Benchmark architecture features via biosynth-class recovery."""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)


def run_sanity(n_splits: int = 5) -> dict:
    ensure_dirs()
    X = np.load(PROCESSED / "feature_matrix.npy")
    meta = pd.read_parquet(PROCESSED / "feature_meta.parquet")
    y = meta["biosynth_class"].astype(str).to_numpy()

    # drop ultra-rare classes for stable CV
    counts = pd.Series(y).value_counts()
    keep = counts[counts >= n_splits].index
    mask = np.isin(y, keep)
    X, y = X[mask], y[mask]
    meta = meta.loc[mask].reset_index(drop=True)
    LOG.info("Sanity classification on %d BGCs; classes: %s", len(y), sorted(set(y)))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    best_name, best_f1, best_pred = None, -1.0, None
    for name, model in models.items():
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        macro = f1_score(y, pred, average="macro")
        weighted = f1_score(y, pred, average="weighted")
        results[name] = {
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
            "report": classification_report(y, pred, output_dict=True),
        }
        LOG.info("%s macro-F1=%.3f weighted-F1=%.3f", name, macro, weighted)
        if macro > best_f1:
            best_name, best_f1, best_pred = name, macro, pred

    labels = sorted(set(y))
    cm = confusion_matrix(y, best_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    ax.set_title(f"Biosynth-class recovery ({best_name}, macro-F1={best_f1:.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES / "sanity_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # per-class F1 bar
    report = results[best_name]["report"]
    rows = [
        {"class": k, "f1": v["f1-score"], "support": v["support"]}
        for k, v in report.items()
        if isinstance(v, dict) and "f1-score" in v and k in labels
    ]
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=rdf, x="class", y="f1", hue="class", ax=ax, legend=False)
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class F1 ({best_name})")
    ax.set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(FIGURES / "sanity_per_class_f1.png", dpi=150)
    plt.close(fig)

    out = {
        "n_bgcs": int(len(y)),
        "n_splits": n_splits,
        "best_model": best_name,
        "results": results,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "sanity_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    LOG.info("Wrote sanity metrics → %s", REPORTS / "sanity_metrics.json")
    return out
