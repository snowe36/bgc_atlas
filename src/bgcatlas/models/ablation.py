"""Class-recovery F1: hashed vs ESM vs combined. Needs esm_embeddings.npy."""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)


def _esm_representation_label() -> str:
    """Label for the active ESM matrix (from V2 manifest when present)."""
    man_path = PROCESSED / "esm_embed_manifest.json"
    if man_path.exists():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
            return str(man.get("representation_label") or man.get("model") or "esm")
        except (OSError, json.JSONDecodeError):
            pass
    return "esm2"


def load_aligned_representation(
    emb_path=None,
    ids_path=None,
    label: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Align hashed features with a BGC embedding matrix by bgc_id.

    Returns (meta, X_hash, X_emb).
    """
    from pathlib import Path

    emb_path = Path(emb_path) if emb_path is not None else PROCESSED / "esm_embeddings.npy"
    ids_path = Path(ids_path) if ids_path is not None else PROCESSED / "esm_bgc_ids.csv"
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"{emb_path.name} / {ids_path.name} not found under {emb_path.parent}. "
            "Produce embeddings first (scripts/run_esm_embed.py or bgc-train-encoder)."
        )

    X_hash_full = np.load(PROCESSED / "feature_matrix.npy")
    meta_full = pd.read_parquet(PROCESSED / "feature_meta.parquet")
    X_emb_full = np.load(emb_path)
    emb_ids = pd.read_csv(ids_path)

    meta_full = meta_full.reset_index(drop=True)
    meta_full["_row"] = np.arange(len(meta_full))
    emb_ids = emb_ids.reset_index(drop=True)
    emb_ids["_emb_row"] = np.arange(len(emb_ids))

    merged = meta_full.merge(emb_ids[["bgc_id", "_emb_row"]], on="bgc_id", how="inner")
    tag = label or emb_path.stem
    LOG.info(
        "Aligned: %d / %d BGCs have both hashed features and %s embeddings",
        len(merged),
        len(meta_full),
        tag,
    )
    X_hash = X_hash_full[merged["_row"].to_numpy()]
    X_emb = X_emb_full[merged["_emb_row"].to_numpy()]
    meta = merged.drop(columns=["_row", "_emb_row"]).reset_index(drop=True)
    return meta, X_hash, X_emb


def _load_aligned() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    """Return (meta, X_hash, X_esm, esm_label) for BGCs present in both representations."""
    esm_label = _esm_representation_label()
    meta, X_hash, X_esm = load_aligned_representation(
        emb_path=PROCESSED / "esm_embeddings.npy",
        ids_path=PROCESSED / "esm_bgc_ids.csv",
        label=esm_label,
    )
    return meta, X_hash, X_esm, esm_label


def _cv_benchmark(X: np.ndarray, y: np.ndarray, n_splits: int) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
    }
    best = {"model": None, "macro_f1": -1.0, "weighted_f1": -1.0}
    for name, model in models.items():
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        macro = f1_score(y, pred, average="macro")
        weighted = f1_score(y, pred, average="weighted")
        if macro > best["macro_f1"]:
            best = {"model": name, "macro_f1": float(macro), "weighted_f1": float(weighted)}
    return best


def run_ablation(n_splits: int = 5) -> dict:
    ensure_dirs()
    meta, X_hash, X_esm, esm_label = _load_aligned()
    y = meta["biosynth_class"].astype(str).to_numpy()

    counts = pd.Series(y).value_counts()
    keep = counts[counts >= n_splits].index
    mask = np.isin(y, keep)
    X_hash, X_esm, y = X_hash[mask], X_esm[mask], y[mask]
    LOG.info("Ablation classification on %d BGCs; classes: %s", len(y), sorted(set(y)))

    X_esm_scaled = StandardScaler().fit_transform(X_esm)
    X_combined = np.hstack([StandardScaler(with_mean=False).fit_transform(X_hash), X_esm_scaled])

    variants = {
        "hashed_architecture": X_hash,
        esm_label: X_esm,
        f"combined_hashed+{esm_label}": X_combined,
    }
    results = {}
    for name, X in variants.items():
        results[name] = _cv_benchmark(X, y, n_splits=n_splits)
        LOG.info(
            "%-20s best=%s macro-F1=%.3f weighted-F1=%.3f",
            name,
            results[name]["model"],
            results[name]["macro_f1"],
            results[name]["weighted_f1"],
        )

    out = {
        "n_bgcs": int(len(y)),
        "n_splits": n_splits,
        "esm_representation": esm_label,
        "results": results,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "ablation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    rows = [
        {"representation": name, "macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"]}
        for name, r in results.items()
    ]
    rdf = pd.DataFrame(rows)
    plot_df = rdf.melt(id_vars="representation", var_name="metric", value_name="f1")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=plot_df, x="representation", y="f1", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Representation ablation (n={len(y)} BGCs, {n_splits}-fold CV)")
    ax.set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_representation_comparison.png", dpi=150)
    plt.close(fig)

    LOG.info("Wrote ablation metrics -> %s", REPORTS / "ablation_metrics.json")
    return out
