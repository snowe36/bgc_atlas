"""Time-split holdout: pre-cutoff manifold vs post-cutoff vs size-matched control."""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from bgcatlas.config import DEFAULT_TEMPORAL_CUTOFF, MAJOR_FAMILIES, PCA_N_COMPONENTS
from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)

DEFAULT_CUTOFF = DEFAULT_TEMPORAL_CUTOFF


def _fit_reference(X_ref: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, PCA, StandardScaler]:
    scaler = StandardScaler(with_mean=False).fit(X_ref)
    Xs_ref = scaler.transform(X_ref)
    n_comp = min(PCA_N_COMPONENTS, Xs_ref.shape[0] - 1, Xs_ref.shape[1])
    pca = PCA(n_components=n_comp, random_state=42).fit(Xs_ref)
    Z_ref = pca.transform(Xs_ref)

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Z_ref)), metric="euclidean").fit(Z_ref)
    dists, _ = nn.kneighbors(Z_ref)
    ref_knn_mean = dists[:, 1:].mean(axis=1)  # drop self
    return Z_ref, ref_knn_mean, pca, scaler


def _score_query(Z_ref: np.ndarray, ref_knn_mean: np.ndarray, Z_query: np.ndarray, k: int) -> np.ndarray:
    """Percentile of query kNN-mean-distance within the reference's own kNN distance distribution."""
    nn = NearestNeighbors(n_neighbors=min(k, len(Z_ref)), metric="euclidean").fit(Z_ref)
    dists, _ = nn.kneighbors(Z_query)
    query_knn_mean = dists.mean(axis=1)
    return np.array([(ref_knn_mean < d).mean() for d in query_knn_mean], dtype=float)


def _run_holdout_on_indices(
    X: np.ndarray,
    meta: pd.DataFrame,
    ref_idx: np.ndarray,
    held_idx: np.ndarray,
    k: int,
    n_controls: int,
    seed: int,
) -> dict:
    """Core temporal holdout math for a fixed reference / held-out index split."""
    if len(held_idx) < 5:
        raise RuntimeError(f"Too few held-out BGCs ({len(held_idx)}) for temporal holdout")
    if len(ref_idx) < max(20, k + 2):
        raise RuntimeError(f"Too few reference BGCs ({len(ref_idx)}) for temporal holdout")

    X_ref, X_held = X[ref_idx], X[held_idx]
    Z_ref, ref_knn_mean, _, _ = _fit_reference(X_ref, k=k)
    scaler = StandardScaler(with_mean=False).fit(X_ref)
    pca = PCA(n_components=Z_ref.shape[1], random_state=42).fit(scaler.transform(X_ref))
    Z_held = pca.transform(scaler.transform(X_held))

    held_novelty = _score_query(Z_ref, ref_knn_mean, Z_held, k=k)

    rng = np.random.default_rng(seed)
    control_means = []
    for _ in range(n_controls):
        perm = rng.permutation(len(ref_idx))
        n_ctrl = min(len(held_idx), len(ref_idx) // 2)
        if n_ctrl < 5:
            break
        ctrl_pos = perm[:n_ctrl]
        rest_pos = perm[n_ctrl:]
        Z_rest, rest_knn_mean, _, _ = _fit_reference(X_ref[rest_pos], k=k)
        scaler_c = StandardScaler(with_mean=False).fit(X_ref[rest_pos])
        pca_c = PCA(n_components=Z_rest.shape[1], random_state=42).fit(scaler_c.transform(X_ref[rest_pos]))
        Z_ctrl = pca_c.transform(scaler_c.transform(X_ref[ctrl_pos]))
        ctrl_novelty = _score_query(Z_rest, rest_knn_mean, Z_ctrl, k=k)
        control_means.append(float(np.mean(ctrl_novelty)))
    control_means = np.array(control_means) if control_means else np.array([0.5])

    perm = rng.permutation(len(ref_idx))
    n_ctrl = min(len(held_idx), len(ref_idx) // 2)
    ctrl_pos, rest_pos = perm[:n_ctrl], perm[n_ctrl:]
    Z_rest, rest_knn_mean, _, _ = _fit_reference(X_ref[rest_pos], k=k)
    scaler_c = StandardScaler(with_mean=False).fit(X_ref[rest_pos])
    pca_c = PCA(n_components=Z_rest.shape[1], random_state=42).fit(scaler_c.transform(X_ref[rest_pos]))
    Z_ctrl_plot = pca_c.transform(scaler_c.transform(X_ref[ctrl_pos]))
    control_novelty_sample = _score_query(Z_rest, rest_knn_mean, Z_ctrl_plot, k=k)

    u_stat, p_value = mannwhitneyu(held_novelty, control_means, alternative="greater")

    meta_held = meta.iloc[held_idx].reset_index(drop=True).copy()
    meta_held["temporal_novelty"] = held_novelty
    by_class = (
        meta_held.groupby("biosynth_class")["temporal_novelty"].agg(["count", "mean", "median"]).reset_index()
    )

    return {
        "n_reference": int(len(ref_idx)),
        "n_heldout": int(len(held_idx)),
        "heldout_novelty_mean": float(held_novelty.mean()),
        "heldout_novelty_median": float(np.median(held_novelty)),
        "random_control_novelty_mean_of_means": float(control_means.mean()),
        "random_control_novelty_std_of_means": float(control_means.std()),
        "n_random_controls": int(len(control_means)),
        "mann_whitney_u": float(u_stat),
        "p_value_heldout_gt_control": float(p_value),
        "heldout_more_novel_than_random_control": bool(
            held_novelty.mean() > control_means.mean() and p_value < 0.05
        ),
        "heldout_novelty_by_class": by_class.to_dict(orient="records"),
        "_held_novelty": held_novelty,
        "_control_novelty_sample": control_novelty_sample,
        "_meta_held": meta_held,
    }


def run_temporal_holdout(
    cutoff: str = DEFAULT_CUTOFF,
    k: int = 5,
    n_controls: int = 50,
    seed: int = 42,
) -> dict:
    """Pre-cutoff fit, post-cutoff score; also reports non-major-family subset."""
    ensure_dirs()
    X = np.load(PROCESSED / "feature_matrix.npy")
    meta = pd.read_parquet(PROCESSED / "feature_meta.parquet")

    if "date_added" not in meta.columns or meta["date_added"].isna().all():
        raise RuntimeError("No date_added metadata found; re-run bgc-download to regenerate it.")

    is_held = meta["date_added"].fillna("9999") >= cutoff
    ref_idx = np.where(~is_held.to_numpy())[0]
    held_idx = np.where(is_held.to_numpy())[0]
    LOG.info(
        "Temporal holdout @ %s: %d reference BGCs, %d held-out (post-cutoff) BGCs",
        cutoff,
        len(ref_idx),
        len(held_idx),
    )

    core = _run_holdout_on_indices(X, meta, ref_idx, held_idx, k, n_controls, seed)
    held_novelty = core.pop("_held_novelty")
    control_novelty_sample = core.pop("_control_novelty_sample")
    meta_held = core.pop("_meta_held")
    p_value = core["p_value_heldout_gt_control"]

    major = set(MAJOR_FAMILIES)
    keep = ~meta["biosynth_class"].isin(major)
    subset_ref = np.where((~is_held & keep).to_numpy())[0]
    subset_held = np.where((is_held & keep).to_numpy())[0]
    non_major: dict | None = None
    if len(subset_held) >= 5 and len(subset_ref) >= 20:
        try:
            nm = _run_holdout_on_indices(X, meta, subset_ref, subset_held, k, n_controls, seed + 1)
            nm.pop("_held_novelty", None)
            nm.pop("_control_novelty_sample", None)
            nm.pop("_meta_held", None)
            nm["excluded_classes"] = list(MAJOR_FAMILIES)
            non_major = nm
            LOG.info(
                "Non-major-family temporal: heldout mean=%.3f vs control=%.3f (p=%.4g)",
                nm["heldout_novelty_mean"],
                nm["random_control_novelty_mean_of_means"],
                nm["p_value_heldout_gt_control"],
            )
        except RuntimeError as exc:
            LOG.warning("Non-major-family temporal subset skipped: %s", exc)

    audit = {
        "cutoff_date": cutoff,
        "k": k,
        **core,
        "non_major_family_subset": non_major,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "temporal_holdout.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    meta_held.sort_values("temporal_novelty", ascending=False)[
        ["bgc_id", "organism", "biosynth_class", "date_added", "temporal_novelty"]
    ].to_csv(REPORTS / "temporal_holdout_ranking.csv", index=False)

    plot_df = pd.concat(
        [
            pd.DataFrame(
                {"novelty": control_novelty_sample, "group": "random control\n(held from reference)"}
            ),
            pd.DataFrame({"novelty": held_novelty, "group": f"post-{cutoff}\n(true temporal holdout)"}),
        ]
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.boxplot(data=plot_df, x="group", y="novelty", hue="group", legend=False, ax=ax)
    sns.stripplot(data=plot_df, x="group", y="novelty", color="black", size=3, alpha=0.35, ax=ax)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="null (0.5)")
    ax.set_title(f"Prospective novelty: random vs. true post-cutoff holdout\n(Mann-Whitney p={p_value:.3g})")
    ax.set_ylabel("novelty percentile vs. reference manifold")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "temporal_holdout.png", dpi=150)
    plt.close(fig)

    LOG.info(
        "Temporal holdout: post-cutoff mean novelty=%.3f vs random-control mean=%.3f (p=%.4g)",
        audit["heldout_novelty_mean"],
        audit["random_control_novelty_mean_of_means"],
        p_value,
    )
    return audit
