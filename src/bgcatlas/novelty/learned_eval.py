"""Honest evaluation of a learned BGC representation.

Runs the same checks as the hashed / ESM2 paths:
1. Class recovery (via generalized ablation loader)
2. Novelty-ranking agreement vs hashed / ESM2
3. Size-confound (Spearman novelty ↔ n_genes)
4. Prospective temporal holdout on the learned embedding matrix

Expects ``learned_embeddings.npy`` + ``learned_bgc_ids.csv`` (+ optional manifest)
under ``data/processed/``, produced by ``bgc-train-encoder``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bgcatlas.config import DEFAULT_NOVELTY_K, DEFAULT_TEMPORAL_CUTOFF, PCA_N_COMPONENTS
from bgcatlas.models.ablation import _cv_benchmark, load_aligned_representation
from bgcatlas.novelty.run import score_novelty
from bgcatlas.novelty.temporal import _run_holdout_on_indices
from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)


def _learned_label(processed: Path | None = None) -> str:
    processed = processed or PROCESSED
    man = processed / "learned_embed_manifest.json"
    if man.exists():
        try:
            return str(json.loads(man.read_text(encoding="utf-8")).get("representation_label") or "learned")
        except (OSError, json.JSONDecodeError):
            pass
    return "learned"


def _pca_novelty(X: np.ndarray, k: int = DEFAULT_NOVELTY_K) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(PCA_N_COMPONENTS, Xs.shape[0] - 1, Xs.shape[1])
    Z = PCA(n_components=n_comp, random_state=42).fit_transform(Xs)
    return score_novelty(Z, k=k)["novelty"]


def run_learned_class_recovery(n_splits: int = 5) -> dict:
    """Class-recovery bake-off: hashed / ESM / learned / combined variants."""
    ensure_dirs()
    label = _learned_label()
    meta, X_hash, X_learned = load_aligned_representation(
        emb_path=PROCESSED / "learned_embeddings.npy",
        ids_path=PROCESSED / "learned_bgc_ids.csv",
        label=label,
    )
    y = meta["biosynth_class"].astype(str).to_numpy()
    counts = pd.Series(y).value_counts()
    keep = counts[counts >= n_splits].index
    mask = np.isin(y, keep)
    X_hash, X_learned, y, meta = X_hash[mask], X_learned[mask], y[mask], meta.loc[mask].reset_index(drop=True)

    # Optional ESM if present
    variants: dict[str, np.ndarray] = {
        "hashed_architecture": X_hash,
        label: X_learned,
        f"combined_hashed+{label}": np.hstack(
            [
                StandardScaler(with_mean=False).fit_transform(X_hash),
                StandardScaler().fit_transform(X_learned),
            ]
        ),
    }
    esm_path = PROCESSED / "esm_embeddings.npy"
    esm_ids = PROCESSED / "esm_bgc_ids.csv"
    if esm_path.exists() and esm_ids.exists():
        meta_e, X_h2, X_esm = load_aligned_representation(emb_path=esm_path, ids_path=esm_ids, label="esm")
        # Align to learned meta via bgc_id
        merged = (
            meta[["bgc_id"]]
            .reset_index()
            .merge(
                meta_e[["bgc_id"]].reset_index().rename(columns={"index": "_e"}),
                on="bgc_id",
                how="inner",
            )
        )
        X_esm_al = X_esm[merged["_e"].to_numpy()]
        # Restrict hash/learned/y to the intersection
        idx = merged["index"].to_numpy()
        X_hash = X_hash[idx]
        X_learned = X_learned[idx]
        y = y[idx]
        variants = {
            "hashed_architecture": X_hash,
            "esm2": X_esm_al,
            label: X_learned,
            f"combined_hashed+{label}": np.hstack(
                [
                    StandardScaler(with_mean=False).fit_transform(X_hash),
                    StandardScaler().fit_transform(X_learned),
                ]
            ),
            f"combined_esm+{label}": np.hstack(
                [
                    StandardScaler().fit_transform(X_esm_al),
                    StandardScaler().fit_transform(X_learned),
                ]
            ),
        }

    results = {}
    for name, X in variants.items():
        results[name] = _cv_benchmark(X, y, n_splits=n_splits)
        LOG.info(
            "%-40s best=%s macro-F1=%.3f",
            name,
            results[name]["model"],
            results[name]["macro_f1"],
        )

    out = {
        "n_bgcs": int(len(y)),
        "n_splits": n_splits,
        "learned_representation": label,
        "results": results,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "learned_ablation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    rows = [
        {"representation": name, "macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"]}
        for name, r in results.items()
    ]
    rdf = pd.DataFrame(rows)
    plot_df = rdf.melt(id_vars="representation", var_name="metric", value_name="f1")
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(rows)), 4.5))
    sns.barplot(data=plot_df, x="representation", y="f1", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Learned-representation ablation (n={len(y)}, {n_splits}-fold CV)")
    ax.set_ylabel("F1")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "learned_ablation_comparison.png", dpi=150)
    plt.close(fig)
    return out


def run_learned_novelty_compare(k: int = DEFAULT_NOVELTY_K, top_frac: float = 0.1) -> dict:
    """Novelty ranking: hashed vs ESM vs learned + size-confound readout."""
    ensure_dirs()
    label = _learned_label()
    meta, X_hash, X_learned = load_aligned_representation(
        emb_path=PROCESSED / "learned_embeddings.npy",
        ids_path=PROCESSED / "learned_bgc_ids.csv",
        label=label,
    )
    # hashed features are sparse counts — match existing embed_compare convention
    from bgcatlas.novelty.embed_compare import _pca_novelty as _pca_nov_flagged

    novelty_hash = _pca_nov_flagged(X_hash, standardize_with_mean=False, k=k)
    novelty_learned = _pca_novelty(X_learned, k=k)

    out = meta[["bgc_id", "organism", "biosynth_class", "n_genes"]].copy()
    out["novelty_hashed"] = novelty_hash
    out["novelty_learned"] = novelty_learned

    esm_rho = None
    if (PROCESSED / "esm_embeddings.npy").exists():
        meta_e, _, X_esm = load_aligned_representation(
            emb_path=PROCESSED / "esm_embeddings.npy",
            ids_path=PROCESSED / "esm_bgc_ids.csv",
            label="esm",
        )
        merged = (
            out[["bgc_id"]]
            .reset_index()
            .merge(
                meta_e[["bgc_id"]].reset_index().rename(columns={"index": "_e"}),
                on="bgc_id",
                how="inner",
            )
        )
        novelty_esm_full = _pca_novelty(X_esm, k=k)
        out["novelty_esm"] = np.nan
        out.loc[merged["index"].to_numpy(), "novelty_esm"] = novelty_esm_full[merged["_e"].to_numpy()]
        sub = out.dropna(subset=["novelty_esm"])
        esm_rho, esm_p = spearmanr(sub["novelty_hashed"], sub["novelty_esm"])
        learned_esm_rho, learned_esm_p = spearmanr(sub["novelty_learned"], sub["novelty_esm"])
    else:
        esm_p = learned_esm_rho = learned_esm_p = None

    rho_hl, p_hl = spearmanr(out["novelty_hashed"], out["novelty_learned"])
    size_h, _ = spearmanr(out["novelty_hashed"], out["n_genes"])
    size_l, _ = spearmanr(out["novelty_learned"], out["n_genes"])

    n_top = max(1, int(round(top_frac * len(out))))
    top_h = set(out.nlargest(n_top, "novelty_hashed")["bgc_id"])
    top_l = set(out.nlargest(n_top, "novelty_learned")["bgc_id"])
    jaccard = len(top_h & top_l) / len(top_h | top_l) if (top_h | top_l) else 0.0

    audit = {
        "n_bgcs": int(len(out)),
        "k": k,
        "learned_representation": label,
        "spearman_hashed_vs_learned": float(rho_hl),
        "spearman_hashed_vs_learned_pvalue": float(p_hl),
        "top_decile_jaccard_hashed_vs_learned": float(jaccard),
        "size_confound_spearman_hashed": float(size_h),
        "size_confound_spearman_learned": float(size_l),
        "size_confound_delta": float(size_l - size_h),
    }
    if esm_rho is not None:
        audit["spearman_hashed_vs_esm"] = float(esm_rho)
        audit["spearman_hashed_vs_esm_pvalue"] = float(esm_p)
        audit["spearman_learned_vs_esm"] = float(learned_esm_rho)
        audit["spearman_learned_vs_esm_pvalue"] = float(learned_esm_p)

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "learned_novelty_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    out.sort_values("novelty_learned", ascending=False).to_csv(
        REPORTS / "learned_novelty_comparison.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sns.scatterplot(
        data=out,
        x="novelty_hashed",
        y="novelty_learned",
        hue="biosynth_class",
        s=14,
        alpha=0.6,
        ax=axes[0],
    )
    axes[0].set_title(f"Hashed vs learned novelty (ρ={rho_hl:.2f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("novelty (hashed)")
    axes[0].set_ylabel("novelty (learned)")
    axes[0].legend(fontsize=7, title_fontsize=7)

    size_df = pd.DataFrame(
        {
            "representation": ["hashed", "learned"],
            "spearman_novelty_vs_n_genes": [size_h, size_l],
        }
    )
    sns.barplot(
        data=size_df,
        x="representation",
        y="spearman_novelty_vs_n_genes",
        hue="representation",
        legend=False,
        ax=axes[1],
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Size confound (lower |ρ| is better)")
    axes[1].set_ylabel("Spearman(novelty, n_genes)")
    fig.tight_layout()
    fig.savefig(FIGURES / "learned_novelty_comparison.png", dpi=150)
    plt.close(fig)

    LOG.info(
        "Learned novelty: hashed-vs-learned ρ=%.3f; size confound %.3f → %.3f",
        rho_hl,
        size_h,
        size_l,
    )
    return audit


def run_learned_temporal_holdout(
    cutoff: str = DEFAULT_TEMPORAL_CUTOFF,
    k: int = DEFAULT_NOVELTY_K,
    n_controls: int = 50,
    seed: int = 42,
) -> dict:
    """Prospective temporal holdout in *learned* embedding space.

    Uses the same math as ``novelty/temporal.py`` but scores on
    ``learned_embeddings.npy``. For a leakage-free story the encoder should
    have been trained with ``--prospective`` / ``train_cutoff=cutoff``.
    """
    ensure_dirs()
    label = _learned_label()
    X = np.load(PROCESSED / "learned_embeddings.npy")
    ids = pd.read_csv(PROCESSED / "learned_bgc_ids.csv")
    meta_full = pd.read_parquet(PROCESSED / "feature_meta.parquet")
    meta = ids.merge(meta_full, on="bgc_id", how="left")

    if "date_added" not in meta.columns or meta["date_added"].isna().all():
        raise RuntimeError("No date_added metadata; re-run bgc-download.")

    is_held = meta["date_added"].fillna("9999").astype(str) >= cutoff
    ref_idx = np.where(~is_held.to_numpy())[0]
    held_idx = np.where(is_held.to_numpy())[0]
    LOG.info(
        "Learned temporal holdout @ %s (%s): %d ref / %d held-out",
        cutoff,
        label,
        len(ref_idx),
        len(held_idx),
    )

    # Standardize for PCA-style distances (learned embeds are already L2-normed
    # but we still run through the same _run_holdout_on_indices path which fits
    # StandardScaler+PCA on the reference).
    core = _run_holdout_on_indices(X, meta, ref_idx, held_idx, k, n_controls, seed)
    held_novelty = core.pop("_held_novelty")
    control_novelty_sample = core.pop("_control_novelty_sample")
    meta_held = core.pop("_meta_held")
    p_value = core["p_value_heldout_gt_control"]

    # Load architecture baseline for side-by-side if present
    arch_baseline = None
    arch_path = REPORTS / "temporal_holdout.json"
    if arch_path.exists():
        try:
            arch_baseline = json.loads(arch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    man = {}
    man_path = PROCESSED / "learned_embed_manifest.json"
    if man_path.exists():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    audit = {
        "representation": label,
        "cutoff_date": cutoff,
        "k": k,
        "encoder_train_cutoff": man.get("train_cutoff"),
        "leakage_safe": man.get("train_cutoff") == cutoff
        or (man.get("train_cutoff") is not None and str(man.get("train_cutoff")) <= cutoff),
        **core,
        "architecture_baseline": {
            "heldout_novelty_mean": arch_baseline.get("heldout_novelty_mean") if arch_baseline else None,
            "random_control_novelty_mean_of_means": (
                arch_baseline.get("random_control_novelty_mean_of_means") if arch_baseline else None
            ),
            "p_value_heldout_gt_control": (
                arch_baseline.get("p_value_heldout_gt_control") if arch_baseline else None
            ),
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "learned_temporal_holdout.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    meta_held.sort_values("temporal_novelty", ascending=False)[
        ["bgc_id", "organism", "biosynth_class", "date_added", "temporal_novelty"]
    ].to_csv(REPORTS / "learned_temporal_holdout_ranking.csv", index=False)

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
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Learned prospective novelty ({label})\n(Mann-Whitney p={p_value:.3g})")
    ax.set_ylabel("novelty percentile vs. reference manifold")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "learned_temporal_holdout.png", dpi=150)
    plt.close(fig)

    LOG.info(
        "Learned temporal: heldout mean=%.3f vs control=%.3f (p=%.4g)",
        audit["heldout_novelty_mean"],
        audit["random_control_novelty_mean_of_means"],
        p_value,
    )
    return audit


def run_learned_eval_suite(
    n_splits: int = 5,
    k: int = DEFAULT_NOVELTY_K,
    cutoff: str = DEFAULT_TEMPORAL_CUTOFF,
    n_controls: int = 50,
) -> dict:
    """Run class recovery + novelty compare + temporal holdout; write summary."""
    ensure_dirs()
    ablation = run_learned_class_recovery(n_splits=n_splits)
    novelty = run_learned_novelty_compare(k=k)
    temporal = run_learned_temporal_holdout(cutoff=cutoff, k=k, n_controls=n_controls)

    summary = {
        "learned_representation": _learned_label(),
        "class_recovery": {
            name: {"macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"], "model": r["model"]}
            for name, r in ablation["results"].items()
        },
        "novelty": {
            "spearman_hashed_vs_learned": novelty["spearman_hashed_vs_learned"],
            "top_decile_jaccard_hashed_vs_learned": novelty["top_decile_jaccard_hashed_vs_learned"],
            "size_confound_spearman_hashed": novelty["size_confound_spearman_hashed"],
            "size_confound_spearman_learned": novelty["size_confound_spearman_learned"],
        },
        "temporal": {
            "heldout_novelty_mean": temporal["heldout_novelty_mean"],
            "control_mean": temporal["random_control_novelty_mean_of_means"],
            "p_value": temporal["p_value_heldout_gt_control"],
            "heldout_more_novel": temporal["heldout_more_novel_than_random_control"],
            "leakage_safe": temporal.get("leakage_safe"),
            "architecture_baseline": temporal.get("architecture_baseline"),
        },
    }
    with open(REPORTS / "learned_eval_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _write_summary_figure(summary)
    LOG.info("Wrote learned eval summary → %s", REPORTS / "learned_eval_summary.json")
    return summary


def _write_summary_figure(summary: dict) -> None:
    """One hero comparison figure for the README."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # Class recovery bars
    cr = summary["class_recovery"]
    names = list(cr.keys())
    f1s = [cr[n]["macro_f1"] for n in names]
    short = [n.replace("combined_hashed+", "hash+").replace("learned_", "L/")[:22] for n in names]
    axes[0].barh(short, f1s, color="#4C72B0")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("macro-F1")
    axes[0].set_title("Class recovery")
    axes[0].axvline(0.84, color="gray", linestyle="--", linewidth=1, label="hash+ESM 0.84")
    axes[0].legend(fontsize=7)

    # Size confound
    nov = summary["novelty"]
    axes[1].bar(
        ["hashed", "learned"],
        [nov["size_confound_spearman_hashed"], nov["size_confound_spearman_learned"]],
        color=["#DD8452", "#55A868"],
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Spearman(novelty, n_genes)")
    axes[1].set_title("Size confound")

    # Temporal
    temp = summary["temporal"]
    arch = temp.get("architecture_baseline") or {}
    groups = ["arch\nheldout", "arch\ncontrol", "learned\nheldout", "learned\ncontrol"]
    vals = [
        arch.get("heldout_novelty_mean") or 0.0,
        arch.get("random_control_novelty_mean_of_means") or 0.0,
        temp["heldout_novelty_mean"],
        temp["control_mean"],
    ]
    colors = ["#DD8452", "#DD8452", "#55A868", "#55A868"]
    axes[2].bar(groups, vals, color=colors)
    axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("mean novelty percentile")
    axes[2].set_title("Prospective temporal holdout")

    fig.suptitle(
        f"Learned BGC representation — {_learned_label()}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "learned_representation_summary.png", dpi=150)
    plt.close(fig)
