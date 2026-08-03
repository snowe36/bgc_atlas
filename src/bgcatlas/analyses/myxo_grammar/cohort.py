"""Matched myxo vs control cohort + phenotype gate (OR≈2.9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm

MYXO_GENERA = {
    "Sorangium",
    "Chondromyces",
    "Stigmatella",
    "Myxococcus",
    "Corallococcus",
    "Cystobacter",
    "Melittangium",
    "Archangium",
    "Pyxidicoccus",
}

# Primary contrast that produced the published OR≈2.9 finding.
DEFAULT_CONTROL_CLADE = "Streptomyces"
SIZE_MAX_DEFAULT = 60
GENE_TOL_DEFAULT = 0.10
N_CONTROLS_DEFAULT = 5


def genus(organism: str) -> str:
    parts = str(organism).split()
    return parts[0] if parts else ""


def assign_clade(organism: str) -> str:
    g = genus(organism)
    if g in MYXO_GENERA:
        return "Myxobacteria"
    if g == "Streptomyces":
        return "Streptomyces"
    return "Other"


def phenotype_gate(
    ranking: pd.DataFrame,
    *,
    size_max: int | None = SIZE_MAX_DEFAULT,
    control_clade: str = DEFAULT_CONTROL_CLADE,
) -> dict:
    """Reproduce the size-controlled myxo vs Streptomyces top-decile OR.

    Uses the same ≤60-gene filter and Mann–Whitney / OR machinery as the
    biological case-study report so the grammar analysis stays tied to that phenotype.
    """
    df = ranking.copy()
    if size_max is not None:
        df = df[df["n_genes"] <= size_max].copy()
    df["clade"] = df["organism"].map(assign_clade)
    # Match biological_case_studies: top-decile threshold on the full size-filtered
    # table (all clades), then contrast myxo vs control membership.
    thr = df["novelty"].quantile(0.9)
    myxo = df.loc[df["clade"] == "Myxobacteria", "novelty"].to_numpy()
    ctrl = df.loc[df["clade"] == control_clade, "novelty"].to_numpy()
    n_myxo_top = int((myxo >= thr).sum())
    n_myxo_rest = int(len(myxo) - n_myxo_top)
    n_ctrl_top = int((ctrl >= thr).sum())
    n_ctrl_rest = int(len(ctrl) - n_ctrl_top)
    a, b, c, d = float(n_myxo_top), float(n_myxo_rest), float(n_ctrl_top), float(n_ctrl_rest)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = float((a / b) / (c / d))
    se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    z = float(norm.ppf(0.975))
    u_stat, p_value = mannwhitneyu(myxo, ctrl, alternative="greater")
    return {
        "size_filter_n_genes_max": size_max,
        "control_clade": control_clade,
        "n_myxo": int(len(myxo)),
        "n_control": int(len(ctrl)),
        "median_myxo": float(np.median(myxo)),
        "median_control": float(np.median(ctrl)),
        "frac_top_decile_myxo": float(n_myxo_top / len(myxo)),
        "frac_top_decile_control": float(n_ctrl_top / len(ctrl)),
        "odds_ratio_top_decile": odds_ratio,
        "odds_ratio_top_decile_ci95": [
            float(np.exp(np.log(odds_ratio) - z * se)),
            float(np.exp(np.log(odds_ratio) + z * se)),
        ],
        "contingency_top_decile": {
            "myxo_top": n_myxo_top,
            "myxo_rest": n_myxo_rest,
            "control_top": n_ctrl_top,
            "control_rest": n_ctrl_rest,
        },
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(p_value),
        # Expect ~OR 2.9 (CI ~1.8–4.9) for the ≤60-gene Streptomyces contrast.
        "gate_ok": bool(2.0 <= odds_ratio <= 5.0 and p_value < 0.01),
    }


def build_matched_cohort(
    ranking: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    size_max: int | None = SIZE_MAX_DEFAULT,
    gene_tol: float = GENE_TOL_DEFAULT,
    n_controls: int = N_CONTROLS_DEFAULT,
    control_clade: str = DEFAULT_CONTROL_CLADE,
    require_same_class: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Match each myxo BGC to size-(and class-)matched controls.

    Returns
    -------
    cohort_bgcs : one row per BGC in the analysis set (myxo + selected controls)
    matches : one row per (myxo_id, control_id) pair
    summary : matching diagnostics
    """
    nov = ranking[["bgc_id", "organism", "biosynth_class", "novelty", "n_genes"]].copy()
    mcols = ["bgc_id", "biosynth_class", "organism", "n_genes"]
    meta_s = meta[mcols].drop_duplicates("bgc_id")
    df = nov.merge(meta_s, on="bgc_id", how="left", suffixes=("", "_meta"))
    if "biosynth_class_meta" in df.columns:
        df["biosynth_class"] = df["biosynth_class"].fillna(df["biosynth_class_meta"])
    if "n_genes_meta" in df.columns:
        df["n_genes"] = pd.to_numeric(df["n_genes"], errors="coerce").fillna(
            pd.to_numeric(df["n_genes_meta"], errors="coerce")
        )
    df["n_genes"] = pd.to_numeric(df["n_genes"], errors="coerce")
    df = df.dropna(subset=["n_genes", "novelty"]).copy()
    if size_max is not None:
        df = df[df["n_genes"] <= size_max].copy()
    df["clade"] = df["organism"].map(assign_clade)

    myxo = df[df["clade"] == "Myxobacteria"].copy()
    pool = df[df["clade"] == control_clade].copy()
    # Drop singleton classes in the myxo set when requiring class match.
    class_counts = myxo["biosynth_class"].value_counts()

    match_rows: list[dict] = []
    used_controls: set[str] = set()
    unmatched = 0
    for row in myxo.itertuples(index=False):
        n = float(row.n_genes)
        lo, hi = n * (1.0 - gene_tol), n * (1.0 + gene_tol)
        cand = pool[(pool["n_genes"] >= lo) & (pool["n_genes"] <= hi)].copy()
        same_class = False
        if require_same_class and class_counts.get(row.biosynth_class, 0) >= 2:
            prefer = cand[cand["biosynth_class"] == row.biosynth_class]
            if len(prefer) >= 1:
                cand = prefer
                same_class = True
        if cand.empty:
            unmatched += 1
            continue
        cand = cand.assign(gene_delta=(cand["n_genes"] - n).abs())
        cand = cand.sort_values(["gene_delta", "bgc_id"])
        chosen = cand.head(n_controls)
        for c in chosen.itertuples(index=False):
            used_controls.add(str(c.bgc_id))
            match_rows.append(
                {
                    "myxo_id": row.bgc_id,
                    "control_id": c.bgc_id,
                    "myxo_n_genes": int(row.n_genes),
                    "control_n_genes": int(c.n_genes),
                    "myxo_class": row.biosynth_class,
                    "control_class": c.biosynth_class,
                    "same_class": bool(same_class and c.biosynth_class == row.biosynth_class),
                    "myxo_novelty": float(row.novelty),
                    "control_novelty": float(c.novelty),
                    "architecture_distance": float(abs(row.novelty - c.novelty)),
                }
            )

    matches = pd.DataFrame(match_rows)
    keep_ids = set(myxo["bgc_id"].astype(str)) | used_controls
    cohort = df[df["bgc_id"].astype(str).isin(keep_ids)].copy()
    cohort["role"] = np.where(cohort["clade"] == "Myxobacteria", "myxo", "control")

    summary = {
        "size_max": size_max,
        "gene_tol": gene_tol,
        "n_controls_per_myxo": n_controls,
        "control_clade": control_clade,
        "n_myxo": int(len(myxo)),
        "n_control_pool": int(len(pool)),
        "n_matched_myxo": int(matches["myxo_id"].nunique()) if len(matches) else 0,
        "n_unique_controls": int(len(used_controls)),
        "n_match_pairs": int(len(matches)),
        "frac_same_class_pairs": float(matches["same_class"].mean()) if len(matches) else None,
        "n_unmatched_myxo": int(unmatched),
        "mean_architecture_distance": float(matches["architecture_distance"].mean())
        if len(matches)
        else None,
    }
    return cohort, matches, summary


def cohort_bgc_ids(cohort: pd.DataFrame) -> tuple[list[str], list[str]]:
    myxo = cohort.loc[cohort["role"] == "myxo", "bgc_id"].astype(str).tolist()
    ctrl = cohort.loc[cohort["role"] == "control", "bgc_id"].astype(str).tolist()
    return myxo, ctrl


def load_atlas_coords(path) -> pd.DataFrame:
    return pd.read_csv(path)
