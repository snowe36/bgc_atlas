"""Orchestrate token enrichment + temporal myxo sneak analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bgcatlas.analyses.myxo_grammar.cohort import assign_clade
from bgcatlas.analyses.myxo_grammar.statistics import enrich_tokens, rarity_contrast
from bgcatlas.analyses.myxo_grammar.tokens import build_token_matrix, rarity_scores


def run_enrichment(
    domains: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    min_token_bgcs: int = 5,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    myxo_ids = cohort.loc[cohort["role"] == "myxo", "bgc_id"].astype(str).tolist()
    control_ids = cohort.loc[cohort["role"] == "control", "bgc_id"].astype(str).tolist()
    all_ids = myxo_ids + control_ids
    mat, idf, _ordered = build_token_matrix(
        domains, all_ids, min_token_bgcs=min_token_bgcs
    )
    table = enrich_tokens(mat, myxo_ids, control_ids, idf=idf)
    rarity = rarity_scores(mat, idf)
    rarity_stats = rarity_contrast(rarity, myxo_ids, control_ids)
    return table, rarity, {"idf": idf, "rarity": rarity_stats, "n_tokens": int(mat.shape[1])}


def temporal_myxo_contrast(
    temporal_json: Path,
    temporal_ranking_csv: Path,
    novelty_ranking: pd.DataFrame,
) -> dict:
    """Do held-out (future) myxo BGCs look different on the novelty axis?

    Uses the existing temporal holdout artifacts. Either outcome is informative:
    lineage-persistent unexplored regions vs global failure.
    """
    if not temporal_json.exists() or not temporal_ranking_csv.exists():
        return {"available": False, "reason": "temporal artifacts missing"}

    meta = json.loads(temporal_json.read_text())
    held = pd.read_csv(temporal_ranking_csv)
    # File is held-out rows; score column is temporal_novelty (fallbacks below).
    if "organism" not in held.columns:
        nov = novelty_ranking[["bgc_id", "organism", "n_genes", "biosynth_class"]].copy()
        held = held.merge(nov, on="bgc_id", how="left")
    if "organism" not in held.columns:
        return {"available": False, "reason": "no organism column after join"}

    held["clade"] = held["organism"].map(assign_clade)
    score_col = None
    for c in ("temporal_novelty", "novelty", "score"):
        if c in held.columns:
            score_col = c
            break
    if score_col is None:
        return {"available": False, "reason": "no novelty column in temporal ranking"}

    myxo = held.loc[held["clade"] == "Myxobacteria", score_col].dropna()
    other = held.loc[held["clade"] != "Myxobacteria", score_col].dropna()
    strep = held.loc[held["clade"] == "Streptomyces", score_col].dropna()

    out: dict = {
        "available": True,
        "cutoff_date": meta.get("cutoff_date"),
        "n_heldout_total": int(len(held)),
        "n_heldout_myxo": int(len(myxo)),
        "n_heldout_streptomyces": int(len(strep)),
        "n_heldout_other": int((held["clade"] == "Other").sum()),
        "global_heldout_mean": meta.get("heldout_novelty_mean"),
        "global_control_mean": meta.get("random_control_novelty_mean_of_means"),
        "global_p_heldout_gt_control": meta.get("p_value_heldout_gt_control"),
        "heldout_mean_myxo": float(myxo.mean()) if len(myxo) else None,
        "heldout_mean_streptomyces": float(strep.mean()) if len(strep) else None,
        "heldout_mean_nonmyxo": float(other.mean()) if len(other) else None,
    }
    if len(myxo) >= 3 and len(other) >= 3:
        from scipy.stats import mannwhitneyu

        u, p = mannwhitneyu(myxo.to_numpy(), other.to_numpy(), alternative="greater")
        out["mannwhitney_myxo_gt_nonmyxo_u"] = float(u)
        out["mannwhitney_myxo_gt_nonmyxo_p"] = float(p)
    return out
