"""Level-2 evaluation: ER-positive cluster labels for metrics ONLY.

Must not be imported by candidates.py ranking. Cards may consume post-hoc
annotations after ranking is complete.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from bgcatlas.analyses.capability_gap.config import CAPABILITIES
from bgcatlas.analyses.capability_gap.candidates import cds_families_homolog
from bgcatlas.analyses.myxo_grammar.tokens import NOISE_DOMAINS


def cluster_contains_known_ER(
    clusters: pd.DataFrame,
    domains: pd.DataFrame,
    *,
    capacity: str = "enoyl_reduction",
) -> set[str]:
    """Ground-truth: family_keys whose members include annotated capacity domains."""
    ban = CAPABILITIES[capacity]
    dom = domains[domains["feature_type"] == "domain"].copy()
    dom = dom[dom["domain_id"].astype(str).str.len() > 0]
    dom = dom[~dom["domain_id"].isin(NOISE_DOMAINS)]
    er_loci: set[tuple[str, str]] = set()
    for (bgc_id, locus), g in dom.groupby(["bgc_id", "locus_tag"]):
        dset = set(map(str, g["domain_id"].tolist()))
        if dset & ban:
            er_loci.add((str(bgc_id), str(locus) if locus == locus else ""))

    clu = clusters.copy()
    clu["bgc_id"] = clu["bgc_id"].astype(str)
    clu["locus_tag"] = clu["locus_tag"].astype(str)
    if "family_key" not in clu.columns:
        clu["family_key"] = "homolog90__" + clu["cluster_id"].astype(str)

    positive: set[str] = set()
    for r in clu.itertuples(index=False):
        if (str(r.bgc_id), str(r.locus_tag)) in er_loci:
            positive.add(str(r.family_key))
    return positive


def per_bgc_ranked_families(
    gap_bgcs: pd.DataFrame,
    families: pd.DataFrame,
    domains: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    capacity: str,
) -> dict[str, list[str]]:
    """For each gap BGC, ordered family_keys present in-cluster (by global rank)."""
    if families.empty:
        return {}
    fam_order = families["family_key"].astype(str).tolist()
    present = cds_families_homolog(
        domains, clusters, capacity=capacity, exclude_same_bgc_capacity=True
    )
    by_bgc: dict[str, set[str]] = defaultdict(set)
    for r in present.itertuples(index=False):
        by_bgc[str(r.bgc_id)].add(str(r.family_key))

    out: dict[str, list[str]] = {}
    for bid in gap_bgcs["bgc_id"].astype(str):
        keys = by_bgc.get(bid, set())
        ranked = [fk for fk in fam_order if fk in keys]
        out[bid] = ranked
    return out


def level2_recovery_metrics(
    gap_bgcs: pd.DataFrame,
    families: pd.DataFrame,
    domains: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    capacity: str = "enoyl_reduction",
    k_list: tuple[int, ...] = (1, 5),
) -> dict:
    """Recall@k / MRR / enrichment@k for ER-positive clusters on gap BGCs.

    Only counts gap BGCs that contain ≥1 ER-positive family among their
    in-cluster homolog members (recoverable cases).
    """
    er_pos = cluster_contains_known_ER(clusters, domains, capacity=capacity)
    ranked = per_bgc_ranked_families(
        gap_bgcs, families, domains, clusters, capacity=capacity
    )

    # Universe: all families that appear in any gap BGC ranking pool
    all_in_gaps: set[str] = set()
    for keys in ranked.values():
        all_in_gaps.update(keys)
    n_universe = max(len(all_in_gaps), 1)
    n_er_in_universe = len(all_in_gaps & er_pos)
    base_rate = n_er_in_universe / n_universe

    recalls = {k: [] for k in k_list}
    rr_list: list[float] = []
    topk_hits = {k: 0 for k in k_list}
    topk_slots = {k: 0 for k in k_list}
    n_scored = 0

    for bid, order in ranked.items():
        gold = [fk for fk in order if fk in er_pos]
        if not gold:
            continue
        n_scored += 1
        # reciprocal rank of first gold
        rr = 0.0
        for i, fk in enumerate(order):
            if fk in er_pos:
                rr = 1.0 / (i + 1)
                break
        rr_list.append(rr)
        for k in k_list:
            hit = any(fk in er_pos for fk in order[:k])
            recalls[k].append(1.0 if hit else 0.0)
            topk_slots[k] += min(k, len(order))
            topk_hits[k] += sum(1 for fk in order[:k] if fk in er_pos)

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    metrics: dict = {
        "representation": "homolog90",
        "n_er_positive_clusters": int(len(er_pos)),
        "n_gap_bgcs": int(len(gap_bgcs)),
        "n_gap_bgcs_scored": int(n_scored),
        "n_families_in_gap_universe": int(n_universe),
        "n_er_positive_in_gap_universe": int(n_er_in_universe),
        "random_baseline_rate": float(base_rate),
        "mrr": _mean(rr_list),
        "note": (
            "Recall@k scores gap BGCs that contain ≥1 member of an ER-positive "
            "homolog90 cluster. At 90% identity this is often zero: annotated ER "
            "proteins form tight clusters that gap BGCs do not share — a real "
            "biological/representation result, not a pipeline failure."
        ),
    }
    for k in k_list:
        metrics[f"recall_at_{k}"] = _mean(recalls[k])
        obs = (topk_hits[k] / topk_slots[k]) if topk_slots[k] else 0.0
        metrics[f"precision_at_{k}"] = float(obs)
        metrics[f"enrichment_at_{k}"] = (
            float(obs / base_rate) if base_rate > 0 else 0.0
        )
    # Global enrichment among ranked families (does not require ER clusters in gap BGCs).
    ranked_keys = families["family_key"].astype(str).tolist() if len(families) else []
    n_ranked = max(len(ranked_keys), 1)
    n_er_ranked = sum(1 for fk in ranked_keys if fk in er_pos)
    all_family_keys = set(clusters["family_key"].astype(str)) if "family_key" in clusters.columns else set()
    if not all_family_keys and "cluster_id" in clusters.columns:
        all_family_keys = {"homolog90__" + str(c) for c in clusters["cluster_id"].unique()}
    n_all = max(len(all_family_keys), 1)
    global_base = len(er_pos) / n_all
    ranked_rate = n_er_ranked / n_ranked
    metrics["n_er_positive_in_ranked_families"] = int(n_er_ranked)
    metrics["n_ranked_families"] = int(len(ranked_keys))
    metrics["global_er_base_rate"] = float(global_base)
    metrics["ranked_er_rate"] = float(ranked_rate)
    metrics["enrichment_ranked_vs_global"] = (
        float(ranked_rate / global_base) if global_base > 0 else 0.0
    )
    metrics["frac_singleton_clusters"] = float(
        (clusters.drop_duplicates("cluster_id")["n_members"] == 1).mean()
        if "n_members" in clusters.columns and len(clusters)
        else 0.0
    )
    return metrics


def posthoc_er_homolog_note(
    family_key: str | None,
    er_positive: set[str],
) -> str | None:
    """Display-only annotation after ranking — not a ranking feature."""
    if not family_key:
        return None
    if family_key in er_positive:
        return "contains_ER_positive_homolog_elsewhere"
    return None
