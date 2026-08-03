"""Rank candidate protein *families*, not fragile individual ORFs."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from bgcatlas.analyses.capability_gap.config import (
    CAPABILITIES,
    CORE_DOMAINS,
    IMPLAUSIBLE_ER_FAMILIES,
    REDOX_HINT_DOMAINS,
)
from bgcatlas.analyses.myxo_grammar.cohort import assign_clade
from bgcatlas.analyses.myxo_grammar.tokens import NOISE_DOMAINS


def _cds_families(domains: pd.DataFrame) -> pd.DataFrame:
    """One row per (bgc_id, locus_tag) with a stable family key.

    Family key = sorted non-noise domains on that CDS, preferring non-core
    domains when present (compensatory / tailoring hypotheses).
    """
    dom = domains[domains["feature_type"] == "domain"].copy()
    dom = dom[dom["domain_id"].astype(str).str.len() > 0]
    dom = dom[~dom["domain_id"].isin(NOISE_DOMAINS)]
    rows = []
    group_cols = ["bgc_id", "locus_tag"]
    if "gene_order" in dom.columns:
        group_cols = ["bgc_id", "gene_order", "locus_tag"]
    for keys, g in dom.groupby(group_cols, dropna=False):
        if len(group_cols) == 3:
            bgc_id, gene_order, locus = keys
        else:
            bgc_id, locus = keys
            gene_order = None
        domains_here = sorted({str(x) for x in g["domain_id"].tolist()})
        noncore = [d for d in domains_here if d not in CORE_DOMAINS]
        # Prefer non-core / unknown / redox-like domains as family keys.
        family_domains = noncore if noncore else []
        if not family_domains:
            # Fall back only to redox-hint domains, never pure assembly cores.
            family_domains = [d for d in domains_here if d in REDOX_HINT_DOMAINS]
        if not family_domains:
            continue
        family_key = "+".join(family_domains[:4])
        product = ""
        if "product" in g.columns:
            product = str(g["product"].dropna().iloc[0]) if g["product"].notna().any() else ""
        rows.append(
            {
                "bgc_id": str(bgc_id),
                "locus_tag": str(locus) if locus == locus else "",
                "gene_order": gene_order,
                "family_key": family_key,
                "domains": ",".join(domains_here),
                "product": product,
            }
        )
    return pd.DataFrame(rows)


def rank_candidate_families(
    gap_bgcs: pd.DataFrame,
    control_bgcs: pd.DataFrame,
    domains: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    capacity: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Score families by association with expectation-gap pathways.

    Evidence axes (v0, no AlphaFold yet):
    - co-occurrence with gap BGCs
    - depleted in matched/architecture controls that already have the capacity
    - myxo-restricted distribution
    - synteny near PKS modules (mean gene_order proximity to KS)
    """
    fam = _cds_families(domains)
    if fam.empty:
        return pd.DataFrame()

    gap_ids = set(gap_bgcs["bgc_id"].astype(str))
    # Controls: same size-filtered manifold that *have* the capacity (satisfied expectation)
    ctrl_ids = set(control_bgcs["bgc_id"].astype(str))

    # Exclude families that are exactly the missing capability's domains
    ban = CAPABILITIES[capacity]

    # Synteny: distance to nearest KS-bearing gene_order within BGC
    ks_orders: dict[str, list[int]] = defaultdict(list)
    dom = domains[domains["feature_type"] == "domain"].copy()
    for r in dom.itertuples(index=False):
        if str(r.domain_id) in {"PKS_KS", "KS", "polyketide_synthase"} and pd.notna(r.gene_order):
            ks_orders[str(r.bgc_id)].append(int(r.gene_order))

    org = ranking.set_index("bgc_id")["organism"].astype(str).to_dict()

    gap_counts: Counter[str] = Counter()
    ctrl_counts: Counter[str] = Counter()
    myxo_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    synteny_sum: dict[str, float] = defaultdict(float)
    synteny_n: Counter[str] = Counter()
    example_bgc: dict[str, str] = {}

    for r in fam.itertuples(index=False):
        bid = str(r.bgc_id)
        fk = r.family_key
        fdoms = set(fk.split("+"))
        if fdoms <= ban:
            continue
        if capacity == "enoyl_reduction" and fdoms <= IMPLAUSIBLE_ER_FAMILIES:
            continue
        if fdoms <= CORE_DOMAINS:
            continue
        total_counts[fk] += 1
        clade = assign_clade(org.get(bid, ""))
        if clade == "Myxobacteria":
            myxo_counts[fk] += 1
        if bid in gap_ids:
            gap_counts[fk] += 1
            example_bgc.setdefault(fk, bid)
            if r.gene_order is not None and bid in ks_orders:
                dmin = min(abs(int(r.gene_order) - k) for k in ks_orders[bid])
                synteny_sum[fk] += float(dmin)
                synteny_n[fk] += 1
        elif bid in ctrl_ids:
            ctrl_counts[fk] += 1

    n_gap = max(len(gap_ids), 1)
    n_ctrl = max(len(ctrl_ids), 1)
    rows = []
    for fk, n_g in gap_counts.most_common():
        n_c = ctrl_counts.get(fk, 0)
        n_tot = total_counts[fk]
        n_myxo = myxo_counts.get(fk, 0)
        gap_frac = n_g / n_gap
        ctrl_frac = n_c / n_ctrl
        # Simple association score
        score = gap_frac - ctrl_frac + 0.25 * (n_myxo / max(n_tot, 1))
        if fdoms & REDOX_HINT_DOMAINS or any(d.startswith("DUF") for d in fdoms):
            score += 0.2
        mean_synteny = synteny_sum[fk] / synteny_n[fk] if synteny_n[fk] else None
        if mean_synteny is not None:
            # closer to KS → small bonus
            score += max(0.0, 0.15 * (1.0 - min(mean_synteny, 10) / 10))
        rows.append(
            {
                "family_key": fk,
                "n_gap_bgcs": int(n_g),
                "n_control_bgcs": int(n_c),
                "n_total_bgcs": int(n_tot),
                "frac_gap_bgcs": float(gap_frac),
                "frac_control_bgcs": float(ctrl_frac),
                "frac_myxo_among_family": float(n_myxo / max(n_tot, 1)),
                "mean_gene_order_dist_to_KS": mean_synteny,
                "association_score": float(score),
                "example_gap_bgc": example_bgc.get(fk),
                "evidence_cooccurrence": n_g >= 2,
                "evidence_control_depleted": ctrl_frac < gap_frac,
                "evidence_myxo_biased": (n_myxo / max(n_tot, 1)) >= 0.5,
                "evidence_synteny": mean_synteny is not None and mean_synteny <= 5,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("association_score", ascending=False).head(top_n).reset_index(drop=True)


def best_family_for_bgc(
    bgc_id: str,
    families: pd.DataFrame,
    domains: pd.DataFrame,
) -> dict | None:
    """Pick the highest-scoring family that actually occurs in this BGC."""
    if families.empty:
        return None
    present = _cds_families(domains)
    present = present[present["bgc_id"].astype(str) == str(bgc_id)]
    if present.empty:
        return None
    keys = set(present["family_key"])
    hit = families[families["family_key"].isin(keys)]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()
