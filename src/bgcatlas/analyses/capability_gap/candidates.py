"""Rank candidate protein *families* via homolog90 clusters (Level 2).

Ranking features must NEVER include ER-positive / capacity labels.
Those belong only in evaluation.py.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from bgcatlas.analyses.capability_gap.config import CAPABILITIES
from bgcatlas.analyses.capability_gap.homologs import REPRESENTATION
from bgcatlas.analyses.myxo_grammar.cohort import assign_clade
from bgcatlas.analyses.myxo_grammar.tokens import NOISE_DOMAINS

# Feature columns emitted by the ranker — tests assert ER labels are absent.
RANKING_FEATURE_COLS = (
    "family_key",
    "n_gap_bgcs",
    "n_control_bgcs",
    "n_total_bgcs",
    "n_members",
    "frac_gap_bgcs",
    "frac_control_bgcs",
    "frac_myxo_among_family",
    "mean_gene_order_dist_to_KS",
    "association_score",
    "example_gap_bgc",
    "evidence_cooccurrence",
    "evidence_control_depleted",
    "evidence_myxo_biased",
    "evidence_synteny",
    "representation",
)


def _cds_domain_sets(domains: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """(bgc_id, locus_tag) → domain set (for same-BGC ER exclusion only)."""
    dom = domains[domains["feature_type"] == "domain"].copy()
    dom = dom[dom["domain_id"].astype(str).str.len() > 0]
    dom = dom[~dom["domain_id"].isin(NOISE_DOMAINS)]
    out: dict[tuple[str, str], set[str]] = {}
    for (bgc_id, locus), g in dom.groupby(["bgc_id", "locus_tag"]):
        key = (str(bgc_id), str(locus) if locus == locus else "")
        out[key] = set(map(str, g["domain_id"].tolist()))
    return out


def _cds_gene_orders(domains: pd.DataFrame) -> dict[tuple[str, str], int | None]:
    out: dict[tuple[str, str], int | None] = {}
    if "gene_order" not in domains.columns:
        return out
    for r in domains.itertuples(index=False):
        key = (str(r.bgc_id), str(r.locus_tag) if pd.notna(r.locus_tag) else "")
        go = int(r.gene_order) if pd.notna(r.gene_order) else None
        if key not in out or (go is not None and out[key] is None):
            out[key] = go
    return out


def cds_families_homolog(
    domains: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    capacity: str,
    exclude_same_bgc_capacity: bool = True,
) -> pd.DataFrame:
    """One row per (bgc_id, locus_tag) with homolog90 family_key.

    Discovery mode: exclude CDS that themselves carry the annotated capacity
    domain in the *same* BGC (trivial leakage). Do NOT ban clusters that
    contain capacity homologs in *other* BGCs.
    """
    if clusters.empty:
        return pd.DataFrame()
    clu = clusters.copy()
    clu["bgc_id"] = clu["bgc_id"].astype(str)
    clu["locus_tag"] = clu["locus_tag"].astype(str)

    dom_sets = _cds_domain_sets(domains)
    gene_orders = _cds_gene_orders(domains)
    ban = CAPABILITIES[capacity]

    rows = []
    for r in clu.itertuples(index=False):
        bid = str(r.bgc_id)
        locus = str(r.locus_tag)
        key = (bid, locus)
        dset = dom_sets.get(key, set())
        if exclude_same_bgc_capacity and dset & ban:
            # Same-BGC annotated capacity CDS — trivial leakage.
            continue
        fk = (
            str(r.family_key)
            if hasattr(r, "family_key") and pd.notna(r.family_key)
            else f"{REPRESENTATION}__{r.cluster_id}"
        )
        if not fk.startswith(f"{REPRESENTATION}__"):
            fk = f"{REPRESENTATION}__{fk}"
        rows.append(
            {
                "bgc_id": bid,
                "locus_tag": locus,
                "gene_order": gene_orders.get(key),
                "family_key": fk,
                "cluster_id": str(r.cluster_id),
                "n_members": int(r.n_members) if pd.notna(r.n_members) else 1,
                "representation": REPRESENTATION,
            }
        )
    return pd.DataFrame(rows)


# Back-compat alias used by cards / tests
def _cds_families(
    domains: pd.DataFrame,
    clusters: pd.DataFrame | None = None,
    *,
    capacity: str = "enoyl_reduction",
) -> pd.DataFrame:
    if clusters is None or clusters.empty:
        return pd.DataFrame()
    return cds_families_homolog(domains, clusters, capacity=capacity)


def rank_candidate_families(
    gap_bgcs: pd.DataFrame,
    control_bgcs: pd.DataFrame,
    domains: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    capacity: str,
    clusters: pd.DataFrame,
    top_n: int = 10,
    exclude_same_bgc_capacity: bool = True,
) -> pd.DataFrame:
    """Score homolog families by association with expectation-gap pathways.

    Allowed evidence only: co-occurrence, control depletion, myxo bias, synteny.
    Forbidden: any ER-positive / capacity label on the cluster.
    """
    fam = cds_families_homolog(
        domains,
        clusters,
        capacity=capacity,
        exclude_same_bgc_capacity=exclude_same_bgc_capacity,
    )
    if fam.empty:
        return pd.DataFrame()

    gap_ids = set(gap_bgcs["bgc_id"].astype(str))
    ctrl_ids = set(control_bgcs["bgc_id"].astype(str))

    ks_orders: dict[str, list[int]] = defaultdict(list)
    dom = domains[domains["feature_type"] == "domain"].copy()
    for r in dom.itertuples(index=False):
        if str(r.domain_id) in {"PKS_KS", "KS", "polyketide_synthase"} and pd.notna(
            r.gene_order
        ):
            ks_orders[str(r.bgc_id)].append(int(r.gene_order))

    org = ranking.set_index("bgc_id")["organism"].astype(str).to_dict()

    gap_counts: Counter[str] = Counter()
    ctrl_counts: Counter[str] = Counter()
    myxo_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    member_n: dict[str, int] = {}
    synteny_sum: dict[str, float] = defaultdict(float)
    synteny_n: Counter[str] = Counter()
    example_bgc: dict[str, str] = {}

    for r in fam.itertuples(index=False):
        bid = str(r.bgc_id)
        fk = r.family_key
        member_n[fk] = max(member_n.get(fk, 0), int(r.n_members))
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
        score = gap_frac - ctrl_frac + 0.25 * (n_myxo / max(n_tot, 1))
        # Prefer multi-member clusters (identity resolution signal, not ER label).
        n_mem = member_n.get(fk, 1)
        if n_mem >= 2:
            score += 0.1
        if n_mem >= 5:
            score += 0.05
        mean_synteny = synteny_sum[fk] / synteny_n[fk] if synteny_n[fk] else None
        if mean_synteny is not None:
            score += max(0.0, 0.15 * (1.0 - min(mean_synteny, 10) / 10))
        rows.append(
            {
                "family_key": fk,
                "n_gap_bgcs": int(n_g),
                "n_control_bgcs": int(n_c),
                "n_total_bgcs": int(n_tot),
                "n_members": int(n_mem),
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
                "representation": REPRESENTATION,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Hard anti-leakage: never emit capacity-label columns.
    assert not any("er_pos" in c.lower() or c.endswith("_ER") for c in out.columns)
    return out.sort_values("association_score", ascending=False).head(top_n).reset_index(
        drop=True
    )


def best_family_for_bgc(
    bgc_id: str,
    families: pd.DataFrame,
    domains: pd.DataFrame,
    *,
    clusters: pd.DataFrame,
    capacity: str = "enoyl_reduction",
) -> dict | None:
    """Pick the highest-scoring homolog family present in this BGC."""
    if families.empty or clusters.empty:
        return None
    present = cds_families_homolog(domains, clusters, capacity=capacity)
    present = present[present["bgc_id"].astype(str) == str(bgc_id)]
    if present.empty:
        return None
    keys = set(present["family_key"])
    hit = families[families["family_key"].isin(keys)]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()
