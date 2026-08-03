"""Interpretable architecture tokens from existing domain annotations.

Does not invent a new embedding — extracts abundance, ordered motifs, and
rarity-weighted vocabulary from MIBiG domain tables already used by featurize.
"""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd

# Annotation noise — not biosynthetic grammar.
NOISE_DOMAINS = {
    "CDS_other",
    "Hypothetical",
    "Regulator",
    "unknown",
}

# Functional capacity buckets (Level 1 aggregates over raw domain ids).
CAPACITY_MAP: dict[str, set[str]] = {
    "capacity::NRPS_adenylation": {"AMP-binding", "A"},
    "capacity::NRPS_condensation": {"Condensation", "C"},
    "capacity::NRPS_thiolation": {"PCP", "PP"},
    "capacity::NRPS_module": {"NRPS_module"},
    "capacity::PKS_KS": {"PKS_KS", "KS", "polyketide_synthase"},
    "capacity::PKS_AT": {"PKS_AT", "AT"},
    "capacity::PKS_KR": {"PKS_KR", "KR"},
    "capacity::PKS_DH": {"PKS_DH", "PKS_DH2", "DH"},
    "capacity::PKS_ER": {"PKS_ER", "ER"},
    "capacity::PKS_module": {"PKS_module"},
    "capacity::ACP": {"ACP", "PKS_PP"},
    "capacity::thioesterase": {"Thioesterase", "thioesterase"},
    "capacity::glycosyltransferase": {"Glycosyltransferase"},
    "capacity::halogenase": {"Halogenase"},
    "capacity::P450": {"P450", "PF00067"},
    "capacity::methyltransferase": {"Methyltransferase", "MT"},
    "capacity::transporter": {"Transporter", "PF00005"},
    "capacity::redox": {"Redox"},
    "capacity::hydrolase": {"Hydrolase"},
    "capacity::kinase": {"Kinase"},
}

# Coarse combo grammars (presence of two capacities).
COMBO_GRAMMARS = [
    ("grammar::PKS+GT", "capacity::PKS_KS", "capacity::glycosyltransferase"),
    ("grammar::PKS+P450", "capacity::PKS_KS", "capacity::P450"),
    ("grammar::PKS+halogenase", "capacity::PKS_KS", "capacity::halogenase"),
    ("grammar::NRPS+oxidase", "capacity::NRPS_adenylation", "capacity::redox"),
    ("grammar::NRPS+P450", "capacity::NRPS_adenylation", "capacity::P450"),
    ("grammar::NRPS+halogenase", "capacity::NRPS_adenylation", "capacity::halogenase"),
    ("grammar::NRPS+MT", "capacity::NRPS_adenylation", "capacity::methyltransferase"),
    ("grammar::PKS+MT", "capacity::PKS_KS", "capacity::methyltransferase"),
    ("grammar::PKS+NRPS_hybrid_parts", "capacity::PKS_KS", "capacity::NRPS_adenylation"),
    ("grammar::tailoring_GT+P450", "capacity::glycosyltransferase", "capacity::P450"),
]


def _ordered_domains(domains: pd.DataFrame) -> dict[str, list[str]]:
    dom = domains[domains["feature_type"] == "domain"].copy()
    dom = dom[dom["domain_id"].astype(str).str.len() > 0]
    dom = dom[~dom["domain_id"].isin(NOISE_DOMAINS)]
    sort_cols = [c for c in ("bgc_id", "gene_order", "locus_tag", "domain_id") if c in dom.columns]
    dom = dom.sort_values(sort_cols)
    out: dict[str, list[str]] = {}
    for bgc_id, g in dom.groupby("bgc_id", sort=False):
        seq = [str(x) for x in g["domain_id"].tolist()]
        # Collapse consecutive duplicates (multi-annotation of same domain).
        collapsed: list[str] = []
        for tok in seq:
            if not collapsed or collapsed[-1] != tok:
                collapsed.append(tok)
        out[str(bgc_id)] = collapsed
    return out


def _motifs(seq: list[str], max_n: int = 3) -> list[str]:
    toks: list[str] = []
    for n in range(2, max_n + 1):
        for i in range(len(seq) - n + 1):
            toks.append(f"motif{n}::" + "-".join(seq[i : i + n]))
    return toks


def extract_tokens_for_bgc(seq: list[str]) -> set[str]:
    """Return the set of interpretable tokens present in one BGC."""
    present: set[str] = set()
    uniq = set(seq)
    for d in uniq:
        present.add(f"dom::{d}")
    for cap, members in CAPACITY_MAP.items():
        if uniq & members:
            present.add(cap)
    for name, a, b in COMBO_GRAMMARS:
        if a in present and b in present:
            present.add(name)
    for m in _motifs(seq, max_n=3):
        present.add(m)
    return present


def build_token_matrix(
    domains: pd.DataFrame,
    bgc_ids: list[str],
    *,
    min_token_bgcs: int = 5,
    max_motif_bgc_frac: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, list[str]]]:
    """Binary BGC × token matrix + corpus IDF + ordered sequences.

    Motifs present in > max_motif_bgc_frac of BGCs are dropped (too common).
    Domains / capacities / grammars keep min_token_bgcs only.
    """
    ordered = _ordered_domains(domains)
    rows: list[dict[str, int | str]] = []
    for bgc_id in bgc_ids:
        seq = ordered.get(str(bgc_id), [])
        toks = extract_tokens_for_bgc(seq)
        row: dict[str, int | str] = {"bgc_id": str(bgc_id)}
        for t in toks:
            row[t] = 1
        rows.append(row)

    mat = pd.DataFrame(rows).fillna(0)
    if "bgc_id" not in mat.columns:
        mat = pd.DataFrame({"bgc_id": [str(b) for b in bgc_ids]})
    mat = mat.set_index("bgc_id")
    # Align to requested order; missing BGCs → all zeros
    mat = mat.reindex([str(b) for b in bgc_ids], fill_value=0)
    # Ensure numeric
    for c in mat.columns:
        mat[c] = mat[c].astype(int)

    n = len(mat)
    keep: list[str] = []
    for col in mat.columns:
        support = int(mat[col].sum())
        if support < min_token_bgcs:
            continue
        if col.startswith("motif") and support > max_motif_bgc_frac * n:
            continue
        keep.append(col)
    mat = mat[keep]

    # IDF over the analysis cohort (rarity-weighted vocabulary).
    idf: dict[str, float] = {}
    for col in mat.columns:
        dfreq = max(int(mat[col].sum()), 1)
        idf[col] = math.log(n / dfreq)

    return mat, idf, ordered


def rarity_scores(mat: pd.DataFrame, idf: dict[str, float]) -> pd.Series:
    """Mean IDF of tokens present in each BGC (rare vocabulary intensity)."""
    if mat.empty:
        return pd.Series(dtype=float)
    weights = pd.Series({c: idf.get(c, 0.0) for c in mat.columns})
    # For each BGC: sum(idf for present tokens) / n_present
    present_n = mat.sum(axis=1).astype(float)
    weighted = mat.mul(weights, axis=1).sum(axis=1).astype(float)
    out = weighted / present_n.replace(0.0, float("nan"))
    return out
