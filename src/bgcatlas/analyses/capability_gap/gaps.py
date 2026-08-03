"""Select architecture–capability mismatches (expectation gaps)."""

from __future__ import annotations

import pandas as pd

from bgcatlas.analyses.capability_gap.config import MIN_EXPECTATION, PILOT_GRAMMAR
from bgcatlas.analyses.capability_gap.expectation import domain_sets_by_bgc, has_any


def pilot_grammar_mask(domains: pd.DataFrame, bgc_ids: list[str]) -> pd.Series:
    """BGCs matching the pilot reductive-PKS grammar (DH* + KR present)."""
    dom_sets = domain_sets_by_bgc(domains)
    required = PILOT_GRAMMAR["required_present"]
    flags = []
    for bid in bgc_ids:
        dset = dom_sets.get(str(bid), set())
        ok = all(has_any(dset, opts) for opts in required)
        flags.append(ok)
    return pd.Series(flags, index=list(map(str, bgc_ids)), name="pilot_grammar")


def select_expectation_gaps(
    expectations: pd.DataFrame,
    domains: pd.DataFrame,
    *,
    capacity: str | None = None,
    min_expectation: float = MIN_EXPECTATION,
    myxo_only: bool = True,
    require_pilot_grammar: bool = True,
) -> pd.DataFrame:
    """Rows with high P(capacity|N) but missing observed capacity."""
    cap = capacity or PILOT_GRAMMAR["expected_capacity"]
    df = expectations.copy()
    df["bgc_id"] = df["bgc_id"].astype(str)
    if require_pilot_grammar:
        mask = pilot_grammar_mask(domains, df["bgc_id"].tolist())
        df = df[df["bgc_id"].map(mask)].copy()
    if myxo_only:
        df = df[df["clade"] == "Myxobacteria"].copy()

    pcol, hcol, gcol = f"p_{cap}", f"has_{cap}", f"gap_{cap}"
    df = df[(df[pcol] >= min_expectation) & (df[hcol] == 0)].copy()
    df["expected_capacity"] = cap
    df["expectation_confidence"] = df[pcol]
    df["capability_gap"] = df[gcol]
    df["gap_type"] = "missing_catalytic_capability"
    df["chemistry"] = PILOT_GRAMMAR["chemistry"]
    df = df.sort_values(["capability_gap", "expectation_confidence"], ascending=False)
    return df.reset_index(drop=True)
