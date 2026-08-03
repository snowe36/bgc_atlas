"""Calibration: would known capacities be flagged if their annotations were absent?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bgcatlas.analyses.capability_gap.config import MIN_EXPECTATION, PILOT_GRAMMAR
from bgcatlas.analyses.capability_gap.expectation import (
    attach_neighborhood_expectations,
    build_inventory,
)
from bgcatlas.analyses.capability_gap.gaps import pilot_grammar_mask


def calibrate_gap_detection(
    ranking: pd.DataFrame,
    domains: pd.DataFrame,
    coords: pd.DataFrame,
    *,
    capacity: str | None = None,
    min_expectation: float = MIN_EXPECTATION,
) -> dict:
    """Calibrate conditional expectations without destroying the neighborhood prior.

    Important: do **not** mask capacity on all positives before estimating
    P(capacity|N) — that erases the signal from the neighborhood. Instead:

    1. Fit P(capacity|architecture neighborhood) on the observed atlas.
    2. For BGCs that *have* the annotated capacity, ask: if that annotation
       were missing, would P(capacity|N) still clear the gap threshold?
    3. Compare those conditional probabilities to true gaps and to non-pilot BGCs.

    This is the counterfactual: \"would the engine have noticed?\"
    """
    cap = capacity or PILOT_GRAMMAR["expected_capacity"]
    inv = build_inventory(ranking, domains)
    exp = attach_neighborhood_expectations(inv, coords, capacities=[cap])
    grammar = pilot_grammar_mask(domains, exp["bgc_id"].tolist())
    pilot = exp[exp["bgc_id"].map(grammar)].copy()
    non_pilot = exp[~exp["bgc_id"].map(grammar)].copy()

    pcol, hcol = f"p_{cap}", f"has_{cap}"
    positives = pilot[pilot[hcol] == 1]
    true_gaps = pilot[pilot[hcol] == 0]

    # Counterfactual gap score if annotation were absent: gap ≈ P(capacity|N)
    pos_p = positives[pcol].to_numpy()
    gap_p = true_gaps[pcol].to_numpy()
    null_p = non_pilot[pcol].to_numpy()

    pos_detect = float((positives[pcol] >= min_expectation).mean()) if len(positives) else None
    true_gap_detect = float((true_gaps[pcol] >= min_expectation).mean()) if len(true_gaps) else None

    def _winrate(a: np.ndarray, b: np.ndarray) -> float | None:
        if len(a) == 0 or len(b) == 0:
            return None
        return float(np.mean([np.mean(x > b) for x in a]))

    win_vs_nonpilot = _winrate(pos_p, null_p)
    # Positives should look *more* ER-expected than true gaps only if ER clusters;
    # more important: both pilot classes >> non-pilot.
    win_gaps_vs_nonpilot = _winrate(gap_p, null_p)

    # Gate on separation quality, not a single hard threshold: atlas neighborhoods
    # that contain reductive-PKS+ER should assign high P(ER|N) relative to non-pilot.
    mean_pos = float(pos_p.mean()) if len(pos_p) else 0.0
    mean_null = float(null_p.mean()) if len(null_p) else 0.0
    calibration_ok = bool(
        win_vs_nonpilot is not None
        and win_vs_nonpilot >= 0.85
        and mean_pos >= 0.5
        and (mean_pos - mean_null) >= 0.35
        and pos_detect is not None
        and pos_detect >= 0.35
    )

    return {
        "capacity": cap,
        "n_annotated_positives": int(len(positives)),
        "n_true_gaps_in_grammar": int(len(true_gaps)),
        "n_non_pilot": int(len(non_pilot)),
        "counterfactual_detection_rate_on_annotated": pos_detect,
        "detection_rate_on_true_gaps": true_gap_detect,
        "mean_p_annotated_positives": float(pos_p.mean()) if len(pos_p) else None,
        "mean_p_true_gaps": float(gap_p.mean()) if len(gap_p) else None,
        "mean_p_non_pilot": float(null_p.mean()) if len(null_p) else None,
        "p_annotated_vs_nonpilot_winrate": win_vs_nonpilot,
        "p_true_gaps_vs_nonpilot_winrate": win_gaps_vs_nonpilot,
        "min_expectation": min_expectation,
        "calibration_ok": calibration_ok,
        "note": (
            "Counterfactual calibration: estimate P(capacity|neighborhood) on the observed "
            "atlas, then ask whether annotated-positive pilot BGCs would clear the gap "
            "threshold if their capacity annotation were absent. Masking all positives "
            "before fitting P(.) is invalid — it erases the neighborhood prior."
        ),
    }
