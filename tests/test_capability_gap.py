"""Smoke tests for capability-gap inference."""

from __future__ import annotations

import pandas as pd
import pytest

from bgcatlas.analyses.capability_gap.calibrate import calibrate_gap_detection
from bgcatlas.analyses.capability_gap.cards import (
    UNCERTAINTY_INSUFFICIENT_FAMILY,
    build_cards,
)
from bgcatlas.analyses.capability_gap.expectation import has_capacity
from bgcatlas.paths import PROCESSED, REPORTS

RANKING = REPORTS / "novelty_ranking.csv"
COORDS = PROCESSED / "atlas_coords.csv"
DOMAINS = PROCESSED / "mibig_domains.csv"


def test_has_capacity_er():
    assert has_capacity({"PKS_ER", "PKS_KS"}, "enoyl_reduction")
    assert not has_capacity({"PKS_KS", "PKS_KR"}, "enoyl_reduction")


def test_cards_split_expectation_and_candidate_confidence():
    """Cards expose two independent uncertainties — never a single overall score."""
    gaps = pd.DataFrame(
        [
            {
                "bgc_id": "BGC_TEST",
                "organism": "Sorangium cellulosum",
                "biosynth_class": "PKS",
                "expected_capacity": "enoyl_reduction",
                "chemistry": "enoyl reduction (ER)",
                "expectation_confidence": 0.9,
                "capability_gap": 0.9,
                "gap_type": "missing_catalytic_capability",
            }
        ]
    )
    families = pd.DataFrame(
        [
            {
                "family_key": "Kinase",
                "association_score": 0.2,
                "evidence_cooccurrence": False,
                "evidence_control_depleted": False,
                "evidence_myxo_biased": False,
                "evidence_synteny": False,
            }
        ]
    )
    cards = build_cards(gaps, families, calibration={"calibration_ok": True}, top_n=1)
    assert len(cards) == 1
    c = cards[0]
    assert "overall_confidence" not in c
    assert "evidence_stars" not in c
    assert c["hypothesis_id"] == "MYXO-001"
    assert c["architecture"]["expectation"]["confidence"] == 0.9
    assert c["candidate"]["confidence"] < c["architecture"]["expectation"]["confidence"]
    assert c["candidate"]["family_key"] == "domain_bag__Kinase"
    assert c["candidate"]["representation"] == "domain_bag"
    assert UNCERTAINTY_INSUFFICIENT_FAMILY in c["candidate"]["uncertainty_reasons"]
    assert "expectation_supported" in c["hypothesis_status"]
    assert "candidate_unresolved" in c["hypothesis_status"]
    assert "interpretation" in c and c["interpretation"]["summary"]


@pytest.mark.skipif(
    not (RANKING.exists() and COORDS.exists() and DOMAINS.exists()),
    reason="processed atlas artifacts missing",
)
def test_calibration_separates_pilot_from_nonpilot():
    ranking = pd.read_csv(RANKING)
    domains = pd.read_csv(DOMAINS)
    coords = pd.read_csv(COORDS)
    cal = calibrate_gap_detection(ranking, domains, coords)
    assert cal["n_annotated_positives"] > 0
    assert cal["mean_p_annotated_positives"] > cal["mean_p_non_pilot"]
    assert cal["p_annotated_vs_nonpilot_winrate"] >= 0.85
