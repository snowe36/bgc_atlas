"""Smoke tests for capability-gap inference (Level 1 + Level 2)."""

from __future__ import annotations

import pandas as pd
import pytest

from bgcatlas.analyses.capability_gap.calibrate import calibrate_gap_detection
from bgcatlas.analyses.capability_gap.candidates import (
    RANKING_FEATURE_COLS,
    cds_families_homolog,
    rank_candidate_families,
)
from bgcatlas.analyses.capability_gap.cards import build_cards
from bgcatlas.analyses.capability_gap.config import FROZEN_CALIBRATION_BASELINE
from bgcatlas.analyses.capability_gap.evaluation import (
    cluster_contains_known_ER,
    level2_recovery_metrics,
)
from bgcatlas.analyses.capability_gap.expectation import has_capacity
from bgcatlas.analyses.capability_gap.run import assert_calibration_frozen
from bgcatlas.paths import PROCESSED, REPORTS

RANKING = REPORTS / "novelty_ranking.csv"
COORDS = PROCESSED / "atlas_coords.csv"
DOMAINS = PROCESSED / "mibig_domains.csv"


def _toy_clusters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bgc_id": "BGC_GAP",
                "locus_tag": "geneA",
                "cluster_id": "clu_00001",
                "family_key": "homolog90__clu_00001",
                "n_members": 5,
                "rep_locus": "geneA",
                "rep_bgc_id": "BGC_GAP",
            },
            {
                "bgc_id": "BGC_ER",
                "locus_tag": "er1",
                "cluster_id": "clu_00001",
                "family_key": "homolog90__clu_00001",
                "n_members": 5,
                "rep_locus": "geneA",
                "rep_bgc_id": "BGC_GAP",
            },
            {
                "bgc_id": "BGC_GAP",
                "locus_tag": "geneB",
                "cluster_id": "clu_00002",
                "family_key": "homolog90__clu_00002",
                "n_members": 2,
                "rep_locus": "geneB",
                "rep_bgc_id": "BGC_GAP",
            },
            {
                "bgc_id": "BGC_CTRL",
                "locus_tag": "c1",
                "cluster_id": "clu_00002",
                "family_key": "homolog90__clu_00002",
                "n_members": 2,
                "rep_locus": "geneB",
                "rep_bgc_id": "BGC_GAP",
            },
        ]
    )


def _toy_domains() -> pd.DataFrame:
    rows = [
        ("BGC_GAP", "geneA", "Redox", 1),
        ("BGC_GAP", "geneB", "Hydrolase", 2),
        ("BGC_GAP", "ks1", "PKS_KS", 0),
        ("BGC_ER", "er1", "PKS_ER", 3),
        ("BGC_ER", "ks1", "PKS_KS", 0),
        ("BGC_CTRL", "c1", "Hydrolase", 1),
        ("BGC_CTRL", "erx", "PKS_ER", 2),
    ]
    return pd.DataFrame(
        [
            {
                "bgc_id": b,
                "feature_type": "domain",
                "locus_tag": loc,
                "domain_id": d,
                "gene_order": go,
                "product": "",
            }
            for b, loc, d, go in rows
        ]
    )


def test_has_capacity_er():
    assert has_capacity({"PKS_ER", "PKS_KS"}, "enoyl_reduction")
    assert not has_capacity({"PKS_KS", "PKS_KR"}, "enoyl_reduction")


def test_ranker_never_emits_er_positive_features():
    clusters = _toy_clusters()
    domains = _toy_domains()
    gaps = pd.DataFrame([{"bgc_id": "BGC_GAP"}])
    controls = pd.DataFrame([{"bgc_id": "BGC_CTRL"}])
    ranking = pd.DataFrame(
        [
            {"bgc_id": "BGC_GAP", "organism": "Sorangium cellulosum"},
            {"bgc_id": "BGC_CTRL", "organism": "Streptomyces coelicolor"},
            {"bgc_id": "BGC_ER", "organism": "Sorangium cellulosum"},
        ]
    )
    fam = rank_candidate_families(
        gaps,
        controls,
        domains,
        ranking,
        capacity="enoyl_reduction",
        clusters=clusters,
        top_n=10,
    )
    assert len(fam)
    for c in fam.columns:
        assert "er_pos" not in c.lower()
        assert c != "cluster_has_ER"
        assert "known_ER" not in c
    assert set(RANKING_FEATURE_COLS).issubset(set(fam.columns))
    assert all(str(k).startswith("homolog90__") for k in fam["family_key"])


def test_same_bgc_er_cds_excluded_from_candidates():
    clusters = _toy_clusters()
    # Add an ER CDS on the gap BGC — must be excluded (trivial leakage)
    clusters = pd.concat(
        [
            clusters,
            pd.DataFrame(
                [
                    {
                        "bgc_id": "BGC_GAP",
                        "locus_tag": "local_er",
                        "cluster_id": "clu_00099",
                        "family_key": "homolog90__clu_00099",
                        "n_members": 3,
                        "rep_locus": "local_er",
                        "rep_bgc_id": "BGC_GAP",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    domains = _toy_domains()
    domains = pd.concat(
        [
            domains,
            pd.DataFrame(
                [
                    {
                        "bgc_id": "BGC_GAP",
                        "feature_type": "domain",
                        "locus_tag": "local_er",
                        "domain_id": "PKS_ER",
                        "gene_order": 9,
                        "product": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    present = cds_families_homolog(
        domains, clusters, capacity="enoyl_reduction", exclude_same_bgc_capacity=True
    )
    assert "local_er" not in set(present["locus_tag"])
    # ER homolog on another BGC still allowed via shared cluster clu_00001
    assert "homolog90__clu_00001" in set(present["family_key"])


def test_evaluation_metrics_keys():
    clusters = _toy_clusters()
    domains = _toy_domains()
    gaps = pd.DataFrame([{"bgc_id": "BGC_GAP"}])
    controls = pd.DataFrame([{"bgc_id": "BGC_CTRL"}])
    ranking = pd.DataFrame(
        [
            {"bgc_id": "BGC_GAP", "organism": "Sorangium cellulosum"},
            {"bgc_id": "BGC_CTRL", "organism": "Streptomyces coelicolor"},
            {"bgc_id": "BGC_ER", "organism": "Sorangium cellulosum"},
        ]
    )
    fam = rank_candidate_families(
        gaps,
        controls,
        domains,
        ranking,
        capacity="enoyl_reduction",
        clusters=clusters,
        top_n=10,
    )
    er_pos = cluster_contains_known_ER(clusters, domains)
    assert "homolog90__clu_00001" in er_pos
    metrics = level2_recovery_metrics(gaps, fam, domains, clusters)
    for key in (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "enrichment_at_5",
        "random_baseline_rate",
        "n_gap_bgcs_scored",
        "n_er_positive_clusters",
    ):
        assert key in metrics


def test_cards_homolog90_not_domain_bag():
    gaps = pd.DataFrame(
        [
            {
                "bgc_id": "BGC_GAP",
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
    clusters = _toy_clusters()
    domains = _toy_domains()
    ranking = pd.DataFrame(
        [
            {"bgc_id": "BGC_GAP", "organism": "Sorangium cellulosum"},
            {"bgc_id": "BGC_CTRL", "organism": "Streptomyces coelicolor"},
            {"bgc_id": "BGC_ER", "organism": "Sorangium cellulosum"},
        ]
    )
    fam = rank_candidate_families(
        gaps,
        pd.DataFrame([{"bgc_id": "BGC_CTRL"}]),
        domains,
        ranking,
        capacity="enoyl_reduction",
        clusters=clusters,
        top_n=10,
    )
    er_pos = cluster_contains_known_ER(clusters, domains)
    cards = build_cards(
        gaps,
        fam,
        calibration={"calibration_ok": True},
        domains=domains,
        clusters=clusters,
        er_positive_families=er_pos,
        top_n=1,
    )
    assert len(cards) == 1
    c = cards[0]
    assert c["candidate"]["representation"] == "homolog90"
    assert c["candidate"]["family_key"].startswith("homolog90__")
    assert "insufficient_family_resolution" not in c["candidate"]["uncertainty_reasons"]
    assert "expectation_supported" in c["hypothesis_status"]
    assert "discovery_readiness" in c
    assert "overall_confidence" not in c


def test_calibration_freeze_helper():
    assert assert_calibration_frozen(dict(FROZEN_CALIBRATION_BASELINE))
    bad = dict(FROZEN_CALIBRATION_BASELINE)
    bad["counterfactual_detection_rate_on_annotated"] = 0.1
    assert not assert_calibration_frozen(bad)


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
    assert assert_calibration_frozen(cal)
