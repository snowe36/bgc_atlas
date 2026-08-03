"""Run capability-gap inference (pilot) end-to-end."""

from __future__ import annotations

import json
import logging

import pandas as pd

from bgcatlas.analyses.capability_gap.calibrate import calibrate_gap_detection
from bgcatlas.analyses.capability_gap.candidates import rank_candidate_families
from bgcatlas.analyses.capability_gap.cards import build_cards, write_cards_json, write_cards_markdown
from bgcatlas.analyses.capability_gap.config import (
    CALIBRATION_FREEZE_ATOL,
    FROZEN_CALIBRATION_BASELINE,
    MIN_EXPECTATION,
    PILOT_GRAMMAR,
    SIZE_MAX,
    TOP_HYPOTHESES,
)
from bgcatlas.analyses.capability_gap.evaluation import (
    cluster_contains_known_ER,
    level2_recovery_metrics,
)
from bgcatlas.analyses.capability_gap.expectation import (
    attach_neighborhood_expectations,
    build_inventory,
)
from bgcatlas.analyses.capability_gap.gaps import select_expectation_gaps
from bgcatlas.analyses.capability_gap.homologs import load_or_build_clusters
from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)
OUT = REPORTS / "capability_gap"


def assert_calibration_frozen(calibration: dict) -> bool:
    """Level-1 freeze: candidate upgrades must not change ER-masking calibration."""
    ok = True
    for key, expected in FROZEN_CALIBRATION_BASELINE.items():
        got = calibration.get(key)
        if isinstance(expected, float):
            if got is None or abs(float(got) - expected) > CALIBRATION_FREEZE_ATOL:
                LOG.error(
                    "Calibration freeze violated: %s expected %s got %s",
                    key,
                    expected,
                    got,
                )
                ok = False
        elif got != expected:
            LOG.error(
                "Calibration freeze violated: %s expected %s got %s",
                key,
                expected,
                got,
            )
            ok = False
    return ok


def run_capability_gap(
    *,
    size_max: int | None = SIZE_MAX,
    min_expectation: float = MIN_EXPECTATION,
    top_n: int = TOP_HYPOTHESES,
    rebuild_clusters: bool = False,
) -> dict:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(REPORTS / "novelty_ranking.csv")
    domains = (
        pd.read_csv(PROCESSED / "mibig_domains.csv")
        if (PROCESSED / "mibig_domains.csv").exists()
        else pd.read_parquet(PROCESSED / "mibig_domains.parquet")
    )
    coords = pd.read_csv(PROCESSED / "atlas_coords.csv")

    capacity = PILOT_GRAMMAR["expected_capacity"]
    LOG.info("Calibration (mask known %s on pilot-grammar BGCs)", capacity)
    calibration = calibrate_gap_detection(
        ranking, domains, coords, capacity=capacity, min_expectation=min_expectation
    )
    calibration_unchanged = assert_calibration_frozen(calibration)
    calibration["calibration_unchanged_vs_frozen_baseline"] = calibration_unchanged
    (OUT / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    det = calibration.get("counterfactual_detection_rate_on_annotated")
    win = calibration.get("p_annotated_vs_nonpilot_winrate")
    if not calibration.get("calibration_ok"):
        LOG.warning(
            "Calibration gate not OK (counterfactual_detection=%s winrate=%s). "
            "Emitting cards as exploratory.",
            det,
            win,
        )
    else:
        LOG.info(
            "Calibration OK: counterfactual_detection=%.2f winrate=%.2f frozen=%s",
            float(det),
            float(win),
            calibration_unchanged,
        )

    LOG.info("Conditional expectations P(capacity|architecture neighborhood)")
    inventory = build_inventory(ranking, domains, size_max=size_max)
    expectations = attach_neighborhood_expectations(
        inventory, coords, capacities=[capacity]
    )
    expectations.to_csv(OUT / "expectations.csv", index=False)

    gaps = select_expectation_gaps(
        expectations,
        domains,
        capacity=capacity,
        min_expectation=min_expectation,
        myxo_only=True,
        require_pilot_grammar=True,
    )
    gaps.to_csv(OUT / "expectation_gaps.csv", index=False)
    LOG.info("Myxo pilot gaps: %d", len(gaps))

    from bgcatlas.analyses.capability_gap.gaps import pilot_grammar_mask

    satisfied = expectations.copy()
    gm = pilot_grammar_mask(domains, satisfied["bgc_id"].tolist())
    satisfied = satisfied[
        satisfied["bgc_id"].map(gm)
        & (satisfied[f"has_{capacity}"] == 1)
        & (satisfied[f"p_{capacity}"] >= min_expectation)
    ]

    LOG.info("Homolog90 clusters (Level-2 candidate identity)")
    clusters = load_or_build_clusters(rebuild=rebuild_clusters)

    LOG.info("Ranking candidate homolog families for gap pathways")
    # Keep all gap-associated families for per-BGC attribution; CSV shows top 25.
    families_all = rank_candidate_families(
        gaps,
        satisfied,
        domains,
        ranking,
        capacity=capacity,
        clusters=clusters,
        top_n=50_000,
        exclude_same_bgc_capacity=True,
    )
    families_all.head(25).to_csv(OUT / "candidate_families.csv", index=False)
    families = families_all

    er_pos = cluster_contains_known_ER(clusters, domains, capacity=capacity)
    level2 = level2_recovery_metrics(
        gaps, families, domains, clusters, capacity=capacity
    )
    level2["calibration_unchanged"] = bool(calibration_unchanged)
    (OUT / "candidate_level2_metrics.json").write_text(
        json.dumps(level2, indent=2), encoding="utf-8"
    )
    LOG.info(
        "Level-2 metrics: recall@1=%.3f recall@5=%.3f mrr=%.3f enrichment@5=%.2f scored=%d",
        level2.get("recall_at_1", 0),
        level2.get("recall_at_5", 0),
        level2.get("mrr", 0),
        level2.get("enrichment_at_5", 0),
        level2.get("n_gap_bgcs_scored", 0),
    )

    cards = build_cards(
        gaps,
        families,
        calibration=calibration,
        domains=domains,
        clusters=clusters,
        er_positive_families=er_pos,
        capacity=capacity,
        top_n=top_n,
    )
    write_cards_json(cards, OUT / "hypothesis_cards.json")
    report = REPORTS / "capability_gap_hypotheses.md"
    write_cards_markdown(cards, report, calibration=calibration)
    (OUT / "capability_gap_hypotheses.md").write_text(
        report.read_text(encoding="utf-8"), encoding="utf-8"
    )

    summary = {
        "pilot": PILOT_GRAMMAR["name"],
        "capacity": capacity,
        "candidate_representation": "homolog90",
        "calibration": calibration,
        "calibration_unchanged": bool(calibration_unchanged),
        "level2_metrics": level2,
        "n_expectation_gaps_myxo": int(len(gaps)),
        "n_cards": int(len(cards)),
        "n_homolog_clusters": int(clusters["cluster_id"].nunique()) if len(clusters) else 0,
        "top_family": families.iloc[0]["family_key"] if len(families) else None,
        "report": str(report),
        "figures_dir": str(FIGURES),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Wrote %s (%d cards)", report, len(cards))
    return summary


def run_capability_gap_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Capability-gap inference: P(capacity|architecture neighborhood) → "
            "homolog90 family hypotheses (pilot: reductive PKS / enoyl reduction)."
        )
    )
    parser.add_argument("--size-max", type=int, default=SIZE_MAX)
    parser.add_argument("--min-expectation", type=float, default=MIN_EXPECTATION)
    parser.add_argument("--top-n", type=int, default=TOP_HYPOTHESES)
    parser.add_argument(
        "--rebuild-clusters",
        action="store_true",
        help="Re-run MMseqs2 homolog90 clustering (ignore cache).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    summary = run_capability_gap(
        size_max=args.size_max,
        min_expectation=args.min_expectation,
        top_n=args.top_n,
        rebuild_clusters=args.rebuild_clusters,
    )
    l2 = summary.get("level2_metrics") or {}
    print(
        f"calibration_ok={summary['calibration'].get('calibration_ok')} "
        f"calibration_unchanged={summary.get('calibration_unchanged')} "
        f"gaps={summary['n_expectation_gaps_myxo']} "
        f"cards={summary['n_cards']} "
        f"top_family={summary['top_family']} "
        f"recall@5={l2.get('recall_at_5')} "
        f"enrichment@5={l2.get('enrichment_at_5')} "
        f"report={summary['report']}"
    )
    return 0
