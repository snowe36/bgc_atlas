"""Run capability-gap inference (pilot) end-to-end."""

from __future__ import annotations

import json
import logging

import pandas as pd

from bgcatlas.analyses.capability_gap.calibrate import calibrate_gap_detection
from bgcatlas.analyses.capability_gap.candidates import rank_candidate_families
from bgcatlas.analyses.capability_gap.cards import build_cards, write_cards_json, write_cards_markdown
from bgcatlas.analyses.capability_gap.config import (
    MIN_EXPECTATION,
    PILOT_GRAMMAR,
    SIZE_MAX,
    TOP_HYPOTHESES,
)
from bgcatlas.analyses.capability_gap.expectation import (
    attach_neighborhood_expectations,
    build_inventory,
)
from bgcatlas.analyses.capability_gap.gaps import select_expectation_gaps
from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)
OUT = REPORTS / "capability_gap"


def run_capability_gap(
    *,
    size_max: int | None = SIZE_MAX,
    min_expectation: float = MIN_EXPECTATION,
    top_n: int = TOP_HYPOTHESES,
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
            "Calibration OK: counterfactual_detection=%.2f winrate=%.2f",
            float(det),
            float(win),
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

    # Satisfied controls: pilot grammar + has capacity (family depletion evidence)
    from bgcatlas.analyses.capability_gap.gaps import pilot_grammar_mask

    satisfied = expectations.copy()
    gm = pilot_grammar_mask(domains, satisfied["bgc_id"].tolist())
    satisfied = satisfied[
        satisfied["bgc_id"].map(gm)
        & (satisfied[f"has_{capacity}"] == 1)
        & (satisfied[f"p_{capacity}"] >= min_expectation)
    ]

    LOG.info("Ranking candidate families for gap pathways")
    families = rank_candidate_families(
        gaps,
        satisfied,
        domains,
        ranking,
        capacity=capacity,
        top_n=25,
    )
    families.to_csv(OUT / "candidate_families.csv", index=False)

    cards = build_cards(
        gaps, families, calibration=calibration, domains=domains, top_n=top_n
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
        "calibration": calibration,
        "n_expectation_gaps_myxo": int(len(gaps)),
        "n_cards": int(len(cards)),
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
            "family-level mechanistic hypotheses (pilot: reductive PKS / enoyl reduction)."
        )
    )
    parser.add_argument("--size-max", type=int, default=SIZE_MAX)
    parser.add_argument("--min-expectation", type=float, default=MIN_EXPECTATION)
    parser.add_argument("--top-n", type=int, default=TOP_HYPOTHESES)
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
    )
    print(
        f"calibration_ok={summary['calibration'].get('calibration_ok')} "
        f"gaps={summary['n_expectation_gaps_myxo']} "
        f"cards={summary['n_cards']} "
        f"top_family={summary['top_family']} "
        f"report={summary['report']}"
    )
    return 0
