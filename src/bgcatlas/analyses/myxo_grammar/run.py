"""Run the myxo architecture-grammar decode end-to-end."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from bgcatlas.analyses.myxo_grammar.cohort import (
    SIZE_MAX_DEFAULT,
    build_matched_cohort,
    phenotype_gate,
)
from bgcatlas.analyses.myxo_grammar.enrichment import run_enrichment, temporal_myxo_contrast
from bgcatlas.analyses.myxo_grammar.figures import make_figure5
from bgcatlas.analyses.myxo_grammar.report import write_report
from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)

OUT_DIR = REPORTS / "myxo_grammar"
FIG_PNG = FIGURES / "fig_myxo_architecture_grammar.png"
FIG_PDF = FIGURES / "fig_myxo_architecture_grammar.pdf"


def run_myxo_grammar(
    *,
    size_max: int | None = SIZE_MAX_DEFAULT,
    gene_tol: float = 0.10,
    n_controls: int = 5,
    min_token_bgcs: int = 5,
) -> dict:
    """Decode which architectural primitives explain the myxo neighborhood shift."""
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(REPORTS / "novelty_ranking.csv")
    meta_path = PROCESSED / "feature_meta.csv"
    if not meta_path.exists():
        meta_path = PROCESSED / "feature_meta.parquet"
    meta = pd.read_csv(meta_path) if meta_path.suffix == ".csv" else pd.read_parquet(meta_path)

    domains_path = PROCESSED / "mibig_domains.csv"
    if domains_path.exists():
        domains = pd.read_csv(domains_path)
    else:
        domains = pd.read_parquet(PROCESSED / "mibig_domains.parquet")

    coords_path = PROCESSED / "atlas_coords.csv"
    coords = pd.read_csv(coords_path) if coords_path.exists() else None

    LOG.info("Phenotype gate (myxo vs Streptomyces, size-controlled)")
    gate = phenotype_gate(ranking, size_max=size_max)
    (OUT_DIR / "phenotype_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if not gate.get("gate_ok"):
        LOG.warning(
            "Phenotype gate did not reproduce expected OR range "
            "(got OR=%s p=%s). Continuing, but interpret cautiously.",
            gate.get("odds_ratio_top_decile"),
            gate.get("mannwhitney_p"),
        )
    else:
        LOG.info(
            "Gate OK: OR=%.2f (95%% CI %.2f–%.2f)",
            gate["odds_ratio_top_decile"],
            gate["odds_ratio_top_decile_ci95"][0],
            gate["odds_ratio_top_decile_ci95"][1],
        )

    LOG.info("Building size/class-matched cohort")
    cohort, matches, match_summary = build_matched_cohort(
        ranking,
        meta,
        size_max=size_max,
        gene_tol=gene_tol,
        n_controls=n_controls,
    )
    cohort.to_csv(OUT_DIR / "matched_cohort.csv", index=False)
    matches.to_csv(OUT_DIR / "match_pairs.csv", index=False)
    (OUT_DIR / "match_summary.json").write_text(
        json.dumps(match_summary, indent=2), encoding="utf-8"
    )

    LOG.info("Token enrichment on matched cohort")
    enrichment, rarity, enrich_meta = run_enrichment(
        domains, cohort, min_token_bgcs=min_token_bgcs
    )
    enrichment.to_csv(OUT_DIR / "token_enrichment.csv", index=False)
    rarity.rename("mean_token_idf").to_csv(OUT_DIR / "rarity_scores.csv", header=True)
    (OUT_DIR / "enrichment_meta.json").write_text(
        json.dumps(
            {
                "n_tokens": enrich_meta["n_tokens"],
                "rarity": enrich_meta["rarity"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    temporal = temporal_myxo_contrast(
        REPORTS / "temporal_holdout.json",
        REPORTS / "temporal_holdout_ranking.csv",
        ranking,
    )
    (OUT_DIR / "temporal_myxo.json").write_text(json.dumps(temporal, indent=2), encoding="utf-8")

    LOG.info("Figure 5")
    make_figure5(
        cohort,
        coords,
        enrichment,
        out_png=FIG_PNG,
        out_pdf=FIG_PDF,
    )

    report_path = REPORTS / "myxo_grammar_report.md"
    write_report(
        report_path,
        gate=gate,
        match_summary=match_summary,
        enrichment=enrichment,
        rarity_stats=enrich_meta["rarity"],
        temporal=temporal,
        figure_rel="figures/fig_myxo_architecture_grammar.png",
    )

    # Also stage under reports/myxo_grammar for the analyses/ layout
    report_path_local = OUT_DIR / "myxo_grammar_report.md"
    report_path_local.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "gate": gate,
        "match_summary": match_summary,
        "n_tokens_tested": int(enrich_meta["n_tokens"]),
        "n_significant_or_ge_2": int(
            ((enrichment["q_bh"] <= 0.05) & (enrichment["odds_ratio"] >= 2)).sum()
        )
        if len(enrichment)
        else 0,
        "rarity": enrich_meta["rarity"],
        "temporal": temporal,
        "figure": str(FIG_PNG),
        "report": str(report_path),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Wrote %s and %s", report_path, FIG_PNG)
    return summary


def run_myxo_grammar_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Decode myxobacterial architecture grammar: which tokens explain "
            "the size-matched neighborhood shift (not a novelty/discovery ranker)."
        )
    )
    parser.add_argument("--size-max", type=int, default=SIZE_MAX_DEFAULT)
    parser.add_argument("--gene-tol", type=float, default=0.10)
    parser.add_argument("--n-controls", type=int, default=5)
    parser.add_argument("--min-token-bgcs", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep -v useful without drowning in font-manager noise.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    summary = run_myxo_grammar(
        size_max=args.size_max,
        gene_tol=args.gene_tol,
        n_controls=args.n_controls,
        min_token_bgcs=args.min_token_bgcs,
    )
    print(
        f"gate_OR={summary['gate']['odds_ratio_top_decile']:.2f} "
        f"sig_tokens(OR≥2,q≤0.05)={summary['n_significant_or_ge_2']} "
        f"report={summary['report']}"
    )
    return 0
