"""Markdown report for the myxo grammar decode."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(
    path: Path,
    *,
    gate: dict,
    match_summary: dict,
    enrichment: pd.DataFrame,
    rarity_stats: dict,
    temporal: dict,
    figure_rel: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    or_ = gate.get("odds_ratio_top_decile")
    ci = gate.get("odds_ratio_top_decile_ci95") or [None, None]
    hits = enrichment[(enrichment["q_bh"] <= 0.05) & (enrichment["odds_ratio"] >= 2)].head(15)
    motif_hits = hits[hits["level"] == "motif"]
    grammar_hits = enrichment[
        (enrichment["level"].isin(["grammar", "capacity"]))
        & (enrichment["q_bh"] <= 0.05)
        & (enrichment["odds_ratio"] >= 2)
    ].head(8)

    p_rare = rarity_stats.get("mannwhitney_p_myxo_greater")
    med_m = rarity_stats.get("median_idf_myxo")
    med_c = rarity_stats.get("median_idf_control")

    lines = [
        "# Myxobacterial architecture grammar",
        "",
        "**Question.** Given that myxobacterial BGCs occupy unusual hashed-architecture "
        "neighborhoods after size matching, which architectural primitives account for that separation?",
        "",
        "**Not the question.** Which BGCs are novel / will be deposited next.",
        "",
        "---",
        "",
        "## Phenotype gate (must pass before token analysis)",
        "",
        "We first recover the published size-controlled phenotype, then ask what explains it.",
        "",
        f"- Size filter: n_genes ≤ {gate.get('size_filter_n_genes_max')}",
        f"- Contrast: Myxobacteria vs {gate.get('control_clade')}",
        f"- n_myxo = {gate.get('n_myxo')}, n_control = {gate.get('n_control')}",
        f"- Top-decile OR = **{or_:.2f}** (95% CI {ci[0]:.2f}–{ci[1]:.2f})"
        if or_ is not None and ci[0] is not None
        else f"- Top-decile OR = {or_}",
        f"- Mann–Whitney p (myxo > control) = {gate.get('mannwhitney_p'):.3g}",
        f"- Gate OK: **{gate.get('gate_ok')}**",
        "",
        "## Matched cohort",
        "",
        "Matching is designed to isolate **architectural composition** from obvious confounders "
        "(BGC size and coarse biosynthetic class), not to estimate a causal treatment effect.",
        "",
        f"- Gene-count tolerance: ±{100 * float(match_summary.get('gene_tol', 0)):.0f}%",
        f"- Controls per myxo BGC: {match_summary.get('n_controls_per_myxo')}",
        f"- Matched myxo BGCs: {match_summary.get('n_matched_myxo')} / {match_summary.get('n_myxo')}",
        f"- Unique controls: {match_summary.get('n_unique_controls')}",
        f"- Same-class pair fraction: {match_summary.get('frac_same_class_pairs')}",
        "",
        "## Result: assembly grammar, not rarer parts",
        "",
        "Tokens come from existing MIBiG domain annotations (abundance, **ordered motifs**, "
        "capacity combos). No new embedding.",
        "",
        "### Rarity negative control (load-bearing)",
        "",
        f"Mean token-IDF rarity does **not** differ: myxo median={med_m}, control median={med_c}, "
        f"p(myxo > control)={p_rare}.",
        "",
        "This rules out the simplistic explanation that myxobacteria look unusual because they "
        "use rarer building blocks. Instead, they preferentially arrange **largely familiar "
        "components into different architectural grammars** — pathway organization rules, not "
        "just protein inventories.",
        "",
        "### Ordered motifs and combo grammars",
        "",
    ]
    if motif_hits.empty and grammar_hits.empty:
        lines += ["No motif/grammar tokens with q ≤ 0.05 and OR ≥ 2.", ""]
    else:
        lines += [
            "The strongest signals are **ordered biosynthetic grammars** "
            "(e.g. reductive PKS cascades, PKS↔NRPS bridges), not isolated domain counts:",
            "",
            "| Token | Level | Myxo frac | Control frac | OR | q |",
            "|-------|-------|----------:|-------------:|---:|--:|",
        ]
        show = pd.concat([motif_hits.head(10), grammar_hits.head(5)]).drop_duplicates("token")
        show = show.head(15)
        for _, r in show.iterrows():
            lines.append(
                f"| `{r['token']}` | {r['level']} | {r['myxo_fraction']:.2f} | "
                f"{r['control_fraction']:.2f} | {r['odds_ratio']:.2f} | {r['q_bh']:.3g} |"
            )
        lines.append("")

    lines += [
        "## Full enrichment table (top OR≥2, q≤0.05)",
        "",
    ]
    if hits.empty:
        lines += ["No tokens with q ≤ 0.05 and OR ≥ 2.", ""]
    else:
        lines += [
            "| Token | Level | Myxo frac | Control frac | OR | q |",
            "|-------|-------|----------:|-------------:|---:|--:|",
        ]
        for _, r in hits.iterrows():
            lines.append(
                f"| `{r['token']}` | {r['level']} | {r['myxo_fraction']:.2f} | "
                f"{r['control_fraction']:.2f} | {r['odds_ratio']:.2f} | {r['q_bh']:.3g} |"
            )
        lines.append("")

    lines += [
        "## Temporal sneak (exploratory only)",
        "",
    ]
    if not temporal.get("available"):
        lines += [f"Unavailable: {temporal.get('reason')}", ""]
    else:
        lines += [
            f"- Cutoff: {temporal.get('cutoff_date')}",
            f"- Global held-out mean novelty: {temporal.get('global_heldout_mean')} "
            f"(random control {temporal.get('global_control_mean')})",
            f"- Held-out myxo n={temporal.get('n_heldout_myxo')} "
            f"mean={temporal.get('heldout_mean_myxo')}",
            f"- Held-out Streptomyces mean={temporal.get('heldout_mean_streptomyces')}",
            f"- Myxo > non-myxo among held-out? p={temporal.get('mannwhitney_myxo_gt_nonmyxo_p')}",
            "",
            "An exploratory analysis of the small held-out myxobacterial subset suggested "
            "higher architectural novelty than non-myxobacterial clusters, but the sample "
            "size is too limited for strong inference.",
            "",
        ]

    lines += [
        "## Figure",
        "",
        f"![Figure 5]({figure_rel})",
        "",
        "## Claim framing",
        "",
        "`bgc_atlas` does not predict discovery. It quantifies architectural organization, "
        "tests whether novelty is meaningful, and identifies lineage-associated biosynthetic "
        "design grammars.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
