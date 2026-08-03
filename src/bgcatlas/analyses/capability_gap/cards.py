"""Hypothesis cards — claims with calibrated, independent uncertainties."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bgcatlas.analyses.capability_gap.candidates import best_family_for_bgc
from bgcatlas.analyses.capability_gap.config import MIN_EXPECTATION, PILOT_GRAMMAR

# Candidate maturity: v0 domain bags cannot clear this bar.
CANDIDATE_SUPPORTED_MIN = 0.6

# First-class uncertainty codes (actionable development targets).
UNCERTAINTY_INSUFFICIENT_FAMILY = "insufficient_family_resolution"
UNCERTAINTY_MISSING_HOMOLOGY = "missing_homology_cluster"
UNCERTAINTY_MULTIPLE = "multiple_plausible_candidates"
UNCERTAINTY_NO_SYNTENY = "no_syntenic_support"
UNCERTAINTY_NO_CANDIDATE = "no_candidate_in_cluster"


def _stars(n: int) -> str:
    n = max(0, min(5, int(n)))
    return "★" * n + "☆" * (5 - n)


def _stars_from_confidence(c: float) -> str:
    """Map [0, 1] confidence to 0–5 stars (round half up)."""
    return _stars(int(round(max(0.0, min(1.0, float(c))) * 5)))


def _domain_bag_key(family_key: str) -> str:
    """Prefix so cards never look like catalytic-family accessions."""
    if family_key.startswith("domain_bag__"):
        return family_key
    return f"domain_bag__{family_key}"


def _expectation_evidence() -> list[str]:
    return [
        "enriched_DH_KR_grammar",
        "kNN_capacity_probability",
        "matched_myxo_manifold",
    ]


def _candidate_evidence_tags(fam: dict | None) -> list[str]:
    if not fam:
        return []
    tags = []
    if fam.get("evidence_synteny"):
        tags.append("local_synteny")
    if fam.get("evidence_cooccurrence"):
        tags.append("gap_cooccurrence")
    if fam.get("evidence_control_depleted"):
        tags.append("control_depleted")
    if fam.get("evidence_myxo_biased"):
        tags.append("myxo_cooccurrence")
    return tags


def _candidate_confidence(fam: dict | None) -> float:
    """Association strength only — capped because v0 keys are domain bags.

    Expectation confidence is never folded in here.
    """
    if not fam:
        return 0.0
    bits = [
        bool(fam.get("evidence_cooccurrence")),
        bool(fam.get("evidence_control_depleted")),
        bool(fam.get("evidence_myxo_biased")),
        bool(fam.get("evidence_synteny")),
    ]
    # Four binary axes → at most ~0.40 before representation penalty.
    raw = 0.05 + 0.08 * sum(bits)
    assoc = fam.get("association_score")
    if assoc is not None:
        raw = max(raw, min(0.45, 0.1 + 0.25 * float(assoc)))
    # Hard cap: domain-bag identity cannot support mechanistic attribution.
    return round(min(raw, 0.35), 3)


def _uncertainty_reasons(fam: dict | None, *, n_plausible: int) -> list[str]:
    reasons: list[str] = []
    if fam is None:
        reasons.append(UNCERTAINTY_NO_CANDIDATE)
        reasons.append(UNCERTAINTY_INSUFFICIENT_FAMILY)
        reasons.append(UNCERTAINTY_MISSING_HOMOLOGY)
        return reasons
    # v0 representation
    reasons.append(UNCERTAINTY_INSUFFICIENT_FAMILY)
    reasons.append(UNCERTAINTY_MISSING_HOMOLOGY)
    if n_plausible >= 2:
        reasons.append(UNCERTAINTY_MULTIPLE)
    if not fam.get("evidence_synteny"):
        reasons.append(UNCERTAINTY_NO_SYNTENY)
    return reasons


def _hypothesis_status(exp_conf: float, cand_conf: float, reasons: list[str]) -> list[str]:
    status = []
    if exp_conf >= MIN_EXPECTATION:
        status.append("expectation_supported")
    else:
        status.append("expectation_weak")
    if (
        cand_conf >= CANDIDATE_SUPPORTED_MIN
        and UNCERTAINTY_INSUFFICIENT_FAMILY not in reasons
        and UNCERTAINTY_NO_CANDIDATE not in reasons
    ):
        status.append("candidate_supported")
    else:
        status.append("candidate_unresolved")
    return status


def _interpretation_summary(
    *,
    chemistry: str,
    exp_conf: float,
    fam: dict | None,
    reasons: list[str],
) -> str:
    exp_clause = (
        f"Architecture strongly predicts {chemistry} "
        f"(expectation confidence {exp_conf:.2f})"
    )
    if fam is None:
        return (
            f"{exp_clause}, but no non-core candidate family scored in-cluster. "
            "Capability gap is supported; protein attribution is unresolved."
        )
    if UNCERTAINTY_INSUFFICIENT_FAMILY in reasons:
        return (
            f"{exp_clause}, but v0 family resolution is insufficient to identify "
            "the responsible catalytic family. Domain-bag keys are functional labels, "
            "not Pfam/UniRef catalytic families."
        )
    return (
        f"{exp_clause}; candidate family `{fam.get('family_key')}` carries "
        f"association evidence but attribution remains provisional."
    )


def _n_plausible_in_bgc(
    bgc_id: str,
    families: pd.DataFrame,
    domains: pd.DataFrame,
    *,
    top_score_frac: float = 0.85,
) -> int:
    """How many ranked families co-occur in this BGC near the top score."""
    from bgcatlas.analyses.capability_gap.candidates import _cds_families

    if families.empty:
        return 0
    present = _cds_families(domains)
    present = present[present["bgc_id"].astype(str) == str(bgc_id)]
    if present.empty:
        return 0
    keys = set(present["family_key"])
    hit = families[families["family_key"].isin(keys)]
    if hit.empty:
        return 0
    top = float(hit.iloc[0]["association_score"])
    if top <= 0:
        return int(len(hit))
    near = hit[hit["association_score"] >= top * top_score_frac]
    return int(len(near))


def build_cards(
    gaps: pd.DataFrame,
    families: pd.DataFrame,
    *,
    calibration: dict,
    domains: pd.DataFrame | None = None,
    top_n: int = 6,
) -> list[dict]:
    """Assemble dual-confidence hypothesis claims (no single overall score)."""
    if gaps.empty:
        return []
    cards = []
    for i, g in gaps.head(top_n).reset_index(drop=True).iterrows():
        hid = f"MYXO-{i+1:03d}"
        fam = None
        n_plausible = 0
        if domains is not None and len(families):
            fam = best_family_for_bgc(str(g["bgc_id"]), families, domains)
            n_plausible = _n_plausible_in_bgc(str(g["bgc_id"]), families, domains)
        elif len(families):
            fam = families.iloc[0].to_dict()

        exp_conf = float(g["expectation_confidence"])
        gap_p = float(g["capability_gap"])
        chemistry = str(g.get("chemistry") or PILOT_GRAMMAR["chemistry"])
        cand_conf = _candidate_confidence(fam)
        reasons = _uncertainty_reasons(fam, n_plausible=n_plausible)
        status = _hypothesis_status(exp_conf, cand_conf, reasons)
        evidence_tags = _candidate_evidence_tags(fam)

        raw_key = fam["family_key"] if fam else None
        family_key = _domain_bag_key(raw_key) if raw_key else None

        card = {
            "hypothesis_id": hid,
            "bgc_id": g["bgc_id"],
            "organism": g.get("organism"),
            "biosynth_class": g.get("biosynth_class"),
            "architecture": {
                "neighborhood": PILOT_GRAMMAR["name"],
                "expectation": {
                    "capability": g["expected_capacity"],
                    "chemistry": chemistry,
                    "confidence": exp_conf,
                    "stars": _stars_from_confidence(exp_conf),
                    "evidence": _expectation_evidence(),
                },
            },
            "capability_gap": {
                "gap_type": g.get("gap_type", "missing_catalytic_capability"),
                "capability_probability": gap_p,
                "observed_inventory": False,
                "observed_inventory_note": (
                    f"No annotated {str(g['expected_capacity']).replace('_', ' ')}"
                ),
                "stars": _stars_from_confidence(gap_p),
            },
            "candidate": {
                "family_key": family_key,
                "representation": "domain_bag",
                "confidence": cand_conf,
                "stars": _stars_from_confidence(cand_conf),
                "evidence": evidence_tags,
                "limitation": (
                    ["domain_bag_not_catalytic_family"] if fam else ["no_family_scored"]
                ),
                "uncertainty_reasons": reasons,
            },
            "hypothesis_status": status,
            "interpretation": {
                "summary": _interpretation_summary(
                    chemistry=chemistry,
                    exp_conf=exp_conf,
                    fam=fam,
                    reasons=reasons,
                ),
            },
            "validation": PILOT_GRAMMAR["validation_hint"],
            # Future atlas evidence slots (empty in v0 — not a collapsed score).
            "future_evidence": {
                "active_site_similarity": False,
                "metabolomics": False,
                "bioactivity_prior": False,
            },
            "priority": i + 1,
            "calibration_gate_ok": bool(calibration.get("calibration_ok")),
        }
        cards.append(card)
    return cards


def write_cards_markdown(cards: list[dict], path: Path, *, calibration: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Capability-gap hypotheses (pilot: reductive PKS)",
        "",
        "Inference layer over the atlas: **conditional capability expectations**, not global "
        "novelty. Each card is a **claim with two independent uncertainties** — expectation "
        "(does this pathway need this chemistry?) vs candidate (do we know which protein "
        "performs it?). There is no single overall confidence score.",
        "",
        "## Calibration gate",
        "",
        f"- Counterfactual detection (annotated positives): "
        f"{calibration.get('counterfactual_detection_rate_on_annotated')}",
        f"- Detection on true gaps: {calibration.get('detection_rate_on_true_gaps')}",
        f"- P(annotated) vs non-pilot win-rate: "
        f"{calibration.get('p_annotated_vs_nonpilot_winrate')}",
        f"- P(true gaps) vs non-pilot win-rate: "
        f"{calibration.get('p_true_gaps_vs_nonpilot_winrate')}",
        f"- Calibration OK: **{calibration.get('calibration_ok')}**",
        "",
        calibration.get("note", ""),
        "",
        "---",
        "",
    ]
    if not cards:
        lines += ["No hypothesis cards emitted.", ""]
    for c in cards:
        arch = c["architecture"]
        exp = arch["expectation"]
        gap = c["capability_gap"]
        cand = c["candidate"]
        status = ", ".join(c["hypothesis_status"])
        lines += [
            f"## Hypothesis {c['hypothesis_id']}",
            "",
            f"**Status.** `{status}`",
            "",
            f"**BGC** `{c['bgc_id']}` · {c.get('organism')} · {c.get('biosynth_class')}",
            "",
            f"| | Confidence |",
            f"|---|---|",
            f"| **Expectation** (architecture → chemistry) | {exp['stars']} ({exp['confidence']:.2f}) |",
            f"| **Candidate** (protein attribution) | {cand['stars']} ({cand['confidence']:.2f}) |",
            "",
            f"**Interpretation.** {c['interpretation']['summary']}",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Architecture neighborhood | {arch['neighborhood']} |",
            f"| Expected capability | {exp['capability']} |",
            f"| Chemistry | {exp['chemistry']} |",
            f"| Gap type | {gap['gap_type']} |",
            f"| Observed inventory | {gap['observed_inventory_note']} |",
            f"| Candidate family | {('`' + cand['family_key'] + '`') if cand['family_key'] else '_(none)_'} |",
            f"| Candidate representation | {cand['representation']} |",
            f"| Priority | {c['priority']} |",
            "",
            "Expectation evidence:",
            "",
        ]
        for e in exp["evidence"]:
            lines.append(f"- `{e}`")
        lines += ["", "Candidate evidence:", ""]
        if cand["evidence"]:
            for e in cand["evidence"]:
                lines.append(f"- `{e}`")
        else:
            lines.append("- _(none)_")
        lines += ["", "Candidate uncertainty reasons:", ""]
        for r in cand["uncertainty_reasons"]:
            lines.append(f"- `{r}`")
        lines += [
            "",
            f"**Validation.** {c['validation']}",
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cards_json(cards: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    return path
