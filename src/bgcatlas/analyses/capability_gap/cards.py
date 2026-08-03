"""Hypothesis cards — claims with calibrated, independent uncertainties."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bgcatlas.analyses.capability_gap.candidates import best_family_for_bgc, cds_families_homolog
from bgcatlas.analyses.capability_gap.config import MIN_EXPECTATION, PILOT_GRAMMAR
from bgcatlas.analyses.capability_gap.homologs import REPRESENTATION

CANDIDATE_SUPPORTED_MIN = 0.55

UNCERTAINTY_INSUFFICIENT_FAMILY = "insufficient_family_resolution"
UNCERTAINTY_MISSING_HOMOLOGY = "missing_homology_cluster"
UNCERTAINTY_MULTIPLE = "multiple_plausible_candidates"
UNCERTAINTY_NO_SYNTENY = "no_syntenic_support"
UNCERTAINTY_NO_CANDIDATE = "no_candidate_in_cluster"


def _stars(n: int) -> str:
    n = max(0, min(5, int(n)))
    return "★" * n + "☆" * (5 - n)


def _stars_from_confidence(c: float) -> str:
    return _stars(int(round(max(0.0, min(1.0, float(c))) * 5)))


def _expectation_evidence() -> list[str]:
    return [
        "enriched_DH_KR_grammar",
        "kNN_capacity_probability",
        "matched_myxo_manifold",
    ]


def _candidate_evidence_tags(fam: dict | None, *, posthoc_er: str | None = None) -> list[str]:
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
    # Post-hoc display only — never used in ranking.
    if posthoc_er:
        tags.append(posthoc_er)
    return tags


def _candidate_confidence(fam: dict | None) -> float:
    """Association strength only — expectation confidence never folded in."""
    if not fam:
        return 0.0
    bits = [
        bool(fam.get("evidence_cooccurrence")),
        bool(fam.get("evidence_control_depleted")),
        bool(fam.get("evidence_myxo_biased")),
        bool(fam.get("evidence_synteny")),
    ]
    raw = 0.15 + 0.12 * sum(bits)
    assoc = fam.get("association_score")
    if assoc is not None:
        raw = max(raw, min(0.85, 0.2 + 0.45 * max(0.0, float(assoc))))
    n_mem = int(fam.get("n_members") or 1)
    if n_mem < 2:
        raw = min(raw, 0.35)
    else:
        raw = min(0.9, raw + 0.05)
    return round(raw, 3)


def _uncertainty_reasons(fam: dict | None, *, n_plausible: int) -> list[str]:
    reasons: list[str] = []
    if fam is None:
        reasons.append(UNCERTAINTY_NO_CANDIDATE)
        reasons.append(UNCERTAINTY_MISSING_HOMOLOGY)
        return reasons
    n_mem = int(fam.get("n_members") or 1)
    if n_mem < 2:
        reasons.append(UNCERTAINTY_MISSING_HOMOLOGY)
    # Domain-bag insufficiency no longer applies under homolog90 multi-member clusters.
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
        and UNCERTAINTY_MISSING_HOMOLOGY not in reasons
    ):
        status.append("candidate_supported")
    else:
        status.append("candidate_unresolved")
    return status


def _discovery_readiness(
    *,
    exp_conf: float,
    cand_conf: float,
    reasons: list[str],
    status: list[str],
) -> dict:
    if "candidate_supported" in status:
        score = min(1.0, 0.55 + 0.3 * cand_conf + 0.15 * exp_conf)
        return {
            "confidence": round(score, 3),
            "stars": _stars_from_confidence(score),
            "blocker": None,
            "next_experiment": (
                "Delete / deplete a member of the candidate family and assay "
                "the predicted metabolite shift."
            ),
        }
    if UNCERTAINTY_NO_CANDIDATE in reasons:
        score = min(0.4, 0.2 + 0.25 * exp_conf)
        blocker = "no_candidate_in_cluster"
        next_exp = "No in-cluster homolog candidate scored; inspect protein inventory."
    elif UNCERTAINTY_MISSING_HOMOLOGY in reasons:
        score = min(0.45, 0.25 + 0.2 * exp_conf + 0.15 * cand_conf)
        blocker = "limited_homology"
        next_exp = "Candidate is a singleton cluster; seek broader homolog support."
    elif UNCERTAINTY_MULTIPLE in reasons:
        score = min(0.55, 0.3 + 0.2 * exp_conf + 0.25 * cand_conf)
        blocker = "multiple_plausible_candidates"
        next_exp = (
            "Multiple homolog families compete; add synteny/structure to disambiguate, "
            "then design knockout of the leading family."
        )
    else:
        score = min(0.55, 0.3 + 0.2 * exp_conf + 0.25 * cand_conf)
        blocker = "no_syntenic_support" if UNCERTAINTY_NO_SYNTENY in reasons else "provisional_attribution"
        next_exp = (
            "Homolog family scored; strengthen synteny/context evidence before wet-lab design."
        )
    return {
        "confidence": round(score, 3),
        "stars": _stars_from_confidence(score),
        "blocker": blocker,
        "next_experiment": next_exp,
    }


def _interpretation_summary(
    *,
    chemistry: str,
    exp_conf: float,
    fam: dict | None,
    reasons: list[str],
    posthoc_er: str | None,
) -> str:
    exp_clause = (
        f"Architecture strongly predicts {chemistry} "
        f"(expectation confidence {exp_conf:.2f})"
    )
    if fam is None:
        return (
            f"{exp_clause}, but no homolog90 candidate family scored in-cluster. "
            "Capability gap is supported; protein attribution is unresolved."
        )
    fk = fam.get("family_key")
    extra = ""
    if posthoc_er:
        extra = " Cluster contains an ER-positive homolog elsewhere (post-hoc annotation)."
    if UNCERTAINTY_MULTIPLE in reasons:
        return (
            f"{exp_clause}; leading homolog family `{fk}` carries association evidence "
            f"but multiple plausible candidates remain.{extra}"
        )
    if UNCERTAINTY_MISSING_HOMOLOGY in reasons:
        return (
            f"{exp_clause}; `{fk}` is a singleton homolog cluster — limited family support.{extra}"
        )
    return (
        f"{exp_clause}; candidate homolog family `{fk}` carries gap-association evidence "
        f"(representation={REPRESENTATION}).{extra}"
    )


def _n_plausible_in_bgc(
    bgc_id: str,
    families: pd.DataFrame,
    domains: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    capacity: str,
    top_score_frac: float = 0.85,
) -> int:
    if families.empty or clusters.empty:
        return 0
    present = cds_families_homolog(domains, clusters, capacity=capacity)
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
    clusters: pd.DataFrame | None = None,
    er_positive_families: set[str] | None = None,
    capacity: str | None = None,
    top_n: int = 6,
) -> list[dict]:
    """Assemble dual-confidence hypothesis claims (no single overall score)."""
    if gaps.empty:
        return []
    cap = capacity or PILOT_GRAMMAR["expected_capacity"]
    er_pos = er_positive_families or set()
    cards = []
    for i, g in gaps.head(top_n).reset_index(drop=True).iterrows():
        hid = f"MYXO-{i+1:03d}"
        fam = None
        n_plausible = 0
        if domains is not None and clusters is not None and len(families):
            fam = best_family_for_bgc(
                str(g["bgc_id"]), families, domains, clusters=clusters, capacity=cap
            )
            n_plausible = _n_plausible_in_bgc(
                str(g["bgc_id"]), families, domains, clusters, capacity=cap
            )

        exp_conf = float(g["expectation_confidence"])
        gap_p = float(g["capability_gap"])
        chemistry = str(g.get("chemistry") or PILOT_GRAMMAR["chemistry"])
        cand_conf = _candidate_confidence(fam)
        reasons = _uncertainty_reasons(fam, n_plausible=n_plausible)
        status = _hypothesis_status(exp_conf, cand_conf, reasons)
        family_key = fam["family_key"] if fam else None
        posthoc = None
        if family_key and family_key in er_pos:
            posthoc = "contains_ER_positive_homolog_elsewhere"
        evidence_tags = _candidate_evidence_tags(fam, posthoc_er=posthoc)
        readiness = _discovery_readiness(
            exp_conf=exp_conf,
            cand_conf=cand_conf,
            reasons=reasons,
            status=status,
        )

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
                "representation": REPRESENTATION,
                "n_members": int(fam["n_members"]) if fam and fam.get("n_members") else None,
                "confidence": cand_conf,
                "stars": _stars_from_confidence(cand_conf),
                "evidence": evidence_tags,
                "limitation": (
                    ["singleton_cluster"]
                    if fam and int(fam.get("n_members") or 1) < 2
                    else (["provisional_homolog_attribution"] if fam else ["no_family_scored"])
                ),
                "uncertainty_reasons": reasons,
                "posthoc_annotation": posthoc,
            },
            "discovery_readiness": readiness,
            "hypothesis_status": status,
            "interpretation": {
                "summary": _interpretation_summary(
                    chemistry=chemistry,
                    exp_conf=exp_conf,
                    fam=fam,
                    reasons=reasons,
                    posthoc_er=posthoc,
                ),
            },
            "validation": PILOT_GRAMMAR["validation_hint"],
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
        "**Architecture can predict missing chemistry — even when we cannot yet "
        "identify the responsible enzyme.**",
        "",
        "Level 1 (architecture → expected chemistry) is frozen and supported. "
        "Level 2 upgrades candidate identity to **homolog90** clusters (MMseqs2, "
        "90% identity). Under ER-masking calibration, the architecture-derived "
        "expectation model recovered hidden ER-associated biology in ~64% of cases. "
        "Candidate attribution is evaluated separately (Recall@k / MRR / enrichment).",
        "",
        "Each card splits **expectation** vs **candidate** uncertainty. Discovery "
        "readiness is the wet-lab interface. Cards rank by expectation / gap strength.",
        "",
        "## Calibration gate (frozen Level-1 baseline)",
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
        ready = c["discovery_readiness"]
        status = ", ".join(c["hypothesis_status"])
        blocker = ready.get("blocker") or "_(none)_"
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
            f"| **Discovery readiness** (wet-lab interface) | {ready['stars']} ({ready['confidence']:.2f}) |",
            "",
            f"**Blocker.** `{blocker}`",
            "",
            f"**Next experiment.** {ready['next_experiment']}",
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
            f"| Cluster size (n_members) | {cand.get('n_members')} |",
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
        if cand["uncertainty_reasons"]:
            for r in cand["uncertainty_reasons"]:
                lines.append(f"- `{r}`")
        else:
            lines.append("- _(none)_")
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
