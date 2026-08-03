"""Pilot configuration: reductive PKS capability gaps in myxo neighborhoods."""

from __future__ import annotations

# Capabilities expressed as domain sets (same vocabulary as featurize / myxo_grammar).
CAPABILITIES: dict[str, set[str]] = {
    "enoyl_reduction": {"PKS_ER", "ER"},
    "ketoreduction": {"PKS_KR", "KR"},
    "dehydration": {"PKS_DH", "PKS_DH2", "DH"},
    "methylation": {"Methyltransferase", "MT"},
    "glycosylation": {"Glycosyltransferase"},
    "halogenation": {"Halogenase"},
    "p450_oxidation": {"P450", "PF00067"},
}

# Pilot: fully reduced PKS grammar expects enoyl reduction when DH+KR present.
PILOT_GRAMMAR = {
    "name": "reductive_pks",
    "required_present": [{"PKS_DH", "PKS_DH2", "DH"}, {"PKS_KR", "KR"}],
    "expected_capacity": "enoyl_reduction",
    "chemistry": "enoyl reduction (ER)",
    "validation_hint": "Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).",
}

# Core assembly domains — not useful as "compensating family" candidates.
CORE_DOMAINS = {
    "PKS_KS",
    "KS",
    "PKS_AT",
    "AT",
    "PKS_KR",
    "KR",
    "PKS_DH",
    "PKS_DH2",
    "PKS_DHt",
    "DH",
    "PKS_ER",
    "ER",
    "ACP",
    "PKS_PP",
    "PCP",
    "PP",
    "PKS_module",
    "NRPS_module",
    "AMP-binding",
    "Condensation",
    "Thioesterase",
    "thioesterase",
    "polyketide_synthase",
    "PKS_Docking",
    "nonribosomal_peptide",
    "CDS_other",
    "Hypothetical",
    "Regulator",
    "Transporter",
}

NEIGHBORHOOD_K = 40
SIZE_MAX = 60
MIN_EXPECTATION = 0.55  # P(capacity|N) to call a gap when absent
TOP_HYPOTHESES = 6

# Families that are biochemically implausible as compensatory *enoyl reduction*
# (other gap types can relax this later).
IMPLAUSIBLE_ER_FAMILIES = {
    "MT",
    "Methyltransferase",
    "Glycosyltransferase",
    "Halogenase",
    "P450",
    "Transporter",
    "Thioesterase",
    "thioesterase",
}

# Preferred compensatory evidence for ER gaps (domain vocabulary).
REDOX_HINT_DOMAINS = {
    "Redox",
    "Hydrolase",
    "PF00106",  # short-chain dehydrogenase/reductase
    "PF00109",
    "PF08659",
    "PF14765",
}
