# Capability-gap hypotheses (pilot: reductive PKS)

**Architecture can predict missing chemistry — even when we cannot yet identify the responsible enzyme.**

Level 1 (architecture → expected chemistry) is frozen and supported. Level 2 upgrades candidate identity to **homolog90** clusters (MMseqs2, 90% identity). Under ER-masking calibration, the architecture-derived expectation model recovered hidden ER-associated biology in ~64% of cases. Candidate attribution is evaluated separately (Recall@k / MRR / enrichment).

Each card splits **expectation** vs **candidate** uncertainty. Discovery readiness is the wet-lab interface. Cards rank by expectation / gap strength.

## Calibration gate (frozen Level-1 baseline)

- Counterfactual detection (annotated positives): 0.6444444444444445
- Detection on true gaps: 0.411214953271028
- P(annotated) vs non-pilot win-rate: 0.9744439058544728
- P(true gaps) vs non-pilot win-rate: 0.906068650590511
- Calibration OK: **True**

Counterfactual calibration: estimate P(capacity|neighborhood) on the observed atlas, then ask whether annotated-positive pilot BGCs would clear the gap threshold if their capacity annotation were absent. Masking all positives before fitting P(.) is invalid — it erases the neighborhood prior.

---

## Hypothesis MYXO-001

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0001051` · Sorangium cellulosum · hybrid

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★★☆ (0.80) |
| **Candidate** (protein attribution) | ★★☆☆☆ (0.35) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.45) |

**Blocker.** `limited_homology`

**Next experiment.** Candidate is a singleton cluster; seek broader homolog support.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.80); leading homolog family `homolog90__clu_15054` carries association evidence but multiple plausible candidates remain.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | `homolog90__clu_15054` |
| Candidate representation | homolog90 |
| Cluster size (n_members) | 1 |
| Priority | 1 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- `local_synteny`
- `control_depleted`
- `myxo_cooccurrence`

Candidate uncertainty reasons:

- `missing_homology_cluster`
- `multiple_plausible_candidates`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---

## Hypothesis MYXO-002

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0000184` · Sorangium cellulosum · PKS

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★★☆ (0.78) |
| **Candidate** (protein attribution) | ★★☆☆☆ (0.35) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.45) |

**Blocker.** `limited_homology`

**Next experiment.** Candidate is a singleton cluster; seek broader homolog support.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.78); leading homolog family `homolog90__clu_03388` carries association evidence but multiple plausible candidates remain.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | `homolog90__clu_03388` |
| Candidate representation | homolog90 |
| Cluster size (n_members) | 1 |
| Priority | 2 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- `local_synteny`
- `control_depleted`
- `myxo_cooccurrence`

Candidate uncertainty reasons:

- `missing_homology_cluster`
- `multiple_plausible_candidates`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---

## Hypothesis MYXO-003

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0000044` · Stigmatella aurantiaca DW4/3-1 · PKS

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★★☆ (0.72) |
| **Candidate** (protein attribution) | ★★☆☆☆ (0.35) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.45) |

**Blocker.** `limited_homology`

**Next experiment.** Candidate is a singleton cluster; seek broader homolog support.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.72); leading homolog family `homolog90__clu_00770` carries association evidence but multiple plausible candidates remain.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | `homolog90__clu_00770` |
| Candidate representation | homolog90 |
| Cluster size (n_members) | 1 |
| Priority | 3 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- `local_synteny`
- `control_depleted`
- `myxo_cooccurrence`

Candidate uncertainty reasons:

- `missing_homology_cluster`
- `multiple_plausible_candidates`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---

## Hypothesis MYXO-004

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0001069` · Sorangium cellulosum · hybrid

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★★☆ (0.70) |
| **Candidate** (protein attribution) | ★★☆☆☆ (0.35) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.44) |

**Blocker.** `limited_homology`

**Next experiment.** Candidate is a singleton cluster; seek broader homolog support.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.70); leading homolog family `homolog90__clu_15502` carries association evidence but multiple plausible candidates remain.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | `homolog90__clu_15502` |
| Candidate representation | homolog90 |
| Cluster size (n_members) | 1 |
| Priority | 4 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- `local_synteny`
- `control_depleted`
- `myxo_cooccurrence`

Candidate uncertainty reasons:

- `missing_homology_cluster`
- `multiple_plausible_candidates`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---

## Hypothesis MYXO-005

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0000986` · Stigmatella aurantiaca DW4/3-1 · hybrid

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★☆☆ (0.65) |
| **Candidate** (protein attribution) | ★★☆☆☆ (0.35) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.43) |

**Blocker.** `limited_homology`

**Next experiment.** Candidate is a singleton cluster; seek broader homolog support.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.65); leading homolog family `homolog90__clu_14034` carries association evidence but multiple plausible candidates remain.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | `homolog90__clu_14034` |
| Candidate representation | homolog90 |
| Cluster size (n_members) | 1 |
| Priority | 5 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- `local_synteny`
- `control_depleted`
- `myxo_cooccurrence`

Candidate uncertainty reasons:

- `missing_homology_cluster`
- `multiple_plausible_candidates`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---

## Hypothesis MYXO-006

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0001751` · Pyxidicoccus sp. MCy9557 · hybrid

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★☆☆ (0.62) |
| **Candidate** (protein attribution) | ★★☆☆☆ (0.35) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.43) |

**Blocker.** `limited_homology`

**Next experiment.** Candidate is a singleton cluster; seek broader homolog support.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.62); leading homolog family `homolog90__clu_23603` carries association evidence but multiple plausible candidates remain.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | `homolog90__clu_23603` |
| Candidate representation | homolog90 |
| Cluster size (n_members) | 1 |
| Priority | 6 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- `local_synteny`
- `control_depleted`
- `myxo_cooccurrence`

Candidate uncertainty reasons:

- `missing_homology_cluster`
- `multiple_plausible_candidates`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---
