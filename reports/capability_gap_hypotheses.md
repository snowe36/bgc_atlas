# Capability-gap hypotheses (pilot: reductive PKS)

Inference layer over the atlas: **conditional capability expectations**, not global novelty. Each card is a **claim with two independent uncertainties** — expectation (does this pathway need this chemistry?) vs candidate (do we know which protein performs it?). Discovery readiness is the wet-lab interface. There is no single overall confidence score. Cards are ranked by **expectation / gap strength**, not candidate confidence.

## Calibration gate (frozen v0 baseline)

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
| **Candidate** (protein attribution) | ☆☆☆☆☆ (0.00) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.40) |

**Blocker.** `family_resolution`

**Next experiment.** Resolve catalytic family identity in-cluster (Pfam/UniRef) before designing a knockout.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.80), but no non-core candidate family scored in-cluster. Capability gap is supported; protein attribution is unresolved.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | _(none)_ |
| Candidate representation | domain_bag |
| Priority | 1 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- _(none)_

Candidate uncertainty reasons:

- `no_candidate_in_cluster`
- `insufficient_family_resolution`
- `missing_homology_cluster`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---

## Hypothesis MYXO-002

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0000184` · Sorangium cellulosum · PKS

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★★☆ (0.78) |
| **Candidate** (protein attribution) | ☆☆☆☆☆ (0.00) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.39) |

**Blocker.** `family_resolution`

**Next experiment.** Resolve catalytic family identity in-cluster (Pfam/UniRef) before designing a knockout.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.78), but no non-core candidate family scored in-cluster. Capability gap is supported; protein attribution is unresolved.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | _(none)_ |
| Candidate representation | domain_bag |
| Priority | 2 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- _(none)_

Candidate uncertainty reasons:

- `no_candidate_in_cluster`
- `insufficient_family_resolution`
- `missing_homology_cluster`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---

## Hypothesis MYXO-003

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0000044` · Stigmatella aurantiaca DW4/3-1 · PKS

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★★☆ (0.72) |
| **Candidate** (protein attribution) | ☆☆☆☆☆ (0.00) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.38) |

**Blocker.** `family_resolution`

**Next experiment.** Resolve catalytic family identity in-cluster (Pfam/UniRef) before designing a knockout.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.72), but no non-core candidate family scored in-cluster. Capability gap is supported; protein attribution is unresolved.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | _(none)_ |
| Candidate representation | domain_bag |
| Priority | 3 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- _(none)_

Candidate uncertainty reasons:

- `no_candidate_in_cluster`
- `insufficient_family_resolution`
- `missing_homology_cluster`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---

## Hypothesis MYXO-004

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0001069` · Sorangium cellulosum · hybrid

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★★☆ (0.70) |
| **Candidate** (protein attribution) | ★☆☆☆☆ (0.21) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.44) |

**Blocker.** `family_resolution`

**Next experiment.** Upgrade family representation (Pfam/UniRef); then design knockout of the leading family member and assay metabolite shift.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.70), but v0 family resolution is insufficient to identify the responsible catalytic family. Domain-bag keys are functional labels, not Pfam/UniRef catalytic families.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | `domain_bag__Redox` |
| Candidate representation | domain_bag |
| Priority | 4 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- `local_synteny`
- `gap_cooccurrence`

Candidate uncertainty reasons:

- `insufficient_family_resolution`
- `missing_homology_cluster`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---

## Hypothesis MYXO-005

**Status.** `expectation_supported, candidate_unresolved`

**BGC** `BGC0000986` · Stigmatella aurantiaca DW4/3-1 · hybrid

| | Confidence |
|---|---|
| **Expectation** (architecture → chemistry) | ★★★☆☆ (0.65) |
| **Candidate** (protein attribution) | ★☆☆☆☆ (0.21) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.43) |

**Blocker.** `family_resolution`

**Next experiment.** Upgrade family representation (Pfam/UniRef); then design knockout of the leading family member and assay metabolite shift.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.65), but v0 family resolution is insufficient to identify the responsible catalytic family. Domain-bag keys are functional labels, not Pfam/UniRef catalytic families.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | `domain_bag__Hydrolase` |
| Candidate representation | domain_bag |
| Priority | 5 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- `local_synteny`
- `gap_cooccurrence`

Candidate uncertainty reasons:

- `insufficient_family_resolution`
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
| **Candidate** (protein attribution) | ★☆☆☆☆ (0.21) |
| **Discovery readiness** (wet-lab interface) | ★★☆☆☆ (0.43) |

**Blocker.** `family_resolution`

**Next experiment.** Upgrade family representation (Pfam/UniRef); then design knockout of the leading family member and assay metabolite shift.

**Interpretation.** Architecture strongly predicts enoyl reduction (ER) (expectation confidence 0.62), but v0 family resolution is insufficient to identify the responsible catalytic family. Domain-bag keys are functional labels, not Pfam/UniRef catalytic families.

| Field | Value |
|-------|-------|
| Architecture neighborhood | reductive_pks |
| Expected capability | enoyl_reduction |
| Chemistry | enoyl reduction (ER) |
| Gap type | missing_catalytic_capability |
| Observed inventory | No annotated enoyl reduction |
| Candidate family | `domain_bag__Hydrolase` |
| Candidate representation | domain_bag |
| Priority | 6 |

Expectation evidence:

- `enriched_DH_KR_grammar`
- `kNN_capacity_probability`
- `matched_myxo_manifold`

Candidate evidence:

- `local_synteny`
- `gap_cooccurrence`

Candidate uncertainty reasons:

- `insufficient_family_resolution`
- `missing_homology_cluster`
- `multiple_plausible_candidates`

**Validation.** Knockout / depletion → accumulation of β-unsaturated intermediate (+2 Da gap vs reduced product).

---
