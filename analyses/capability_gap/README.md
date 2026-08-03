# Capability-gap inference (pilot)

The atlases are **priors**. This layer is an **inference engine**.

```text
architecture neighborhood
        ↓
P(capacity | neighborhood)
        ↓
capability gap (expectation unmet)
        ↓
candidate protein families
        ↓
hypothesis cards (dual confidence)
```

Not novelty. Not “missing annotation.”  
**Architecture–capability mismatch** conditioned on local grammar.

## Dual confidence

Each card splits two independent uncertainties — never a single overall score:

| Axis | Question |
|------|----------|
| **Expectation** | Does this pathway need this chemistry? |
| **Candidate** | Do we know which protein performs it? |

`hypothesis_status` tracks maturity (`expectation_supported` + `candidate_unresolved` → later `candidate_supported`).  
`candidate.uncertainty_reasons` are first-class codes (`insufficient_family_resolution`, …) — actionable upgrade targets.  
`discovery_readiness` is the wet-lab interface (blocker + next experiment). Cards rank by expectation/gap strength, not candidate confidence.

v0 family keys are `domain_bag__…` labels, not catalytic families. Low candidate confidence means attribution is limited, not that the gap failed.

## Pilot

Reductive PKS grammar (DH* + KR) → expected **enoyl reduction**.

## Run

```bash
uv run --no-sync bgc-capability-gap
```

## Outputs

- `reports/capability_gap_hypotheses.md` — hypothesis cards
- `reports/capability_gap/calibration.json` — mask-known-capacity recovery gate
- `reports/capability_gap/expectation_gaps.csv`
- `reports/capability_gap/candidate_families.csv`
- `reports/capability_gap/hypothesis_cards.json` — structured dual-confidence schema

## Design rules

1. Expectations are **conditional** on atlas neighborhood, not global domain frequencies.
2. Candidates are **families**, not ORFs.
3. Calibration runs **before** novel cards: hide annotated capacity; ask whether gaps appear.
4. Expectation confidence and candidate confidence stay independent.
5. Catalyst / Metabo / ABX evidence slots exist on cards but stay empty in v0.
