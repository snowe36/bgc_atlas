# Capability-gap inference (pilot)

**Architecture can predict missing chemistry — even when we cannot yet identify the responsible enzyme.**

Atlases are **priors**. This layer is an **inference engine**. Level 1 (architecture → expected chemistry) is not a failed Level 2 (chemistry → protein family) — they are different inference problems.

```text
capability_gap/
├── expectation.py      # FROZEN
├── gaps.py             # FROZEN
├── calibrate.py        # FROZEN
│
├── homologs.py         # Level 2 — MMseqs2 homolog90
├── candidates.py       # Level 2 — ranking features (no ER labels)
├── evaluation.py       # Level 2 — ER-positive labels + Recall/MRR/enrichment
├── cards.py
└── run.py
```

```text
architecture neighborhood
        ↓
P(capacity | neighborhood)
        ↓
capability gap (expectation unmet)     ← Level 1  ✓ FROZEN
        ↓
homolog90 candidate families           ← Level 2  (this upgrade)
        ↓
hypothesis cards (dual confidence)
```

## Main result

| Level | Inference | Status |
|-------|-----------|--------|
| 1 | Architecture → expected chemistry | ✓ frozen (~64% ER-masking recovery) |
| 2 | Expected chemistry → protein family | homolog90 in progress |
| 3–4 | Mechanism → metabolite / phenotype | future |

**Freeze rule.** Neighborhoods, capability probabilities, gap scoring, and calibration stay fixed. Only candidate representation upgrades. Success = **uncertainty movement** on cards (off blanket `family_resolution`), not all cards becoming `candidate_supported`.

## Dual confidence

| Axis | Question |
|------|----------|
| **Expectation** | Does this pathway need this chemistry? |
| **Candidate** | Do we know which protein family performs it? |
| **Discovery readiness** | Is this ready for a wet-lab design? |

Ranking features: co-occurrence, control depletion, myxo bias, synteny.  
**Forbidden in ranking:** `cluster_has_ER`. Evaluation may use ER-positive labels for Recall@k / MRR / enrichment@k only.

## Pilot

Reductive PKS grammar (DH* + KR) → expected **enoyl reduction**.

Requires **MMseqs2** on PATH (`brew install mmseqs2`).

## Run

```bash
uv run --no-sync bgc-capability-gap
# rebuild clusters:
uv run --no-sync bgc-capability-gap --rebuild-clusters
```

## Outputs

- `reports/capability_gap_hypotheses.md` — hypothesis cards
- `reports/capability_gap/calibration.json` — frozen Level-1 baseline
- `reports/capability_gap/candidate_level2_metrics.json` — Recall@k / MRR / enrichment
- `reports/capability_gap/candidate_families.csv`
- `data/processed/protein_homolog_clusters.csv` — homolog90 map (cached)

## Design rules

1. Expectations are **conditional** on atlas neighborhood.
2. Candidates are **homolog families**, not ORFs or domain bags.
3. Calibration runs **before** cards; freeze assert vs baseline.
4. Expectation and candidate confidence stay independent.
5. Expectation modules must not import `homologs`.
