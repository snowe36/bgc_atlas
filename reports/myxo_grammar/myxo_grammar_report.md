# Myxobacterial architecture grammar

**Question.** Given that myxobacterial BGCs occupy unusual hashed-architecture neighborhoods after size matching, which architectural primitives account for that separation?

**Not the question.** Which BGCs are novel / will be deposited next.

---

## Phenotype gate (must pass before token analysis)

We first recover the published size-controlled phenotype, then ask what explains it.

- Size filter: n_genes ≤ 60
- Contrast: Myxobacteria vs Streptomyces
- n_myxo = 84, n_control = 826
- Top-decile OR = **2.95** (95% CI 1.78–4.88)
- Mann–Whitney p (myxo > control) = 0.000751
- Gate OK: **True**

## Matched cohort

Matching is designed to isolate **architectural composition** from obvious confounders (BGC size and coarse biosynthetic class), not to estimate a causal treatment effect.

- Gene-count tolerance: ±10%
- Controls per myxo BGC: 5
- Matched myxo BGCs: 84 / 84
- Unique controls: 183
- Same-class pair fraction: 0.9847094801223242

## Result: assembly grammar, not rarer parts

Tokens come from existing MIBiG domain annotations (abundance, **ordered motifs**, capacity combos). No new embedding.

### Rarity negative control (load-bearing)

Mean token-IDF rarity does **not** differ: myxo median=1.5077292564727618, control median=1.5303500251655309, p(myxo > control)=0.3713263455816208.

This rules out the simplistic explanation that myxobacteria look unusual because they use rarer building blocks. Instead, they preferentially arrange **largely familiar components into different architectural grammars** — pathway organization rules, not just protein inventories.

### Ordered motifs and combo grammars

The strongest signals are **ordered biosynthetic grammars** (e.g. reductive PKS cascades, PKS↔NRPS bridges), not isolated domain counts:

| Token | Level | Myxo frac | Control frac | OR | q |
|-------|-------|----------:|-------------:|---:|--:|
| `motif2::PKS_DH2-PKS_KR` | motif | 0.29 | 0.02 | 17.90 | 4.31e-08 |
| `motif3::PKS_DH2-PKS_KR-PKS_KS` | motif | 0.29 | 0.02 | 17.90 | 4.31e-08 |
| `motif2::PKS_AT-PKS_DH2` | motif | 0.18 | 0.00 | 81.85 | 7.19e-07 |
| `motif3::PKS_PP-NRPS_module-nonribosomal_peptide` | motif | 0.20 | 0.01 | 46.18 | 7.19e-07 |
| `motif3::PKS_AT-PKS_DH-PKS_ER` | motif | 0.35 | 0.08 | 6.36 | 5.19e-06 |
| `motif2::PKS_DH2-PKS_ER` | motif | 0.15 | 0.00 | 69.29 | 6.2e-06 |
| `motif2::PKS_DH-PKS_ER` | motif | 0.36 | 0.09 | 5.80 | 6.84e-06 |
| `motif3::PKS_DH-PKS_ER-PKS_KR` | motif | 0.36 | 0.09 | 5.80 | 6.84e-06 |
| `motif3::PKS_KS-PKS_PP-NRPS_module` | motif | 0.26 | 0.04 | 8.92 | 7.23e-06 |
| `motif3::MT-PCP-NRPS_module` | motif | 0.20 | 0.02 | 15.22 | 1.17e-05 |
| `grammar::PKS+MT` | grammar | 0.60 | 0.24 | 4.65 | 1.38e-06 |
| `grammar::NRPS+MT` | grammar | 0.56 | 0.26 | 3.68 | 3.7e-05 |
| `capacity::PKS_DH` | capacity | 0.71 | 0.41 | 3.60 | 5.27e-05 |
| `capacity::methyltransferase` | capacity | 0.68 | 0.38 | 3.49 | 6.71e-05 |
| `capacity::NRPS_thiolation` | capacity | 0.65 | 0.40 | 2.86 | 0.000867 |

## Full enrichment table (top OR≥2, q≤0.05)

| Token | Level | Myxo frac | Control frac | OR | q |
|-------|-------|----------:|-------------:|---:|--:|
| `motif2::PKS_DH2-PKS_KR` | motif | 0.29 | 0.02 | 17.90 | 4.31e-08 |
| `motif3::PKS_DH2-PKS_KR-PKS_KS` | motif | 0.29 | 0.02 | 17.90 | 4.31e-08 |
| `dom::PKS_DH2` | domain | 0.36 | 0.05 | 10.74 | 4.31e-08 |
| `dom::MT` | domain | 0.62 | 0.21 | 6.00 | 4.31e-08 |
| `motif2::PKS_AT-PKS_DH2` | motif | 0.18 | 0.00 | 81.85 | 7.19e-07 |
| `motif3::PKS_PP-NRPS_module-nonribosomal_peptide` | motif | 0.20 | 0.01 | 46.18 | 7.19e-07 |
| `grammar::PKS+MT` | grammar | 0.60 | 0.24 | 4.65 | 1.38e-06 |
| `motif3::PKS_AT-PKS_DH-PKS_ER` | motif | 0.35 | 0.08 | 6.36 | 5.19e-06 |
| `motif2::PKS_DH2-PKS_ER` | motif | 0.15 | 0.00 | 69.29 | 6.2e-06 |
| `motif2::PKS_DH-PKS_ER` | motif | 0.36 | 0.09 | 5.80 | 6.84e-06 |
| `motif3::PKS_DH-PKS_ER-PKS_KR` | motif | 0.36 | 0.09 | 5.80 | 6.84e-06 |
| `motif3::PKS_KS-PKS_PP-NRPS_module` | motif | 0.26 | 0.04 | 8.92 | 7.23e-06 |
| `motif3::MT-PCP-NRPS_module` | motif | 0.20 | 0.02 | 15.22 | 1.17e-05 |
| `motif2::MT-PKS_AT` | motif | 0.17 | 0.01 | 36.40 | 1.24e-05 |
| `motif3::PKS_DH2-PKS_ER-PKS_KR` | motif | 0.14 | 0.00 | 63.28 | 1.32e-05 |

## Temporal sneak (exploratory only)

- Cutoff: 2022-09-16
- Global held-out mean novelty: 0.3973440464233903 (random control 0.4948408293561743)
- Held-out myxo n=6 mean=0.6390237324703344
- Held-out Streptomyces mean=0.4909532215357457
- Myxo > non-myxo among held-out? p=0.0326806394850657

An exploratory analysis of the small held-out myxobacterial subset suggested higher architectural novelty than non-myxobacterial clusters, but the sample size is too limited for strong inference.

## Figure

![Figure 5](figures/fig_myxo_architecture_grammar.png)

## Claim framing

`bgc_atlas` does not predict discovery. It quantifies architectural organization, tests whether novelty is meaningful, and identifies lineage-associated biosynthetic design grammars.
