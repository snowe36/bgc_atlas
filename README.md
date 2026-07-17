# bgc_atlas

**Biosynthetic architecture atlas** — a CPU-only, reproducible pipeline that maps microbial biosynthetic gene cluster (BGC) space and ranks architecturally novel regions, with a prospective (time-split) test of whether that novelty score actually predicts what gets discovered next.

[![CI](https://github.com/snowe36/bgc_atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/snowe36/bgc_atlas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Repo: [github.com/snowe36/bgc_atlas](https://github.com/snowe36/bgc_atlas)

---

## The question

Microbial genomes encode far more biosynthetic gene clusters than have been experimentally characterized ([MIBiG](https://mibig.secondarymetabolites.org/); [antiSMASH](https://docs.antismash.secondarymetabolites.org/)). Rule-based tools already answer "is this a BGC, and what class?" well. The harder, more useful question is:

**Can we identify BGCs that sit far from known biosynthetic architecture — and does that novelty score actually mean anything?**

This repo builds a reference atlas of MIBiG architecture space, ranks clusters by distance from characterized neighborhoods, and then stress-tests that ranking with label-leakage checks, size-confound checks, and a prospective (time-split) holdout — reporting the result honestly even where it doesn't confirm the hypothesis.

![Biosynthetic space by class](reports/figures/atlas_by_class.png)

**Talk track:** *I represented BGC architecture, showed the representation recovers known families, ranked clusters by distance from characterized neighborhoods, and validated the strategy with explicit leakage/size-confound checks plus a prospective temporal holdout — reporting the negative result on the holdout rather than hiding it.*

---

## Key results at a glance

| Check | Result |
|-------|--------|
| Representation recovers known biosynthetic classes | Random forest macro-F1 **0.76**, weighted-F1 **0.79** (5-fold stratified CV) |
| Class labels leak into novelty features | **No** (`bgc-validate` audit) |
| Novelty score dominated by cluster size | **No** — Spearman(novelty, gene count) = **0.12** |
| Top-decile novelty hits share their class with their nearest neighbor | **67%** of the time (i.e. novelty ≠ misclassification) |
| Prospective test: do chronologically newer MIBiG entries score as architecture-novel? | **Not supported** — see [Prospective validation](#prospective-temporal-holdout-validation) below; reported as a negative result, not smoothed over |
| Test suite | **4/4 passing**, run on every push via CI (badge above) |

---

## Why this matters

Public resources such as **MIBiG** (experimentally characterized BGCs) and **antiSMASH** (genome-wide BGC prediction) make it clear that sequenced genomes contain many more clusters than have been linked to molecules ([MIBiG 3.0/4.0](https://mibig.secondarymetabolites.org/); [Blin et al., *NAR*](https://doi.org/10.1093/nar/gkad984)). Deciding *where to look next* for new chemistry is a triage problem, not a classification problem — and triage tools need to be validated, not just built.

That's the standard this project holds itself to: the same rigor a wet-lab experiment demands — negative controls, holdouts, and reporting results whether or not they support the hypothesis — applied to a computational discovery pipeline. Concretely: no result here is reported without checking whether it's an artifact of class labels, cluster size, or evaluation leakage, and the prospective holdout in this repo is a genuine hold-out-in-time test that gets reported even when it comes back negative.

This project intentionally starts with **interpretable, CPU-friendly representations** rather than large pretrained models, to understand biosynthetic space structure and validate discovery heuristics before adding more complex embeddings (see [GPU / protein language model embeddings](#gpu--protein-language-model-embeddings-in-progress) below).

---

## Reproducibility

- **Environment is pinned**: Python 3.11+ (see `.python-version`), dependencies pinned by lower bound in [`pyproject.toml`](pyproject.toml).
- **CI runs the test suite on every push/PR** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) — see badge at the top of this README.
- **One-command reproduction**: `bash scripts/reproduce.sh` runs the full pipeline end-to-end from a clean environment (download → featurize → sanity-check → atlas → novelty → validate → apply); takes a few minutes on a laptop CPU.
- **Unit tests** cover parsing, featurization, and novelty scoring on tiny fixtures ([`tests/test_pipeline.py`](tests/test_pipeline.py)) — no network or GPU required.
- **Every reported number has a file behind it**: metrics, rankings, and audits are written to [`reports/`](reports/) as JSON/CSV, not just printed to a log.

Quick start (full walkthrough, including the macOS Python install step, is in [How to reproduce](#how-to-reproduce)):

```bash
git clone https://github.com/snowe36/bgc_atlas.git && cd bgc_atlas
python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
bash scripts/reproduce.sh && pytest -q
```

---

## Data

| Item | Detail |
|------|--------|
| Source | [MIBiG 4.0](https://mibig.secondarymetabolites.org/) JSON + GenBank ([download mirror](https://dl.secondarymetabolites.org/mibig/)) |
| Scope | Experimentally characterized reference BGCs |
| Parsed | **3,013** JSON entries · **2,636** GenBank records |
| Featurized | **2,762** BGCs with gene annotations |
| Classes | PKS 717 · NRPS 556 · other 482 · hybrid 413 · RiPP 413 · terpene 181 |
| Demo set | Curated predicted BGCs in [`data/external/`](data/external/) (workflow illustration only) |

Pipeline: `bgc-download` → tidy parquet/CSV under `data/processed/`.

---

## Representation

CPU-friendly **pathway architecture features** (no protein language models in v1):

- Domain / CDS-product token counts
- Hashed ordered architecture (domain unigrams + bigrams)
- Cluster size statistics (gene count, length, etc.)

**2,762 × 342** feature matrix. Biosynth class labels are used for coloring and sanity checks only — **never** as novelty features (verified by `bgc-validate`; see [Validation](#validation)).

---

## Representation benchmark

If the representation recovers known families, distances in that space are biologically meaningful for architecture-novelty ranking.

Models (intentionally simple, 5-fold stratified CV):

| Model | Macro-F1 | Weighted-F1 |
|-------|---------:|------------:|
| Logistic regression | 0.65 | 0.68 |
| **Random forest** | **0.76** | **0.79** |

![Confusion matrix](reports/figures/sanity_confusion_matrix.png)

![Per-class F1](reports/figures/sanity_per_class_f1.png)

NRPS/PKS separate cleanly; hybrid and "other" are harder (as expected). Full metrics: [`reports/sanity_metrics.json`](reports/sanity_metrics.json).

---

## Atlas

Architecture features → standardized PCA (**50-D**, ~72% variance) for distances; **2-D PCA map** for visualization (UMAP when `umap-learn` installs cleanly; PCA fallback is the default on this stack).

PCA is used as a compact representation for distance calculations to reduce noise from sparse, high-dimensional architecture features before nearest-neighbor scoring. The 2-D visualizations below use robust (percentile-based) axis limits so that a handful of extreme size outliers — already flagged by the validation audit — don't collapse the whole plot into a corner; the count of off-frame points is annotated on the plot itself rather than hidden.

![Per-class neighborhoods](reports/figures/atlas_class_facets.png)

---

## Novelty scoring

Leave-one-out **kNN distance** + **local rarity** in PCA space → composite score ∈ [0, 1].

> **Definition.** Here, *novelty* means divergence in **biosynthetic architecture space**, not experimentally confirmed chemical novelty. Different architectures can yield related molecules, and similar architectures can produce divergent chemistry.

Hero artifact: [`reports/novelty_ranking.csv`](reports/novelty_ranking.csv)

| Rank | BGC ID | Organism | Class | Score | Nearest MIBiG |
|-----:|--------|----------|-------|------:|---------------|
| 1 | BGC0002977 | *Bacillus subtilis* fmb60 | hybrid | 1.00 | BGC0000081 |
| 2 | BGC0000103 | *Mycobacterium ulcerans* Agy99 | PKS | 1.00 | BGC0000038 |
| 3 | BGC0002124 | *Actinomadura verrucosospora* | PKS | 1.00 | BGC0002587 |
| 4 | BGC0000315 | *Streptomyces coelicolor* A3(2) | NRPS | 1.00 | BGC0000324 |
| 5 | BGC0002808 | *Streptomyces scabiei* 87.22 | PKS | 1.00 | BGC0001063 |

![Known vs high-novelty overlay](reports/figures/novelty_overlay.png)

![Novelty by class](reports/figures/novelty_by_class.png)

Hybrids and PKS sit higher on average; RiPPs are denser / more self-similar in this feature space.

---

## Validation

Integrity checks are first-class (`bgc-validate` → [`reports/validation_audit.json`](reports/validation_audit.json)):

| Check | Result |
|-------|--------|
| **Class-label leakage into features** | **none** |
| Top-decile same-class neighbor rate | **0.67** |
| Novelty ↔ gene-count correlation | **0.12** (not size-dominated) |
| Top-50 size outliers flagged | **4** (e.g. mis-scaled mega-clusters) |
| Checks passed | **yes** |

![Stratified novelty audit](reports/figures/validation_novelty_by_class.png)

Top-ranked hits should be inspected for annotation quirks (e.g. unusually large gene counts) before treating them as priorities — the audit surfaces those cases explicitly.

---

## Prospective (temporal-holdout) validation

The checks above confirm the novelty score isn't an artifact of labels or size — but they don't show it's *predictive* of anything. MIBiG's changelog carries a real submission date per entry, which makes a genuine prospective test possible: **fit the reference manifold only on BGCs added before a cutoff date, then ask whether entries added *after* that cutoff — which the model never saw — score as architecture-novel**, relative to a size-matched random-holdout control drawn from the same reference corpus (`bgc-temporal` → [`src/bgcatlas/novelty/temporal.py`](src/bgcatlas/novelty/temporal.py), [`reports/temporal_holdout.json`](reports/temporal_holdout.json)).

This is the BGC analogue of a time-split evaluation: if the score were more than a rank-ordering artifact, chronologically newer entries should skew novel relative to a random slice of the same reference corpus.

| Cutoff | Reference BGCs | Held-out (post-cutoff) BGCs | Held-out mean novelty | Random-control mean novelty (50 resamples) | Mann-Whitney p (held-out > control) |
|--------|----------------:|-----------------------------:|-----------------------:|---------------------------------------------:|--------------------------------------:|
| 2022-09-16 | 2,472 | 290 | **0.397** | **0.495** (± 0.018) | **0.997** |

![Prospective novelty: random vs. true post-cutoff holdout](reports/figures/temporal_holdout.png)

**This does not support the hypothesis.** Post-cutoff MIBiG entries scored *less* architecture-novel than a random control, not more — the opposite of what "novelty predicts what gets added next" would require. Reported here as-is rather than reframed after the fact. The most likely reading: recent MIBiG additions skew toward incremental variants of well-studied families (NRPS/PKS/hybrid mean novelty stayed high; RiPP/terpene mean novelty stayed low — consistent with the class-level pattern in [Validation](#validation) rather than a temporal effect), so "added recently" and "architecturally novel" are measuring different things in this corpus. That's a useful, falsifiable finding about what this novelty score does and doesn't capture, not a result to paper over.

---

## Apply to new genomes (demo)

**Demonstration dataset only — not a discovery claim.** Curated predicted BGC domain tables in [`data/external/`](data/external/) illustrate the ranking workflow: featurize with the MIBiG vocabulary and score against the MIBiG manifold.

[`reports/predicted_novelty_ranking.csv`](reports/predicted_novelty_ranking.csv):

| Rank | Genome (demo label) | BGC | Predicted class | Score | Nearest MIBiG |
|-----:|---------------------|-----|-----------------|------:|---------------|
| 1 | Rare_actinobacterium_predicted | PRED0006 | other | 0.64 | BGC0002148 |
| 2 | Rare_actinobacterium_predicted | PRED0007 | NRPS | 0.64 | BGC0002608 |
| 3 | Myxococcus_sp_predicted | PRED0008 | hybrid | 0.64 | BGC0002608 |

Workflow: **reference atlas → architecture-novelty score → prioritize non-MIBiG candidates**.

---

## GPU / protein language model embeddings (in progress)

The representation above is deliberately CPU-only and interpretable so the discovery strategy could be validated first. The natural next step — underway, not yet benchmarked in this README — is to see whether a protein language model changes the picture:

- [`scripts/run_esm_embed.py`](scripts/run_esm_embed.py) embeds MIBiG CDS translations with **ESM2** (`facebook/esm2_t33_650M_UR50D`) and mean-pools to per-BGC vectors; it's the one GPU-dependent step, designed to run standalone on a rented GPU pod rather than inside the CPU-only local pipeline.
- The planned `bgc-ablation` benchmark will compare hashed architecture features vs. ESM2 embeddings vs. the two combined, under the same 5-fold CV protocol as the [representation benchmark](#representation-benchmark) above, so any lift from the foundation-model embeddings is measured against the existing CPU baseline rather than asserted.

This section will be replaced with real numbers once that benchmark lands — intentionally not claimed here ahead of the evidence.

---

## Limitations

- Scores reflect **architecture** divergence, not proven new chemistry
- The prospective temporal holdout (above) did not confirm that architecture-novelty predicts which entries get added to MIBiG next — treat the novelty score as a within-corpus divergence measure, not a validated discovery-timing signal
- Product-class / bioactivity prediction are out of scope
- Raw MIBiG GenBank lacks antiSMASH domain calls; domains are inferred from CDS products
- UMAP is optional (`pip install '.[umap]'`); PCA is the reliable default here
- Predicted set is a curated demo, not full antiSMASH-DB scale

---

## Future directions

- GPU protein-embedding ablation (hashed architecture vs. ESM2 vs. combined) — see [GPU / protein language model embeddings](#gpu--protein-language-model-embeddings-in-progress)
- Investigate *why* the temporal holdout came back negative (e.g. does restricting to non-major-family entries, or a longer lead time before the cutoff, change the picture?)
- Richer domain calls from antiSMASH-annotated GenBank
- Larger curated antiSMASH-DB expansions for real genome prioritization

---

## How to reproduce

Requires **Python 3.11+**:

```bash
brew install python@3.11   # if needed
git clone https://github.com/snowe36/bgc_atlas.git
cd bgc_atlas
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/reproduce.sh
pytest -q
```

Step-through CLI:

```text
bgc-download → bgc-featurize → bgc-sanity → bgc-atlas → bgc-novelty → bgc-validate → bgc-apply → bgc-temporal
```

---

## Project layout

```text
src/bgcatlas/        package (data, featurize, models, atlas, novelty)
scripts/             thin wrappers + reproduce.sh
data/raw|processed/  MIBiG download + feature matrices (gitignored bulk)
data/external/       demo predicted BGCs (not discovery claims)
reports/             rankings, metrics, figures
tests/               parse / featurize / novelty unit tests
.github/workflows/   CI (pytest on every push/PR)
```

---

## Pipeline / git history

The commit history is the scientific story:

1. Initialize project
2. Acquire biological data
3. Represent biosynthetic pathways
4. Benchmark ML representations
5. Map biosynthetic space
6. Identify unexplored regions
7. Validate discovery strategy
8. Apply to new genomes
9. Prospective temporal-holdout validation

---

## Acknowledgements

Developed with the assistance of [Cursor](https://cursor.com), the AI code editor, which supported code authoring, refactoring, and documentation throughout this project.

---

## License

MIT
