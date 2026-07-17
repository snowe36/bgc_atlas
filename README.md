# bgc_atlas

**Biosynthetic novelty atlas** — a CPU-only, reproducible pipeline to map microbial BGC space and rank unexplored regions.

> Inspired by the observation that microbial genomes contain vast unexplored chemical diversity, this project builds a CPU-only ML pipeline to map biosynthetic novelty and identify candidate unexplored regions of BGC space.

**Talk track:** *Known natural products are a thin slice of microbial chemistry. I built a representation of biosynthetic architecture, showed it recovers known BGC families, then ranked clusters by distance from characterized neighborhoods.*

Repo: [github.com/snowe36/bgc_atlas](https://github.com/snowe36/bgc_atlas)

---

## Motivation

Microbial genomes encode biosynthetic gene clusters (BGCs) for polyketides, nonribosomal peptides, terpenes, RiPPs, and more. Many are silent or chemically uncharacterized.

The interesting discovery question is **not** “can we classify known BGCs?” (antiSMASH already does this well), but:

**Can we identify biosynthetic space that is distant from known chemistry?**

That is the centerpiece of this repo: an atlas of MIBiG architecture space plus a ranked novelty table.

---

## Data

| Item | Detail |
|------|--------|
| Source | [MIBiG 4.0](https://mibig.secondarymetabolites.org/) JSON + GenBank ([download mirror](https://dl.secondarymetabolites.org/mibig/)) |
| Scope | Experimentally characterized reference BGCs |
| Parsed | **3,013** JSON entries · **2,636** GenBank records |
| Featurized | **2,762** BGCs with gene annotations |
| Classes | PKS 717 · NRPS 556 · other 482 · hybrid 413 · RiPP 413 · terpene 181 |
| Predicted set | Curated non-MIBiG candidates in [`data/external/`](data/external/) |

Pipeline: `np-download` → tidy parquet/CSV under `data/processed/`.

---

## Representation

CPU-friendly **pathway architecture features** (no protein language models in v1):

- Domain / CDS-product token counts
- Hashed ordered architecture (domain unigrams + bigrams)
- Cluster size statistics (gene count, length, etc.)

**2,762 × 342** feature matrix. Biosynth class labels are used for coloring and sanity checks only — **never** as novelty features.

---

## Representation benchmark

If the representation recovers known families, distances in that space are biologically meaningful for novelty.

Models (intentionally simple, 5-fold stratified CV):

| Model | Macro-F1 | Weighted-F1 |
|-------|---------:|------------:|
| Logistic regression | 0.65 | 0.68 |
| **Random forest** | **0.76** | **0.79** |

![Confusion matrix](reports/figures/sanity_confusion_matrix.png)

![Per-class F1](reports/figures/sanity_per_class_f1.png)

NRPS/PKS separate cleanly; hybrid and “other” are harder (as expected). Full metrics: [`reports/sanity_metrics.json`](reports/sanity_metrics.json).

---

## Atlas

Architecture features → standardized PCA (**50-D**, ~72% variance) for distances; **2-D PCA map** for visualization (UMAP when `umap-learn` installs cleanly; PCA fallback is the default on this stack).

![Biosynthetic space by class](reports/figures/atlas_by_class.png)

![Per-class neighborhoods](reports/figures/atlas_class_facets.png)

---

## Novelty ranking (hero)

Leave-one-out **kNN distance** + **local rarity** in PCA space → composite novelty ∈ [0, 1].

Hero artifact: [`reports/novelty_ranking.csv`](reports/novelty_ranking.csv)

| Rank | BGC ID | Organism | Class | Novelty | Nearest MIBiG |
|-----:|--------|----------|-------|--------:|---------------|
| 1 | BGC0002977 | *Bacillus subtilis* fmb60 | hybrid | 1.00 | BGC0000081 |
| 2 | BGC0000103 | *Mycobacterium ulcerans* Agy99 | PKS | 1.00 | BGC0000038 |
| 3 | BGC0002124 | *Actinomadura verrucosospora* | PKS | 1.00 | BGC0002587 |
| 4 | BGC0000315 | *Streptomyces coelicolor* A3(2) | NRPS | 1.00 | BGC0000324 |
| 5 | BGC0002808 | *Streptomyces scabiei* 87.22 | PKS | 1.00 | BGC0001063 |

![Known vs high-novelty overlay](reports/figures/novelty_overlay.png)

![Novelty by class](reports/figures/novelty_by_class.png)

Hybrids and PKS sit higher in novelty on average; RiPPs are denser / more self-similar in this feature space.

---

## Validation

Integrity checks (`np-validate` → [`reports/validation_audit.json`](reports/validation_audit.json)):

| Check | Result |
|-------|--------|
| Class-label leakage into features | **none** |
| Top-decile same-class neighbor rate | **0.67** |
| Novelty ↔ gene-count correlation | **0.12** (not size-dominated) |
| Top-50 size outliers flagged | **4** (e.g. mis-scaled mega-clusters) |
| Checks passed | **yes** |

![Stratified novelty audit](reports/figures/validation_novelty_by_class.png)

Top-ranked hits should be inspected for annotation quirks (e.g. unusually large gene counts) before treating them as discovery priorities — the audit surfaces those cases explicitly.

---

## Apply to new genomes

Curated predicted BGC domain tables are featurized with the MIBiG vocabulary and scored against the MIBiG manifold.

[`reports/predicted_novelty_ranking.csv`](reports/predicted_novelty_ranking.csv):

| Rank | Genome | BGC | Predicted class | Novelty | Nearest MIBiG |
|-----:|--------|-----|-----------------|--------:|---------------|
| 1 | Rare_actinobacterium_predicted | PRED0006 | other | 0.64 | BGC0002148 |
| 2 | Rare_actinobacterium_predicted | PRED0007 | NRPS | 0.64 | BGC0002608 |
| 3 | Myxococcus_sp_predicted | PRED0008 | hybrid | 0.64 | BGC0002608 |

This is the discovery loop: **reference atlas → novelty score → prioritize non-MIBiG candidates**.

---

## Limitations

- Product-class / bioactivity prediction are out of scope
- Raw MIBiG GenBank lacks antiSMASH domain calls; domains are inferred from CDS products
- UMAP is optional (`pip install '.[umap]'`); PCA is the reliable default here
- Predicted set is curated, not full antiSMASH-DB scale
- High novelty ≠ proven new chemistry — it is a prioritization signal

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
np-download → np-featurize → np-sanity → np-atlas → np-novelty → np-validate → np-apply
```

---

## Project layout

```text
src/npdiscovery/     package (data, featurize, models, atlas, novelty)
scripts/             thin wrappers + reproduce.sh
data/raw|processed/  MIBiG download + feature matrices (gitignored bulk)
data/external/       curated predicted BGCs
reports/             novelty rankings, metrics, figures
tests/               parse / featurize / novelty unit tests
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

---

## License

MIT
