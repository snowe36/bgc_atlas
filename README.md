# bgc_atlas

**Biosynthetic architecture atlas** — a CPU-only, reproducible pipeline to map microbial BGC space and rank architecture-novel regions.

> Microbial genomes encode far more biosynthetic gene clusters than have been experimentally characterized ([MIBiG](https://mibig.secondarymetabolites.org/); [antiSMASH](https://docs.antismash.secondarymetabolites.org/)). This project builds a CPU-only ML pipeline to map **biosynthetic architecture space** and prioritize clusters that sit far from known neighborhoods.

**Talk track:** *I represented BGC architecture, showed the representation recovers known families, then ranked clusters by distance from characterized neighborhoods—with explicit checks against label leakage and size artifacts.*

Repo: [github.com/snowe36/bgc_atlas](https://github.com/snowe36/bgc_atlas)

---

## Motivation

Microbial genomes encode biosynthetic gene clusters (BGCs) for polyketides, nonribosomal peptides, terpenes, RiPPs, and more. Public resources such as **MIBiG** (experimentally characterized BGCs) and **antiSMASH** (genome-wide BGC prediction) make clear that sequenced genomes contain many more clusters than have been linked to molecules ([MIBiG 3.0/4.0](https://mibig.secondarymetabolites.org/); [Blin et al., *NAR*](https://doi.org/10.1093/nar/gkad984)).

The interesting discovery question is **not** “can we classify known BGCs?” (rule-based tools already do this well), but:

**Can we identify BGCs that sit far from known biosynthetic architecture?**

That is the centerpiece of this repo: an atlas of MIBiG architecture space plus a ranked **architecture-novelty** table.

This project intentionally uses **interpretable, CPU-friendly representations** rather than large pretrained models. The goal is to understand biosynthetic space structure and evaluate discovery heuristics before introducing more complex embeddings.

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

**2,762 × 342** feature matrix. Biosynth class labels are used for coloring and sanity checks only — **never** as novelty features.

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

NRPS/PKS separate cleanly; hybrid and “other” are harder (as expected). Full metrics: [`reports/sanity_metrics.json`](reports/sanity_metrics.json).

---

## Atlas

Architecture features → standardized PCA (**50-D**, ~72% variance) for distances; **2-D PCA map** for visualization (UMAP when `umap-learn` installs cleanly; PCA fallback is the default on this stack).

PCA is used as a compact representation for distance calculations to reduce noise from sparse, high-dimensional architecture features before nearest-neighbor scoring.

![Biosynthetic space by class](reports/figures/atlas_by_class.png)

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

## Limitations

- Scores reflect **architecture** divergence, not proven new chemistry
- Product-class / bioactivity prediction are out of scope
- Raw MIBiG GenBank lacks antiSMASH domain calls; domains are inferred from CDS products
- UMAP is optional (`pip install '.[umap]'`); PCA is the reliable default here
- Predicted set is a curated demo, not full antiSMASH-DB scale

---

## Future directions

- **Temporal holdout:** fit the reference manifold on older MIBiG releases and ask whether later-added entries rank as architecture-novel *before* they enter the catalog (BGC analogue of a prospective / time split)
- Richer domain calls from antiSMASH-annotated GenBank
- Optional protein-embedding features once CPU baselines are solid
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
bgc-download → bgc-featurize → bgc-sanity → bgc-atlas → bgc-novelty → bgc-validate → bgc-apply
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
