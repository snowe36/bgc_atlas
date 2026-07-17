# bgc_atlas

**Biosynthetic architecture atlas** — a CPU-only, reproducible pipeline that maps microbial biosynthetic gene cluster (BGC) space and ranks architecturally novel regions, with a prospective (time-split) test of whether that novelty score actually predicts what gets discovered next.

[CI](https://github.com/snowe36/bgc_atlas/actions/workflows/ci.yml)
[License: MIT](LICENSE)
Python 3.11+

Repo: [github.com/snowe36/bgc_atlas](https://github.com/snowe36/bgc_atlas)

---



## The question

Microbial genomes encode far more biosynthetic gene clusters than have been experimentally characterized ([MIBiG](https://mibig.secondarymetabolites.org/); [antiSMASH](https://docs.antismash.secondarymetabolites.org/)). Rule-based tools already answer "is this a BGC, and what class?" well. The harder, more useful question is:

**Can we identify BGCs that sit far from known biosynthetic architecture — and does that novelty score actually mean anything?**

This repo builds a reference atlas of MIBiG architecture space, ranks clusters by distance from characterized neighborhoods, and then stress-tests that ranking with label-leakage checks, size-confound checks, and a prospective (time-split) holdout — reporting the result honestly even where it doesn't confirm the hypothesis.

Biosynthetic space by class

**Talk track:** *I represented BGC architecture, showed the representation recovers known families, ranked clusters by distance from characterized neighborhoods, and validated the strategy with explicit leakage/size-confound checks plus a prospective temporal holdout — reporting the negative result on the holdout rather than hiding it.*

---



## Key results at a glance


| Check                                                                                 | Result                                                                                                                                             |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Representation recovers known biosynthetic classes                                    | Random forest macro-F1 **0.76**, weighted-F1 **0.79** (5-fold stratified CV)                                                                       |
| Class labels leak into novelty features                                               | **No** (`bgc-validate` audit)                                                                                                                      |
| Novelty score dominated by cluster size                                               | **No** — Spearman(novelty, gene count) = **0.12**                                                                                                  |
| Top-decile novelty hits share their class with their nearest neighbor                 | **67%** of the time (i.e. novelty ≠ misclassification)                                                                                             |
| Prospective test: do chronologically newer MIBiG entries score as architecture-novel? | **Not supported** — see [Prospective validation](#prospective-temporal-holdout-validation) below; reported as a negative result, not smoothed over |
| ESM2 protein embeddings alone vs. hashed architecture (class recovery, macro-F1)      | ESM2 **0.76** ≈ hashed **0.78** — comparable, not better, alone                                                                                    |
| Combining hashed + ESM2 (macro-F1)                                                    | **0.83** — clear lift over either alone (see [GPU embeddings](#gpu--protein-language-model-embeddings))                                            |
| Does the novelty *ranking* agree across representations?                              | **No** — hashed vs. ESM2 novelty is weakly *negatively* correlated (Spearman ρ=-0.42); top-decile overlap is only 1.5%                             |
| Test suite                                                                            | **6/6 passing**, run on every push via CI (badge above)                                                                                            |


---



## Why this matters

Public resources such as **MIBiG** (experimentally characterized BGCs) and **antiSMASH** (genome-wide BGC prediction) make it clear that sequenced genomes contain many more clusters than have been linked to molecules ([MIBiG 3.0/4.0](https://mibig.secondarymetabolites.org/); [Blin et al., *NAR](https://doi.org/10.1093/nar/gkad984)*). Deciding *where to look next* for new chemistry is a triage problem, not a classification problem — and triage tools need to be validated, not just built.

That's the standard this project holds itself to: the same rigor a wet-lab experiment demands — negative controls, holdouts, and reporting results whether or not they support the hypothesis — applied to a computational discovery pipeline. Concretely: no result here is reported without checking whether it's an artifact of class labels, cluster size, or evaluation leakage, and the prospective holdout in this repo is a genuine hold-out-in-time test that gets reported even when it comes back negative.

This project intentionally starts with **interpretable, CPU-friendly representations** rather than large pretrained models, to understand biosynthetic space structure and validate discovery heuristics before adding more complex embeddings (see [GPU / protein language model embeddings](#gpu--protein-language-model-embeddings) below).

---



## Reproducibility

- **Environment is pinned**: Python 3.11+ (see `.python-version`), locked deps in [`uv.lock`](uv.lock) via [uv](https://docs.astral.sh/uv/).
- **CI runs the test suite on every push/PR** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) — see badge at the top of this README.
- **One-command reproduction**: `bash scripts/reproduce.sh` runs the full pipeline end-to-end from a clean environment (download → featurize → sanity-check → atlas → novelty → validate → apply); takes a few minutes on a laptop CPU.
- **Unit tests** cover parsing, featurization, and novelty scoring on tiny fixtures ([`tests/test_pipeline.py`](tests/test_pipeline.py)) — no network or GPU required.
- **Every reported number has a file behind it**: metrics, rankings, and audits are written to [`reports/`](reports/) as JSON/CSV, not just printed to a log.

Quick start (full walkthrough is in [How to reproduce](#how-to-reproduce)):

```bash
git clone https://github.com/snowe36/bgc_atlas.git && cd bgc_atlas
uv sync --extra dev
bash scripts/reproduce.sh && uv run pytest -q
```

---



## Data


| Item              | Detail                                                                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Source            | [MIBiG 4.0](https://mibig.secondarymetabolites.org/) JSON + GenBank ([download mirror](https://dl.secondarymetabolites.org/mibig/))                                |
| Scope             | Experimentally characterized reference BGCs                                                                                                                        |
| Parsed            | **3,013** JSON entries · **2,636** GenBank records                                                                                                                 |
| Featurized        | **2,762** BGCs with gene annotations                                                                                                                               |
| Classes           | PKS 717 · NRPS 556 · other 482 · hybrid 413 · RiPP 413 · terpene 181                                                                                               |
| Protein sequences | **46,957** CDS translations extracted from GenBank (used for [GPU embeddings](#gpu--protein-language-model-embeddings)); **2,636** BGCs have ≥1 usable translation |
| Temporal metadata | **100%** of BGCs have a real submission date from MIBiG's changelog (used for [prospective validation](#prospective-temporal-holdout-validation))                  |
| Demo set          | Curated predicted BGCs in `[data/external/](data/external/)` (workflow illustration only)                                                                          |


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


| Model               | Macro-F1 | Weighted-F1 |
| ------------------- | -------- | ----------- |
| Logistic regression | 0.65     | 0.68        |
| **Random forest**   | **0.76** | **0.79**    |


Confusion matrixPer-class F1

NRPS/PKS separate cleanly; hybrid and "other" are harder (as expected). Full metrics: `[reports/sanity_metrics.json](reports/sanity_metrics.json)`.

---



## Atlas

Architecture features → standardized PCA (**50-D**, ~72% variance) for distances; **2-D PCA map** for visualization (UMAP when `umap-learn` installs cleanly; PCA fallback is the default on this stack).

PCA is used as a compact representation for distance calculations to reduce noise from sparse, high-dimensional architecture features before nearest-neighbor scoring. The 2-D visualizations below use robust (percentile-based) axis limits so that a handful of extreme size outliers — already flagged by the validation audit — don't collapse the whole plot into a corner; the count of off-frame points is annotated on the plot itself rather than hidden.

Per-class neighborhoods

---



## Novelty scoring

Leave-one-out **kNN distance** + **local rarity** in PCA space → composite score ∈ [0, 1].

> **Definition.** Here, *novelty* means divergence in **biosynthetic architecture space**, not experimentally confirmed chemical novelty. Different architectures can yield related molecules, and similar architectures can produce divergent chemistry.

Hero artifact: `[reports/novelty_ranking.csv](reports/novelty_ranking.csv)`


| Rank | BGC ID     | Organism                        | Class  | Score | Nearest MIBiG |
| ---- | ---------- | ------------------------------- | ------ | ----- | ------------- |
| 1    | BGC0002977 | *Bacillus subtilis* fmb60       | hybrid | 1.00  | BGC0000081    |
| 2    | BGC0000103 | *Mycobacterium ulcerans* Agy99  | PKS    | 1.00  | BGC0000038    |
| 3    | BGC0002124 | *Actinomadura verrucosospora*   | PKS    | 1.00  | BGC0002587    |
| 4    | BGC0000315 | *Streptomyces coelicolor* A3(2) | NRPS   | 1.00  | BGC0000324    |
| 5    | BGC0002808 | *Streptomyces scabiei* 87.22    | PKS    | 1.00  | BGC0001063    |


Known vs high-novelty overlayNovelty by class

Hybrids and PKS sit higher on average; RiPPs are denser / more self-similar in this feature space.

---



## Validation

Integrity checks are first-class (`bgc-validate` → `[reports/validation_audit.json](reports/validation_audit.json)`):


| Check                                 | Result                                |
| ------------------------------------- | ------------------------------------- |
| **Class-label leakage into features** | **none**                              |
| Top-decile same-class neighbor rate   | **0.67**                              |
| Novelty ↔ gene-count correlation      | **0.12** (not size-dominated)         |
| Top-50 size outliers flagged          | **4** (e.g. mis-scaled mega-clusters) |
| Checks passed                         | **yes**                               |


Stratified novelty audit

Top-ranked hits should be inspected for annotation quirks (e.g. unusually large gene counts) before treating them as priorities — the audit surfaces those cases explicitly.

---



## Prospective (temporal-holdout) validation

The checks above confirm the novelty score isn't an artifact of labels or size — but they don't show it's *predictive* of anything. MIBiG's changelog carries a real submission date per entry, which makes a genuine prospective test possible: **fit the reference manifold only on BGCs added before a cutoff date, then ask whether entries added *after* that cutoff — which the model never saw — score as architecture-novel**, relative to a size-matched random-holdout control drawn from the same reference corpus (`bgc-temporal` → `[src/bgcatlas/novelty/temporal.py](src/bgcatlas/novelty/temporal.py)`, `[reports/temporal_holdout.json](reports/temporal_holdout.json)`).

This is the BGC analogue of a time-split evaluation: if the score were more than a rank-ordering artifact, chronologically newer entries should skew novel relative to a random slice of the same reference corpus.


| Cutoff     | Reference BGCs | Held-out (post-cutoff) BGCs | Held-out mean novelty | Random-control mean novelty (50 resamples) | Mann-Whitney p (held-out > control) |
| ---------- | -------------- | --------------------------- | --------------------- | ------------------------------------------ | ----------------------------------- |
| 2022-09-16 | 2,472          | 290                         | **0.397**             | **0.495** (± 0.018)                        | **0.997**                           |


Prospective novelty: random vs. true post-cutoff holdout

**This does not support the hypothesis.** Post-cutoff MIBiG entries scored *less* architecture-novel than a random control, not more — the opposite of what "novelty predicts what gets added next" would require. Reported here as-is rather than reframed after the fact. The most likely reading: recent MIBiG additions skew toward incremental variants of well-studied families (NRPS/PKS/hybrid mean novelty stayed high; RiPP/terpene mean novelty stayed low — consistent with the class-level pattern in [Validation](#validation) rather than a temporal effect), so "added recently" and "architecturally novel" are measuring different things in this corpus. That's a useful, falsifiable finding about what this novelty score does and doesn't capture, not a result to paper over.

---



## Apply to new genomes (demo)

**Demonstration dataset only — not a discovery claim.** Curated predicted BGC domain tables in `[data/external/](data/external/)` illustrate the ranking workflow: featurize with the MIBiG vocabulary and score against the MIBiG manifold.

`[reports/predicted_novelty_ranking.csv](reports/predicted_novelty_ranking.csv)`:


| Rank | Genome (demo label)            | BGC      | Predicted class | Score | Nearest MIBiG |
| ---- | ------------------------------ | -------- | --------------- | ----- | ------------- |
| 1    | Rare_actinobacterium_predicted | PRED0006 | other           | 0.64  | BGC0002148    |
| 2    | Rare_actinobacterium_predicted | PRED0007 | NRPS            | 0.64  | BGC0002608    |
| 3    | Myxococcus_sp_predicted        | PRED0008 | hybrid          | 0.64  | BGC0002608    |


Workflow: **reference atlas → architecture-novelty score → prioritize non-MIBiG candidates**.

---



## GPU / protein language model embeddings

The representation above is deliberately CPU-only and interpretable so the discovery strategy could be validated first (see [Representation benchmark](#representation-benchmark)). The natural next step: does a real protein language model change the picture? `[scripts/run_esm_embed.py](scripts/run_esm_embed.py)` embeds every MIBiG CDS translation with **ESM2** (`facebook/esm2_t30_150M_UR50D`) and mean-pools per BGC into a 640-D vector — the one GPU-dependent step in the pipeline, run standalone on a rented GPU pod (a $0.44/hr A40, ~10 minutes of actual compute) rather than inside the CPU-only local pipeline, then copied back in.

**Classification ablation** (`bgc-ablation` → `[src/bgcatlas/models/ablation.py](src/bgcatlas/models/ablation.py)`, `[reports/ablation_metrics.json](reports/ablation_metrics.json)`) — same 5-fold CV protocol as the representation benchmark, on the 2,636 BGCs with both representations:


| Representation                     | Macro-F1 | Weighted-F1 |
| ---------------------------------- | -------- | ----------- |
| Hashed architecture (CPU baseline) | 0.78     | 0.81        |
| ESM2 (150M, mean-pooled) alone     | 0.76     | 0.76        |
| **Combined (hashed + ESM2)**       | **0.83** | **0.85**    |


Representation ablation

ESM2 embeddings alone are *not* better than the hand-built architecture features for recovering known biosynthetic classes — but they carry complementary signal: concatenating the two lifts macro-F1 from 0.78 to 0.83. That's the honest result: a foundation model isn't automatically an upgrade here, but it isn't wasted either.

**Does this change the novelty ranking?** (`bgc-novelty-compare` → `[src/bgcatlas/novelty/embed_compare.py](src/bgcatlas/novelty/embed_compare.py)`, `[reports/novelty_representation_comparison.json](reports/novelty_representation_comparison.json)`) — a lot, it turns out:


| Comparison                  | Spearman ρ | Top-decile Jaccard overlap |
| --------------------------- | ---------- | -------------------------- |
| Hashed vs. ESM2 novelty     | **-0.42**  | **1.5%**                   |
| Hashed vs. combined novelty | 0.06       | 19.5%                      |
| ESM2 vs. combined novelty   | —          | 46.7%                      |


Novelty representation comparison

Hashed-architecture novelty and ESM2-embedding novelty pick almost entirely *different* BGCs as "most novel," and are weakly negatively correlated. This is an important caveat, not a footnote: "architecture-novel" is representation-dependent, not a property of the BGC itself. Treat any single novelty ranking (this repo's headline one included) as one lens on divergence, not a ground truth — this is exactly why the [validation](#validation) and [prospective holdout](#prospective-temporal-holdout-validation) checks matter more than the ranking alone.

---



## Limitations

- Scores reflect **architecture** divergence, not proven new chemistry
- The prospective temporal holdout (above) did not confirm that architecture-novelty predicts which entries get added to MIBiG next — treat the novelty score as a within-corpus divergence measure, not a validated discovery-timing signal
- **Novelty rankings are representation-dependent**: hashed-architecture and ESM2-embedding novelty barely agree (Spearman ρ=-0.42, 1.5% top-decile overlap) — see [GPU embeddings](#gpu--protein-language-model-embeddings). Don't read the headline ranking as *the* answer; it's one lens among several this repo checks against each other
- Product-class / bioactivity prediction are out of scope
- Raw MIBiG GenBank lacks antiSMASH domain calls; domains are inferred from CDS products
- UMAP is optional (`uv sync --extra umap`); PCA is the reliable default here
- ESM2 embeddings use the 150M-parameter checkpoint mean-pooled per BGC (capped at 700 aa/protein, 60 proteins/BGC) — a larger checkpoint or a pooling scheme aware of domain boundaries would likely change the ablation numbers
- Predicted set is a curated demo, not full antiSMASH-DB scale

---



## Future directions

- Investigate *why* the temporal holdout came back negative (e.g. does restricting to non-major-family entries, or a longer lead time before the cutoff, change the picture?)
- Investigate *why* hashed and ESM2 novelty disagree so strongly — e.g. do they disagree more for some biosynth classes than others?
- Larger ESM2 checkpoint (650M+) with domain-aware pooling instead of uniform mean-pooling
- Richer domain calls from antiSMASH-annotated GenBank
- Larger curated antiSMASH-DB expansions for real genome prioritization

---



## How to reproduce

Requires [uv](https://docs.astral.sh/uv/) (installs/manages Python 3.11+ for you):

```bash
# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
git clone https://github.com/snowe36/bgc_atlas.git
cd bgc_atlas
uv sync --extra dev
bash scripts/reproduce.sh
uv run pytest -q
```

Step-through CLI (prefix with `uv run`, all CPU-only except the optional GPU embedding step):

```text
bgc-download → bgc-featurize → bgc-sanity → bgc-atlas → bgc-novelty → bgc-validate → bgc-apply → bgc-temporal
                                                                                                        │
                                          [GPU pod] scripts/run_esm_embed.py ──────────────────────────┘
                                                          │
                                                          ▼
                                          bgc-ablation → bgc-novelty-compare
```

---



## Project layout

```text
src/bgcatlas/        package (data, featurize, models, atlas, novelty)
scripts/             thin wrappers + reproduce.sh + run_esm_embed.py (GPU-only)
uv.lock              locked dependency versions (uv)
data/raw|processed/  MIBiG download + feature matrices (gitignored bulk)
data/external/       demo predicted BGCs (not discovery claims)
reports/             rankings, metrics, figures
tests/               parse / featurize / novelty unit tests
.github/workflows/   CI (uv sync + pytest on every push/PR)
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
10. GPU protein embeddings + representation ablation

---



## Acknowledgements

Developed with the assistance of [Cursor](https://cursor.com), the AI code editor, which supported code authoring, refactoring, and documentation throughout this project.

---



## License

MIT