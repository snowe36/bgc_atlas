# np-discovery

**Biosynthetic novelty atlas** — a CPU-only ML pipeline to map microbial BGC space and rank unexplored regions.

> Inspired by the observation that microbial genomes contain vast unexplored chemical diversity, this project builds a CPU-only ML pipeline to map biosynthetic novelty and identify candidate unexplored regions of BGC space.

**Talk track:** *Known natural products are a thin slice of microbial chemistry. I built a representation of biosynthetic architecture, showed it recovers known BGC families, then ranked clusters by distance from characterized neighborhoods.*

Remote: [github.com/snowe36/bgc_atlas](https://github.com/snowe36/bgc_atlas)

---

## Motivation

Microbial genomes encode biosynthetic gene clusters (BGCs) for polyketides, nonribosomal peptides, terpenes, RiPPs, and more. Many are silent or chemically uncharacterized. The interesting discovery question is not “can we classify known BGCs?” (antiSMASH already does this well) but:

**Can we identify biosynthetic space that is distant from known chemistry?**

---

## Data

| Source | Role |
|--------|------|
| [MIBiG 4.0](https://mibig.secondarymetabolites.org/) JSON + GenBank | Experimentally characterized reference BGCs (~3k entries) |
| Curated predicted BGCs (`data/external/`) | Stretch: score non-MIBiG candidates against the MIBiG manifold |

Pipeline: download → parse metadata/domains → tidy parquet tables under `data/processed/`.

---

## Representation

CPU-friendly **pathway architecture features** (no protein LMs in v1):

- Domain / product-derived token counts
- Hashed ordered domain architecture (unigrams + bigrams)
- Cluster size statistics (gene count, length, etc.)

Biosynth class labels are used for coloring and sanity checks only — **not** as novelty features.

---

## Representation benchmark

Logistic regression and random forest recover coarsened MIBiG classes (NRPS / PKS / RiPP / terpene / hybrid / other) from the same features via stratified CV.

If families separate, distances in this space are biologically meaningful for novelty ranking.

See `reports/sanity_metrics.json` and `reports/figures/sanity_*.png`.

---

## Atlas

PCA (50-D for distances; 2-D map via UMAP when installed, else PCA) of architecture space, colored by biosynth class.

Figures: `reports/figures/atlas_by_class.png`, `atlas_class_facets.png`.

---

## Novelty ranking (hero)

Leave-one-out **kNN distance** + **local rarity** in PCA space → composite novelty ∈ [0, 1].

Hero artifact: [`reports/novelty_ranking.csv`](reports/novelty_ranking.csv)

| Rank | BGC | Organism | Class | Novelty | Nearest MIBiG |
|------|-----|----------|-------|---------|---------------|
| … | … | … | … | … | … |

Overlays: `novelty_overlay.png`, `novelty_by_class.png`.

---

## Validation

- No class-label leakage into features
- Stratified novelty summaries by biosynth class
- Flag top-novelty size outliers (mis-parsed mega-clusters)
- Same-class neighbor rate among high-novelty BGCs

`np-validate` → `reports/validation_audit.json`

---

## Apply to new genomes

Curated predicted BGC domain tables are featurized with the MIBiG vocabulary and scored against the MIBiG reference manifold (`np-apply` → `reports/predicted_novelty_ranking.csv`).

---

## Limitations

- Product-class / bioactivity prediction are out of scope
- Raw MIBiG GenBank lacks antiSMASH domain calls; domains are inferred from CDS products
- UMAP optional (numba/llvmlite may fail on some macOS setups); PCA fallback is default here
- Not full antiSMASH-DB scale — curated expansion only

---

## How to reproduce

Requires **Python 3.11+** (Homebrew: `brew install python@3.11`).

```bash
git clone https://github.com/snowe36/bgc_atlas.git
cd bgc_atlas   # or np-discovery
/usr/local/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/reproduce.sh
```

Or step through: `np-download` → `np-featurize` → `np-sanity` → `np-atlas` → `np-novelty` → `np-validate` → `np-apply`.

```bash
pytest -q
```

---

## Pipeline / git history

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
