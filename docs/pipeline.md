# Pipeline & CLI reference

Full command surface for the CPU atlas, validation, apply, and temporal holdout.
GPU / ESM2 / contrastive encoder commands live in [`esm.md`](esm.md).

## One-command reproduce

```bash
git clone https://github.com/snowe36/bgc_atlas.git && cd bgc_atlas
uv sync --extra dev
bash scripts/reproduce.sh && uv run pytest -q
```

CPU stages (locked deps via [`uv.lock`](../uv.lock); CI on every push):

```text
bgc-download → bgc-featurize → bgc-sanity → bgc-atlas → bgc-novelty
    → run_case_studies.py → bgc-validate → bgc-apply → bgc-temporal
```

UMAP for 2-D maps (optional): `uv sync --extra umap` (PCA is the default).

## Stage commands

```bash
uv run bgc-download
uv run bgc-featurize
uv run bgc-sanity
uv run bgc-atlas
uv run bgc-novelty          # -k neighbors (default from config)
uv run python scripts/run_case_studies.py
uv run bgc-validate
uv run bgc-apply            # see Apply below
uv run bgc-temporal         # --n-controls, -k
```

Most CLIs accept `-v` / `--verbose`.

## Apply to new genomes

Score predicted BGCs against the MIBiG manifold
(`bgc-apply` → [`reports/predicted_novelty_ranking.csv`](../reports/predicted_novelty_ranking.csv)).
Supports antiSMASH region GenBanks and JSON.

```bash
# curated demo (default)
uv run bgc-apply

# antiSMASH region GenBanks (preferred — domains from aSDomain / PFAM_domain)
uv run bgc-apply --input /path/to/antismash_outdir --genome MyStreptomyces

# antiSMASH JSON (areas/products; CDS domains when present)
uv run bgc-apply --input /path/to/result.json

# pre-normalized domains CSV
# columns: genome,bgc_id,predicted_class,gene_order,domain_id,n_genes
uv run bgc-apply --input data/external/predicted_domains.csv
```

Flags: `--input`, `--genome`, `-k` (neighbors).

## Project layout

```text
src/bgcatlas/        package (config, embed_pool, data/antismash, featurize, models, atlas, novelty)
scripts/             reproduce.sh, run_case_studies.py, run_esm_embed.py
docs/esm.md          GPU embedding + encoder commands
docs/pipeline.md     this file
uv.lock              locked dependency versions
data/raw|processed/  MIBiG download + feature matrices (gitignored bulk)
data/external/       demo predicted BGCs + last apply cache
reports/             rankings, metrics, figures, biological_case_studies.json
tests/               unit tests + antiSMASH fixtures
.github/workflows/   CI (uv sync + ruff + pytest)
```
