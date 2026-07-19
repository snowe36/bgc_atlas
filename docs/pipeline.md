# Pipeline & CLI

CPU atlas, validation, apply, and temporal holdout.
GPU / ESM2 / encoder: [`esm.md`](esm.md).

## Reproduce

```bash
git clone https://github.com/snowe36/bgc_atlas.git && cd bgc_atlas
uv sync --extra dev
bash scripts/reproduce.sh && uv run pytest -q
```

```text
bgc-download → bgc-featurize → bgc-sanity → bgc-atlas → bgc-novelty
    → run_case_studies.py → bgc-validate → bgc-apply → bgc-temporal
```

Optional UMAP maps: `uv sync --extra umap` (PCA is default).

## Stages

```bash
uv run bgc-download
uv run bgc-featurize
uv run bgc-sanity
uv run bgc-atlas
uv run bgc-novelty          # -k neighbors
uv run python scripts/run_case_studies.py
uv run bgc-validate
uv run bgc-apply
uv run bgc-temporal         # --n-controls, -k
```

Most CLIs take `-v` / `--verbose`.

## Apply

Score predicted BGCs against the MIBiG manifold
→ [`reports/predicted_novelty_ranking.csv`](../reports/predicted_novelty_ranking.csv).

```bash
uv run bgc-apply                                          # curated demo
uv run bgc-apply --input /path/to/antismash_outdir --genome MyStreptomyces
uv run bgc-apply --input /path/to/result.json
uv run bgc-apply --input data/external/predicted_domains.csv
# CSV columns: genome,bgc_id,predicted_class,gene_order,domain_id,n_genes
```

Flags: `--input`, `--genome`, `-k`.

## Layout

```text
src/bgcatlas/        package
scripts/             reproduce.sh, run_case_studies.py, run_esm_embed.py
docs/                pipeline.md, esm.md
data/raw|processed/  MIBiG + features (bulk gitignored)
data/external/       demo predicted BGCs
reports/             rankings, metrics, figures
tests/               unit tests + antiSMASH fixtures
```
