# GPU / ESM2 embeddings

Optional path for frozen protein-language-model embeddings and the novelty-representation comparison. The CPU atlas and myxobacteria case studies do **not** require this.

## Defaults

| Knob | Legacy (150M) | Current default |
|------|---------------|-----------------|
| Model | `esm2_t30_150M` | `esm2_t33_650M` |
| Pooling | uniform mean | **length-weighted** (longer enzymes count more) |
| Max AA / proteins | 700 / 60 | 1024 / 80 |
| Cache | BGC matrix only | + protein-level cache + `esm_embed_manifest.json` |

## Commands

```bash
uv sync --extra embed

# full GPU embed (writes esm_embeddings.npy + protein cache + manifest)
python scripts/run_esm_embed.py

# re-pool without GPU after the first run
python scripts/run_esm_embed.py --from-cache --pooling mean

# legacy 150M mean-pool bake-off
python scripts/run_esm_embed.py --model facebook/esm2_t30_150M_UR50D --pooling mean --max-aa 700

uv run bgc-ablation && uv run bgc-novelty-compare
```

Labels and knobs are recorded in `data/processed/esm_embed_manifest.json`.

## Contrastive encoder (requires protein cache)

```bash
uv sync --extra train
uv run bgc-train-encoder --objective supcon --pooling attention --embed-dim 256 --epochs 40 --prospective -v
uv run bgc-learned-eval -v
python scripts/run_encoder_sweep.py --device cuda   # or --quick --device cpu
```

GPU launch helpers: [`scripts/runpod/launch_train_job.py`](../scripts/runpod/launch_train_job.py) (`--sweep` / `--terminate`).
