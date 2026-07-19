# GPU / ESM2

Optional. CPU atlas and myxo case studies do not need this.

## Defaults

| Knob | Legacy (150M) | Default |
|------|---------------|---------|
| Model | `esm2_t30_150M` | `esm2_t33_650M` |
| Pooling | mean | **length-weighted** |
| Max AA / proteins | 700 / 60 | 1024 / 80 |
| Cache | BGC matrix only | + protein cache + manifest |

## Commands

```bash
uv sync --extra embed
python scripts/run_esm_embed.py
python scripts/run_esm_embed.py --from-cache --pooling mean
python scripts/run_esm_embed.py --model facebook/esm2_t30_150M_UR50D --pooling mean --max-aa 700
uv run bgc-ablation && uv run bgc-novelty-compare
```

Manifest: `data/processed/esm_embed_manifest.json`.

## Contrastive encoder

Needs the protein cache.

```bash
uv sync --extra train
uv run bgc-train-encoder --objective supcon --pooling attention --embed-dim 256 --epochs 40 --prospective -v
uv run bgc-learned-eval -v
python scripts/run_encoder_sweep.py --device cuda   # or --quick --device cpu
```

RunPod helpers: [`scripts/runpod/launch_train_job.py`](../scripts/runpod/launch_train_job.py).
