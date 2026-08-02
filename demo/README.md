# bgc_atlas demo (5 minutes)

See the architecture-novelty finding **without** re-downloading MIBiG (uses committed figures; scores curated predicted BGCs when a local manifold exists).

## Run

```bash
# from repo root
bash demo/run_demo.sh
# or:
bgc-demo
```

## What you get

| Output | Meaning |
|--------|---------|
| `outputs/biological_case_studies.png` | Myxo enrichment survives ≤60-gene control |
| `outputs/novelty_representation_comparison.png` | Architecture vs ESM novelty disagreement |
| `outputs/predicted_novelty_ranking.csv` | Curated demo genomes scored on the manifold |

## Requirements

- Python ≥ 3.11 (`uv` preferred; pip works)
- CPU only

## Not in the demo

Full `scripts/reproduce.sh` (download → featurize → temporal) for regenerating the MIBiG manifold from scratch.
