# np-discovery

**Biosynthetic novelty atlas** — a CPU-only ML pipeline to map microbial BGC space and rank unexplored regions.

> Inspired by the observation that microbial genomes contain vast unexplored chemical diversity, this project builds a CPU-only ML pipeline to map biosynthetic novelty and identify candidate unexplored regions of BGC space.

---

## Motivation

Known natural products (MIBiG) represent only a fraction of microbial chemical diversity. Many biosynthetic gene clusters (BGCs) are silent or encode chemistry distant from characterized pathways.

**Scientific question:** Can we identify BGCs that sit far from known chemistry in sequence/architecture space?

---

## Pipeline (git history mirrors this story)

1. Initialize project
2. Acquire biological data (MIBiG)
3. Represent biosynthetic pathways
4. Benchmark ML representations
5. Map biosynthetic space
6. Identify unexplored regions
7. Validate discovery strategy
8. Apply to new genomes

---

## Status

Scaffold only — subsequent commits build the discovery pipeline.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## License

MIT
