"""Shared pipeline defaults (hash dims, PCA, novelty k, temporal cutoffs)."""

from __future__ import annotations

# Featurize
N_HASH_DIMS = 256
MIN_DOMAIN_FREQ = 3

# Atlas / novelty embeddings
PCA_N_COMPONENTS = 50
DEFAULT_NOVELTY_K = 5

# Temporal holdout
DEFAULT_TEMPORAL_CUTOFF = "2022-09-16"
# Classes treated as "major families" for stratified temporal analysis
MAJOR_FAMILIES = ("PKS", "NRPS", "hybrid")
