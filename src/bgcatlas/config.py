"""Shared pipeline defaults (hash dims, PCA, novelty k, temporal cutoffs, ESM V2)."""

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

# ESM / GPU embeddings (V2 defaults)
DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
LEGACY_ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
DEFAULT_ESM_POOLING = "length_weighted"  # mean | length_weighted
DEFAULT_ESM_MAX_AA = 1024
DEFAULT_ESM_MAX_PROTEINS = 80
DEFAULT_ESM_BATCH_TOKENS = 6000
