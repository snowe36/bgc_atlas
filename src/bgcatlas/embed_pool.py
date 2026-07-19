"""CPU-side helpers for pooling protein ESM embeddings into per-BGC vectors."""

from __future__ import annotations

import numpy as np


def pool_bgcs(
    embeds: np.ndarray,
    bgc_ids: np.ndarray,
    aa_lengths: np.ndarray,
    pooling: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Pool protein embeddings → per-BGC matrix (mean | length_weighted)."""
    unique = sorted(set(bgc_ids.tolist()))
    id_to_pos = {b: i for i, b in enumerate(unique)}
    dim = embeds.shape[1]
    sums = np.zeros((len(unique), dim), dtype=np.float64)
    weights = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)

    for i, bgc_id in enumerate(bgc_ids):
        pos = id_to_pos[bgc_id]
        if pooling == "length_weighted":
            w = float(max(aa_lengths[i], 1))
        elif pooling == "mean":
            w = 1.0
        else:
            raise ValueError(f"Unknown pooling={pooling!r}; use mean|length_weighted")
        sums[pos] += embeds[i] * w
        weights[pos] += w
        counts[pos] += 1

    bgc_embeds = (sums / np.maximum(weights, 1e-9)[:, None]).astype(np.float32)
    return unique, bgc_embeds, counts


def representation_label(model: str, pooling: str) -> str:
    short = model.split("/")[-1].replace("esm2_", "esm2-").replace("_UR50D", "")
    return f"{short}_{pooling}"
