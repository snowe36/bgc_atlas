"""Self-contained demo: score curated predicted BGCs; no MIBiG re-download required
when processed features already exist (or after one local curate).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from bgcatlas.paths import FIGURES, PROCESSED, REPORTS, ROOT, ensure_dirs

DEMO = ROOT / "demo"
DEMO_OUT = DEMO / "outputs"

_HERO_FIGS = [
    "biological_case_studies.png",
    "novelty_representation_comparison.png",
    "atlas_by_class.png",
    "temporal_holdout.png",
    "novelty_by_class.png",
]


def _stage_committed() -> list[str]:
    copied: list[str] = []
    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    for name in _HERO_FIGS:
        src = FIGURES / name
        if src.exists():
            shutil.copy2(src, DEMO_OUT / name)
            copied.append(name)
    for name in (
        "biological_case_studies.json",
        "predicted_novelty_ranking.csv",
        "temporal_holdout.json",
    ):
        src = REPORTS / name
        if src.exists():
            shutil.copy2(src, DEMO_OUT / name)
            copied.append(name)
    return copied


def _headline() -> str:
    path = REPORTS / "biological_case_studies.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    phy = data.get("phylogeny") or {}
    sc = (
        phy.get("size_controlled_n_genes_le_60")
        or phy.get("size_controlled")
        or data.get("size_controlled")
        or {}
    )
    or_ = sc.get("odds_ratio_top_decile")
    if or_ is None:
        return "See biological_case_studies.json for the myxo enrichment finding."
    return (
        f"Myxobacteria architecture enrichment (≤60 genes): OR={float(or_):.1f} "
        f"(see demo/outputs/biological_case_studies.png)."
    )


def run_demo() -> dict:
    """Stage hero outputs; score demo predicted BGCs when the manifold is present."""
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    ensure_dirs()
    DEMO_OUT.mkdir(parents=True, exist_ok=True)

    print("=== bgc_atlas demo (curated predicted BGCs; no fresh MIBiG download) ===")
    copied = _stage_committed()

    features = PROCESSED / "feature_matrix.npy"
    if features.exists():
        from bgcatlas.cli import apply_main

        apply_main([])
        ranking = REPORTS / "predicted_novelty_ranking.csv"
        if ranking.exists():
            shutil.copy2(ranking, DEMO_OUT / ranking.name)
            if ranking.name not in copied:
                copied.append(ranking.name)
        print("Scored curated demo BGCs against the local MIBiG manifold.")
    else:
        print(
            "No local feature_matrix.npy — staged committed figures/tables only.\n"
            "For a full regenerate: bash scripts/reproduce.sh"
        )

    headline = _headline()
    summary = {"outputs": str(DEMO_OUT), "copied": copied, "headline": headline}
    print("")
    print(f"Demo outputs → {DEMO_OUT}")
    for c in sorted(set(copied)):
        print(f"  - {c}")
    if headline:
        print(headline)
    return summary
