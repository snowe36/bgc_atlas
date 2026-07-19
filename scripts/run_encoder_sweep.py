#!/usr/bin/env python
"""Sweep contrastive BGC encoder hyperparameters and collect honest-eval metrics.

Grid (compact — fits a ~$5–15 A40 hour):
  objective × pooling × embed_dim × keep_frac × seed

For each config:
  1. Train with --prospective (leakage-safe temporal split)
  2. Run learned eval suite (class recovery, novelty, temporal)
  3. Record key metrics into reports/encoder_sweep_results.json

Usage:
    uv sync --extra train
    python scripts/run_encoder_sweep.py --device cuda
    # quick local smoke:
    python scripts/run_encoder_sweep.py --quick --device cpu
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time

import matplotlib.pyplot as plt
import pandas as pd

from bgcatlas.models.train_contrastive import TrainConfig, train_encoder
from bgcatlas.novelty.learned_eval import run_learned_eval_suite
from bgcatlas.paths import FIGURES, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)


def build_grid(quick: bool) -> list[dict]:
    if quick:
        return [
            {
                "objective": "simclr",
                "pooling": "attention",
                "embed_dim": 128,
                "keep_frac": 0.7,
                "feat_dropout": 0.1,
                "seed": 42,
                "epochs": 3,
            }
        ]
    # Compact grid (~12 cells): fits a ~1h A40 job with full honest eval each.
    # Covers objective × pooling × embed_dim; keep_frac/seed held at defaults,
    # plus one seed-replicate on the default config.
    grid = []
    for obj, pool, dim in itertools.product(
        ["simclr", "supcon"],
        ["attention", "mean", "deepsets"],
        [128, 256],
    ):
        grid.append(
            {
                "objective": obj,
                "pooling": pool,
                "embed_dim": dim,
                "keep_frac": 0.7,
                "feat_dropout": 0.1,
                "seed": 42,
                "epochs": 30,
            }
        )
    # Seed replicate on the default (simclr/attention/256)
    grid.append(
        {
            "objective": "simclr",
            "pooling": "attention",
            "embed_dim": 256,
            "keep_frac": 0.7,
            "feat_dropout": 0.1,
            "seed": 7,
            "epochs": 30,
        }
    )
    return grid


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Contrastive BGC encoder sweep")
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true", help="Single tiny run for smoke tests")
    ap.add_argument("--max-runs", type=int, default=None, help="Cap number of grid cells")
    args = ap.parse_args()

    ensure_dirs()
    grid = build_grid(quick=args.quick)
    if args.max_runs is not None:
        grid = grid[: args.max_runs]

    results: list[dict] = []
    t0 = time.time()
    for i, cell in enumerate(grid, 1):
        LOG.info("=== sweep %d/%d: %s ===", i, len(grid), cell)
        cfg = TrainConfig(
            objective=cell["objective"],
            pooling=cell["pooling"],
            embed_dim=cell["embed_dim"],
            keep_frac=cell["keep_frac"],
            feat_dropout=cell["feat_dropout"],
            seed=cell["seed"],
            epochs=cell["epochs"],
            train_cutoff="2022-09-16",
            device=args.device,
            batch_size=64 if not args.quick else 16,
        )
        try:
            manifest = train_encoder(cfg)
            # Keep eval lean during sweeps (honest protocol, smaller CV/controls).
            summary = run_learned_eval_suite(
                n_splits=3 if args.quick else 5,
                n_controls=10 if args.quick else 25,
            )
            row = {
                **cell,
                "status": "ok",
                "representation_label": manifest.get("representation_label"),
                "final_loss": manifest.get("final_loss"),
                "elapsed_s": manifest.get("elapsed_s"),
                "macro_f1_learned": summary["class_recovery"]
                .get(manifest.get("representation_label", ""), {})
                .get("macro_f1"),
                "macro_f1_hashed": summary["class_recovery"].get("hashed_architecture", {}).get("macro_f1"),
                "size_confound_learned": summary["novelty"]["size_confound_spearman_learned"],
                "size_confound_hashed": summary["novelty"]["size_confound_spearman_hashed"],
                "temporal_heldout_mean": summary["temporal"]["heldout_novelty_mean"],
                "temporal_control_mean": summary["temporal"]["control_mean"],
                "temporal_p": summary["temporal"]["p_value"],
                "temporal_win": summary["temporal"]["heldout_more_novel"],
            }
            # Prefer best learned macro-F1 among keys containing "learned"
            learned_f1s = [
                v["macro_f1"]
                for k, v in summary["class_recovery"].items()
                if "learned" in k and "combined" not in k
            ]
            if learned_f1s:
                row["macro_f1_learned"] = max(learned_f1s)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Sweep cell failed: %s", exc)
            row = {**cell, "status": "error", "error": str(exc)}
        results.append(row)
        # Incremental save
        out_path = REPORTS / "encoder_sweep_results.json"
        out_path.write_text(
            json.dumps({"n_runs": len(results), "results": results}, indent=2),
            encoding="utf-8",
        )

    _plot_sweep(results)
    LOG.info(
        "Sweep done: %d runs in %.0fs → %s",
        len(results),
        time.time() - t0,
        REPORTS / "encoder_sweep_results.json",
    )
    return 0


def _plot_sweep(results: list[dict]) -> None:
    ok = [r for r in results if r.get("status") == "ok" and r.get("macro_f1_learned") is not None]
    if not ok:
        return
    df = pd.DataFrame(ok)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # F1 by objective/pooling
    df["cfg"] = df["objective"] + "/" + df["pooling"]
    agg = df.groupby("cfg")["macro_f1_learned"].mean().sort_values()
    axes[0].barh(agg.index.astype(str), agg.values, color="#4C72B0")
    axes[0].set_xlabel("mean macro-F1 (learned)")
    axes[0].set_title("Class recovery by config")
    axes[0].set_xlim(0, 1)

    # Size confound
    agg2 = df.groupby("cfg")["size_confound_learned"].mean().sort_values()
    axes[1].barh(agg2.index.astype(str), agg2.values, color="#55A868")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("mean Spearman(novelty, n_genes)")
    axes[1].set_title("Size confound (lower |ρ| better)")

    # Temporal heldout - control
    df["temporal_delta"] = df["temporal_heldout_mean"] - df["temporal_control_mean"]
    agg3 = df.groupby("cfg")["temporal_delta"].mean().sort_values()
    axes[2].barh(agg3.index.astype(str), agg3.values, color="#C44E52")
    axes[2].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_xlabel("heldout − control novelty")
    axes[2].set_title("Prospective signal (want > 0)")

    fig.suptitle("Contrastive BGC encoder sweep", fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "encoder_sweep_summary.png", dpi=150)
    plt.close(fig)

    # Also write a CSV for easy inspection
    df.to_csv(REPORTS / "encoder_sweep_results.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
