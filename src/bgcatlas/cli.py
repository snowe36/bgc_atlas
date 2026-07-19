"""CLI entry points for the bgc-* console scripts."""

from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def download_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and parse MIBiG BGC data")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Parse existing raw files without re-downloading",
    )
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.data.curate import curate_mibig
    from bgcatlas.data.download import download_mibig

    if not args.skip_download:
        download_mibig()
    curate_mibig()
    return 0


def featurize_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build BGC architecture feature matrices")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.featurize.run import run_featurize

    run_featurize()
    return 0


def sanity_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark representations via biosynth-class recovery")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.models.run import run_sanity

    run_sanity(n_splits=args.cv)
    return 0


def atlas_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map biosynthetic space (PCA/UMAP)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.atlas.run import run_atlas

    run_atlas()
    return 0


def novelty_main(argv: list[str] | None = None) -> int:
    from bgcatlas.config import DEFAULT_NOVELTY_K

    parser = argparse.ArgumentParser(description="Score biosynthetic novelty and rank BGCs")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-k", type=int, default=DEFAULT_NOVELTY_K, help="Neighbors for kNN novelty")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.novelty.run import run_novelty

    run_novelty(k=args.k)
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate novelty discovery strategy")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.novelty.validate import run_validate

    run_validate()
    return 0


def apply_main(argv: list[str] | None = None) -> int:
    from bgcatlas.config import DEFAULT_NOVELTY_K

    parser = argparse.ArgumentParser(description="Score predicted BGCs against MIBiG")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--input",
        default=None,
        help="antiSMASH .gbk/.json, directory of region GBKs, or domains CSV "
        "(default: curated demo under data/external/)",
    )
    parser.add_argument("--genome", default=None, help="Optional genome label for antiSMASH ingest")
    parser.add_argument("-k", type=int, default=DEFAULT_NOVELTY_K, help="Neighbors for kNN novelty")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.data.apply import run_apply

    run_apply(k=args.k, input_path=args.input, genome=args.genome)
    return 0


def temporal_main(argv: list[str] | None = None) -> int:
    from bgcatlas.config import DEFAULT_NOVELTY_K, DEFAULT_TEMPORAL_CUTOFF

    parser = argparse.ArgumentParser(description="Temporal holdout novelty test")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--cutoff",
        default=DEFAULT_TEMPORAL_CUTOFF,
        help="ISO date; entries added on/after this are held out",
    )
    parser.add_argument("-k", type=int, default=DEFAULT_NOVELTY_K, help="Neighbors for kNN novelty")
    parser.add_argument("--n-controls", type=int, default=50, help="Random-holdout control repeats")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.novelty.temporal import run_temporal_holdout

    run_temporal_holdout(cutoff=args.cutoff, k=args.k, n_controls=args.n_controls)
    return 0


def ablation_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Class recovery: hashed vs ESM2 vs combined"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.models.ablation import run_ablation

    run_ablation(n_splits=args.cv)
    return 0


def novelty_compare_main(argv: list[str] | None = None) -> int:
    from bgcatlas.config import DEFAULT_NOVELTY_K

    parser = argparse.ArgumentParser(
        description="Compare novelty rankings across representations"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-k", type=int, default=DEFAULT_NOVELTY_K)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.novelty.embed_compare import run_novelty_representation_comparison

    run_novelty_representation_comparison(k=args.k)
    return 0


def train_encoder_main(argv: list[str] | None = None) -> int:
    from bgcatlas.config import (
        DEFAULT_ENCODER_BATCH_SIZE,
        DEFAULT_ENCODER_EMBED_DIM,
        DEFAULT_ENCODER_EPOCHS,
        DEFAULT_ENCODER_FEAT_DROPOUT,
        DEFAULT_ENCODER_HIDDEN,
        DEFAULT_ENCODER_KEEP_FRAC,
        DEFAULT_ENCODER_LR,
        DEFAULT_ENCODER_OBJECTIVE,
        DEFAULT_ENCODER_POOLING,
        DEFAULT_ENCODER_SEED,
        DEFAULT_ENCODER_TEMPERATURE,
        DEFAULT_TEMPORAL_CUTOFF,
    )

    parser = argparse.ArgumentParser(
        description="Train contrastive BGC set-encoder on cached ESM2 proteins"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--objective",
        default=DEFAULT_ENCODER_OBJECTIVE,
        choices=["simclr", "supcon"],
        help="Contrastive objective",
    )
    parser.add_argument(
        "--pooling",
        default=DEFAULT_ENCODER_POOLING,
        choices=["attention", "mean", "deepsets"],
    )
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_ENCODER_EMBED_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_ENCODER_HIDDEN)
    parser.add_argument("--epochs", type=int, default=DEFAULT_ENCODER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ENCODER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_ENCODER_LR)
    parser.add_argument("--temperature", type=float, default=DEFAULT_ENCODER_TEMPERATURE)
    parser.add_argument("--keep-frac", type=float, default=DEFAULT_ENCODER_KEEP_FRAC)
    parser.add_argument("--feat-dropout", type=float, default=DEFAULT_ENCODER_FEAT_DROPOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_ENCODER_SEED)
    parser.add_argument(
        "--prospective",
        action="store_true",
        help=f"Train only on BGCs with date_added < {DEFAULT_TEMPORAL_CUTOFF}",
    )
    parser.add_argument(
        "--train-cutoff",
        default=None,
        help="ISO date; train only on date_added < cutoff (overrides --prospective default)",
    )
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto if omitted)")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.models.train_contrastive import run_train_from_args

    run_train_from_args(
        objective=args.objective,
        pooling=args.pooling,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        keep_frac=args.keep_frac,
        feat_dropout=args.feat_dropout,
        seed=args.seed,
        train_cutoff=args.train_cutoff,
        prospective=args.prospective,
        device=args.device,
    )
    return 0


def learned_eval_main(argv: list[str] | None = None) -> int:
    from bgcatlas.config import DEFAULT_NOVELTY_K, DEFAULT_TEMPORAL_CUTOFF

    parser = argparse.ArgumentParser(
        description="Eval learned embeddings (class recovery, novelty, temporal)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("-k", type=int, default=DEFAULT_NOVELTY_K)
    parser.add_argument("--cutoff", default=DEFAULT_TEMPORAL_CUTOFF)
    parser.add_argument("--n-controls", type=int, default=50)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.novelty.learned_eval import run_learned_eval_suite

    run_learned_eval_suite(n_splits=args.cv, k=args.k, cutoff=args.cutoff, n_controls=args.n_controls)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    dispatch = {
        "download": download_main,
        "featurize": featurize_main,
        "sanity": sanity_main,
        "atlas": atlas_main,
        "novelty": novelty_main,
        "validate": validate_main,
        "apply": apply_main,
        "temporal": temporal_main,
        "ablation": ablation_main,
        "novelty-compare": novelty_compare_main,
        "train-encoder": train_encoder_main,
        "learned-eval": learned_eval_main,
    }
    if cmd in dispatch:
        raise SystemExit(dispatch[cmd](rest))
    print(f"Usage: python -m bgcatlas.cli {{{'|'.join(dispatch)}}}", file=sys.stderr)
    raise SystemExit(2)
