"""Console entry points: bgc-download, bgc-featurize, bgc-sanity, bgc-atlas, bgc-novelty, bgc-validate, bgc-apply."""

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
    parser = argparse.ArgumentParser(
        description="Benchmark representations via biosynth-class recovery"
    )
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
    parser = argparse.ArgumentParser(
        description="Score curated predicted BGCs against the MIBiG manifold"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.data.apply import run_apply

    run_apply()
    return 0


def temporal_main(argv: list[str] | None = None) -> int:
    from bgcatlas.config import DEFAULT_NOVELTY_K, DEFAULT_TEMPORAL_CUTOFF

    parser = argparse.ArgumentParser(
        description="Prospective validation: fit on pre-cutoff MIBiG entries, "
        "score post-cutoff entries as a held-out temporal novelty test"
    )
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
        description="Compare hashed-architecture vs ESM2 vs combined representations"
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
        description="Compare novelty rankings across hashed/ESM2/combined representations"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-k", type=int, default=DEFAULT_NOVELTY_K)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from bgcatlas.novelty.embed_compare import run_novelty_representation_comparison

    run_novelty_representation_comparison(k=args.k)
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
    }
    if cmd in dispatch:
        raise SystemExit(dispatch[cmd](rest))
    print(f"Usage: python -m bgcatlas.cli {{{'|'.join(dispatch)}}}", file=sys.stderr)
    raise SystemExit(2)
