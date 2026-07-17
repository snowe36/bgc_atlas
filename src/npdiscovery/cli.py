"""Console entry points: np-download, np-featurize, np-sanity, np-atlas, np-novelty, np-validate, np-apply."""

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
    from npdiscovery.data.curate import curate_mibig
    from npdiscovery.data.download import download_mibig

    if not args.skip_download:
        download_mibig()
    curate_mibig()
    return 0


def featurize_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build BGC architecture feature matrices")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from npdiscovery.featurize.run import run_featurize

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
    from npdiscovery.models.run import run_sanity

    run_sanity(n_splits=args.cv)
    return 0


def atlas_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map biosynthetic space (PCA/UMAP)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from npdiscovery.atlas.run import run_atlas

    run_atlas()
    return 0


def novelty_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score biosynthetic novelty and rank BGCs")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-k", type=int, default=5, help="Neighbors for kNN novelty")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from npdiscovery.novelty.run import run_novelty

    run_novelty(k=args.k)
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate novelty discovery strategy")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from npdiscovery.novelty.validate import run_validate

    run_validate()
    return 0


def apply_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score curated predicted BGCs against the MIBiG manifold"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from npdiscovery.data.apply import run_apply

    run_apply()
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
    }
    if cmd in dispatch:
        raise SystemExit(dispatch[cmd](rest))
    print(f"Usage: python -m npdiscovery.cli {{{'|'.join(dispatch)}}}", file=sys.stderr)
    raise SystemExit(2)
