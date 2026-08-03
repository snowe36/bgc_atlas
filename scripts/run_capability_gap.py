#!/usr/bin/env python3
"""CLI wrapper: capability-gap inference → family-level hypothesis cards."""

from __future__ import annotations

import sys

from bgcatlas.analyses.capability_gap.run import run_capability_gap_main

if __name__ == "__main__":
    raise SystemExit(run_capability_gap_main(sys.argv[1:]))
