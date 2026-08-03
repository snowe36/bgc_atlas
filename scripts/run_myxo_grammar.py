#!/usr/bin/env python3
"""CLI wrapper: decode myxobacterial architecture grammar."""

from __future__ import annotations

import sys

from bgcatlas.analyses.myxo_grammar.run import run_myxo_grammar_main

if __name__ == "__main__":
    raise SystemExit(run_myxo_grammar_main(sys.argv[1:]))
