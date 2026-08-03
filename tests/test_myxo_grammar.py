"""Smoke tests for the myxo grammar decode (uses committed ranking)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bgcatlas.analyses.myxo_grammar.cohort import phenotype_gate
from bgcatlas.analyses.myxo_grammar.tokens import extract_tokens_for_bgc
from bgcatlas.paths import REPORTS

RANKING = REPORTS / "novelty_ranking.csv"


@pytest.mark.skipif(not RANKING.exists(), reason="novelty_ranking.csv not committed")
def test_phenotype_gate_reproduces_size_controlled_or():
    ranking = pd.read_csv(RANKING)
    gate = phenotype_gate(ranking, size_max=60)
    assert gate["n_myxo"] == 84
    assert gate["gate_ok"] is True
    assert 2.0 <= gate["odds_ratio_top_decile"] <= 5.0
    lo, hi = gate["odds_ratio_top_decile_ci95"]
    assert lo < gate["odds_ratio_top_decile"] < hi


def test_extract_tokens_includes_capacity_and_grammar():
    seq = ["PKS_KS", "PKS_AT", "PKS_DH", "PKS_KR", "Methyltransferase", "Glycosyltransferase"]
    toks = extract_tokens_for_bgc(seq)
    assert "dom::PKS_KS" in toks
    assert "capacity::PKS_KS" in toks
    assert "capacity::glycosyltransferase" in toks
    assert "grammar::PKS+GT" in toks
    assert "motif2::PKS_KS-PKS_AT" in toks
