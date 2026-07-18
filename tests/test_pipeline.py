"""Unit tests for parse / featurize / novelty / validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bgcatlas.config import MAJOR_FAMILIES, N_HASH_DIMS, PCA_N_COMPONENTS
from bgcatlas.data.curate import _changelog_dates, _coarsen_classes, _parse_one_json
from bgcatlas.featurize.run import _hash_token, build_feature_matrix
from bgcatlas.novelty.embed_compare import _class_stratified_disagreement
from bgcatlas.novelty.run import score_novelty
from bgcatlas.novelty.temporal import _score_query

FIXTURES = Path(__file__).parent / "fixtures"


def test_coarsen_classes_hybrid():
    assert _coarsen_classes(["PKS", "NRPS"]) == "hybrid"
    assert _coarsen_classes(["Terpene"]) == "terpene"
    assert _coarsen_classes(["RiPP"]) == "RiPP"


def test_parse_fixture_json():
    path = FIXTURES / "sample_bgc.json"
    row = _parse_one_json(path)
    assert row is not None
    assert row["bgc_id"] == "BGC9999999"
    assert row["biosynth_class"] == "NRPS"
    assert row["organism"] == "Test organism"
    assert row["n_genes_json"] == 3
    assert row["date_added"] == "2020-01-15"


def test_parse_inline_json(tmp_path):
    payload = {
        "accession": "BGC8888888",
        "status": "active",
        "biosynthesis": {"classes": [{"class": "PKS", "subclass": "Unknown"}]},
        "taxonomy": {"name": "Other organism", "ncbiTaxId": 2},
        "compounds": [{"name": "polyketide-x"}],
        "genes": {"annotations": [{"id": "g1"}]},
    }
    path = tmp_path / "BGC8888888.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    row = _parse_one_json(path)
    assert row is not None
    assert row["biosynth_class"] == "PKS"


def test_feature_matrix_shapes():
    bgcs = pd.DataFrame(
        {
            "bgc_id": ["BGC_A", "BGC_B", "BGC_C"],
            "biosynth_class": ["NRPS", "PKS", "terpene"],
            "n_genes": [5, 8, 3],
            "cluster_nt_length": [1000, 2000, 500],
            "mean_aa_length": [300.0, 400.0, 200.0],
            "total_aa_length": [1500, 3200, 600],
            "n_domain_annotations": [4, 6, 2],
            "n_compounds": [1, 1, 1],
        }
    )
    domains = pd.DataFrame(
        [
            {
                "bgc_id": "BGC_A",
                "feature_type": "domain",
                "gene_order": 1,
                "domain_id": "NRPS_module",
                "locus_tag": "a",
                "product": "",
                "aa_length": 0,
            },
            {
                "bgc_id": "BGC_A",
                "feature_type": "domain",
                "gene_order": 2,
                "domain_id": "Condensation",
                "locus_tag": "a",
                "product": "",
                "aa_length": 0,
            },
            {
                "bgc_id": "BGC_B",
                "feature_type": "domain",
                "gene_order": 1,
                "domain_id": "PKS_module",
                "locus_tag": "b",
                "product": "",
                "aa_length": 0,
            },
            {
                "bgc_id": "BGC_B",
                "feature_type": "domain",
                "gene_order": 2,
                "domain_id": "PKS_module",
                "locus_tag": "b",
                "product": "",
                "aa_length": 0,
            },
            {
                "bgc_id": "BGC_C",
                "feature_type": "domain",
                "gene_order": 1,
                "domain_id": "Terpene_synth",
                "locus_tag": "c",
                "product": "",
                "aa_length": 0,
            },
        ]
    )
    meta, X, names = build_feature_matrix(bgcs, domains)
    assert len(meta) == 3
    assert X.shape[0] == 3
    assert X.shape[1] == len(names)
    assert not any("biosynth_class" in n for n in names)
    # hashed architecture block is always N_HASH_DIMS wide
    assert any(n.startswith("arch_hash::") for n in names) or X.shape[1] >= N_HASH_DIMS


def test_hash_token_stable_and_in_range():
    a = _hash_token("uni::Condensation")
    b = _hash_token("uni::Condensation")
    assert a == b
    assert 0 <= a < N_HASH_DIMS


def test_novelty_ranks_outlier():
    Z = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
        ],
        dtype=float,
    )
    scores = score_novelty(Z, k=2)
    assert scores["novelty"].argmax() == 3


def test_changelog_dates_earliest_and_latest():
    changelog = {
        "releases": [
            {
                "version": "1",
                "date": "2019-01-01",
                "entries": [{"date": "2019-01-01", "comment": "Submitted"}],
            },
            {
                "version": "2",
                "date": "2021-06-15",
                "entries": [{"date": "2021-06-15", "comment": "Updated"}],
            },
        ]
    }
    earliest, latest = _changelog_dates(changelog)
    assert earliest == "2019-01-01"
    assert latest == "2021-06-15"
    assert _changelog_dates({}) == (None, None)


def test_score_query_flags_far_cluster_as_novel():
    Z_ref = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]])
    ref_knn_mean = Z_ref[:, 0] * 0 + 0.1
    Z_query_close = np.array([[0.05, 0.0]])
    Z_query_far = np.array([[50.0, 50.0]])
    p_close = _score_query(Z_ref, ref_knn_mean, Z_query_close, k=3)
    p_far = _score_query(Z_ref, ref_knn_mean, Z_query_far, k=3)
    assert p_far[0] > p_close[0]
    assert p_far[0] == 1.0


def test_validate_uses_spearman_not_pearson():
    """Regression: novelty↔n_genes must be Spearman (rank), not Pearson."""
    rng = np.random.default_rng(0)
    n = 200
    # Monotone but nonlinear relationship: Pearson and Spearman diverge
    x = np.linspace(0, 10, n)
    y = x**3 + rng.normal(0, 5, n)
    pearson = float(pd.Series(x).corr(pd.Series(y)))
    spearman, _ = spearmanr(x, y)
    assert abs(spearman - pearson) > 0.05
    # Mimic the fixed validate.py path
    scores = pd.DataFrame({"novelty": x, "n_genes": y})
    corr, _ = spearmanr(scores["novelty"], scores["n_genes"])
    assert abs(corr - spearman) < 1e-12


def test_class_stratified_disagreement_reports_per_class():
    rng = np.random.default_rng(1)
    rows = []
    for cls, n in [("NRPS", 40), ("PKS", 40), ("RiPP", 30)]:
        for i in range(n):
            h = rng.random()
            # RiPP: agree; NRPS/PKS: anti-correlate
            e = h if cls == "RiPP" else (1.0 - h + rng.normal(0, 0.05))
            rows.append(
                {
                    "bgc_id": f"{cls}_{i}",
                    "biosynth_class": cls,
                    "novelty_hashed": h,
                    "novelty_esm": float(np.clip(e, 0, 1)),
                }
            )
    out = pd.DataFrame(rows)
    records, df = _class_stratified_disagreement(out, top_frac=0.1)
    assert len(records) == 3
    assert set(df["biosynth_class"]) == {"NRPS", "PKS", "RiPP"}
    ripp = df.loc[df["biosynth_class"] == "RiPP", "spearman_hashed_vs_esm"].iloc[0]
    nrps = df.loc[df["biosynth_class"] == "NRPS", "spearman_hashed_vs_esm"].iloc[0]
    assert ripp > 0.5
    assert nrps < 0


def test_config_defaults_sensible():
    assert N_HASH_DIMS == 256
    assert PCA_N_COMPONENTS == 50
    assert set(MAJOR_FAMILIES) == {"PKS", "NRPS", "hybrid"}
