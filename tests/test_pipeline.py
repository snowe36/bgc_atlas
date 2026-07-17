"""Unit tests for parse / featurize / novelty on tiny fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from npdiscovery.data.curate import _coarsen_classes, _parse_one_json
from npdiscovery.featurize.run import build_feature_matrix
from npdiscovery.novelty.run import score_novelty


FIXTURES = Path(__file__).parent / "fixtures"


def test_coarsen_classes_hybrid():
    assert _coarsen_classes(["PKS", "NRPS"]) == "hybrid"
    assert _coarsen_classes(["Terpene"]) == "terpene"
    assert _coarsen_classes(["RiPP"]) == "RiPP"


def test_parse_fixture_json(tmp_path):
    payload = {
        "accession": "BGC9999999",
        "status": "active",
        "biosynthesis": {"classes": [{"class": "NRPS", "subclass": "Unknown"}]},
        "taxonomy": {"name": "Test organism", "ncbiTaxId": 1},
        "compounds": [{"name": "testomycin"}],
        "genes": {"annotations": [{"id": "g1"}, {"id": "g2"}]},
    }
    path = tmp_path / "BGC9999999.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    row = _parse_one_json(path)
    assert row is not None
    assert row["bgc_id"] == "BGC9999999"
    assert row["biosynth_class"] == "NRPS"
    assert row["organism"] == "Test organism"


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
            {"bgc_id": "BGC_A", "feature_type": "domain", "gene_order": 1, "domain_id": "NRPS_module", "locus_tag": "a", "product": "", "aa_length": 0},
            {"bgc_id": "BGC_A", "feature_type": "domain", "gene_order": 2, "domain_id": "Condensation", "locus_tag": "a", "product": "", "aa_length": 0},
            {"bgc_id": "BGC_B", "feature_type": "domain", "gene_order": 1, "domain_id": "PKS_module", "locus_tag": "b", "product": "", "aa_length": 0},
            {"bgc_id": "BGC_B", "feature_type": "domain", "gene_order": 2, "domain_id": "PKS_module", "locus_tag": "b", "product": "", "aa_length": 0},
            {"bgc_id": "BGC_C", "feature_type": "domain", "gene_order": 1, "domain_id": "Terpene_synth", "locus_tag": "c", "product": "", "aa_length": 0},
        ]
    )
    meta, X, names = build_feature_matrix(bgcs, domains)
    assert len(meta) == 3
    assert X.shape[0] == 3
    assert X.shape[1] == len(names)
    assert not any("biosynth_class" in n for n in names)


def test_novelty_ranks_outlier():
    # three tight points + one far outlier
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
