"""Score predicted BGCs (demo CSV or antiSMASH GBK/JSON) against the MIBiG manifold."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from bgcatlas.config import PCA_N_COMPONENTS
from bgcatlas.data.antismash import load_predicted_domains
from bgcatlas.featurize.run import _hash_token
from bgcatlas.paths import DATA, PROCESSED, REPORTS, ensure_dirs

LOG = logging.getLogger(__name__)

EXTERNAL = DATA / "external"


def _ensure_curated_predicted() -> Path:
    """Write a small curated predicted-BGC domain table if missing."""
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL / "predicted_domains.csv"
    if path.exists():
        return path

    rows = []
    catalog = [
        {
            "genome": "Streptomyces_sp_predicted_1",
            "bgc_id": "PRED0001",
            "predicted_class": "NRPS",
            "domains": ["NRPS_module", "Condensation", "AMP-binding", "Thioesterase", "Transporter"],
            "n_genes": 12,
        },
        {
            "genome": "Streptomyces_sp_predicted_1",
            "bgc_id": "PRED0002",
            "predicted_class": "PKS",
            "domains": ["PKS_module", "PKS_module", "ACP", "Thioesterase", "P450", "Glycosyltransferase"],
            "n_genes": 18,
        },
        {
            "genome": "Micromonospora_sp_predicted",
            "bgc_id": "PRED0003",
            "predicted_class": "hybrid",
            "domains": ["PKS_module", "NRPS_module", "Condensation", "ACP", "Methyltransferase"],
            "n_genes": 22,
        },
        {
            "genome": "Bacillus_sp_predicted",
            "bgc_id": "PRED0004",
            "predicted_class": "RiPP",
            "domains": ["RiPP_lanthipeptide", "RiPP_other", "Transporter", "Regulator"],
            "n_genes": 8,
        },
        {
            "genome": "Pseudomonas_sp_predicted",
            "bgc_id": "PRED0005",
            "predicted_class": "terpene",
            "domains": ["Terpene_synth", "P450", "Redox", "Regulator"],
            "n_genes": 6,
        },
        {
            "genome": "Rare_actinobacterium_predicted",
            "bgc_id": "PRED0006",
            "predicted_class": "other",
            "domains": ["Halogenase", "P450", "Redox", "Hydrolase", "Kinase", "Hypothetical", "Hypothetical"],
            "n_genes": 15,
        },
        {
            "genome": "Rare_actinobacterium_predicted",
            "bgc_id": "PRED0007",
            "predicted_class": "NRPS",
            "domains": [
                "NRPS_module",
                "NRPS_module",
                "Condensation",
                "Halogenase",
                "P450",
                "Glycosyltransferase",
                "Transporter",
                "Regulator",
            ],
            "n_genes": 28,
        },
        {
            "genome": "Myxococcus_sp_predicted",
            "bgc_id": "PRED0008",
            "predicted_class": "hybrid",
            "domains": ["PKS_module", "PKS_module", "NRPS_module", "Methyltransferase", "Halogenase", "P450"],
            "n_genes": 35,
        },
    ]
    for entry in catalog:
        for order, dom in enumerate(entry["domains"], start=1):
            rows.append(
                {
                    "genome": entry["genome"],
                    "bgc_id": entry["bgc_id"],
                    "predicted_class": entry["predicted_class"],
                    "gene_order": order,
                    "domain_id": dom,
                    "n_genes": entry["n_genes"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    meta = (
        df.groupby(["genome", "bgc_id", "predicted_class", "n_genes"], as_index=False)
        .size()
        .rename(columns={"size": "n_domain_tokens"})
    )
    meta.to_csv(EXTERNAL / "predicted_bgcs.csv", index=False)
    LOG.info("Wrote curated predicted BGCs → %s", path)
    return path


def _vectorize_predicted(
    pred_domains: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build feature matrix aligned to MIBiG feature vocabulary."""
    meta = pred_domains.groupby(["genome", "bgc_id", "predicted_class"], as_index=False).agg(
        n_genes=("n_genes", "first"), n_domain_annotations=("domain_id", "count")
    )
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    X = np.zeros((len(meta), len(feature_names)), dtype=np.float32)

    ordered = pred_domains.sort_values(["bgc_id", "gene_order"]).groupby("bgc_id")["domain_id"].apply(list)
    for i, row in meta.iterrows():
        bgc_id = row["bgc_id"]
        seq = ordered.get(bgc_id, [])
        for tok, cnt in pd.Series(seq).value_counts().items():
            key = f"dom::{tok}"
            if key in name_to_idx:
                X[i, name_to_idx[key]] = float(cnt)
        for tok in seq:
            h = _hash_token(f"uni::{tok}")
            key = f"arch_hash::{h}"
            if key in name_to_idx:
                X[i, name_to_idx[key]] += 1.0
        for a, b in zip(seq, seq[1:], strict=False):
            h = _hash_token(f"bi::{a}::{b}")
            key = f"arch_hash::{h}"
            if key in name_to_idx:
                X[i, name_to_idx[key]] += 1.0
        for col, val in [
            ("size::n_genes", row["n_genes"]),
            ("size::n_domain_annotations", row["n_domain_annotations"]),
            ("size::cluster_nt_length", row["n_genes"] * 1000),
            ("size::mean_aa_length", 350.0),
            ("size::total_aa_length", row["n_genes"] * 350),
            ("size::n_compounds", 0),
        ]:
            if col in name_to_idx:
                X[i, name_to_idx[col]] = float(val)
    return meta.reset_index(drop=True), X


def run_apply(
    k: int = 5,
    input_path: str | Path | None = None,
    genome: str | None = None,
) -> pd.DataFrame:
    """Score predicted BGCs against MIBiG.

    input_path:
      None / missing → curated demo CSV under data/external/
      .csv           → domains table
      .json / .gbk   → antiSMASH output
      directory      → antiSMASH region GBKs or JSON
    """
    ensure_dirs()
    if input_path is None:
        pred_path = _ensure_curated_predicted()
        pred_domains = pd.read_csv(pred_path)
        source = "demo_csv"
    else:
        pred_path = Path(input_path)
        pred_domains = load_predicted_domains(pred_path, genome=genome)
        source = str(pred_path)
        # cache normalized table for inspection
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        cache = EXTERNAL / "last_apply_domains.csv"
        pred_domains.to_csv(cache, index=False)
        LOG.info("Cached normalized domains → %s", cache)

    feature_names = pd.read_csv(PROCESSED / "feature_names.csv")["feature"].tolist()
    X_ref = np.load(PROCESSED / "feature_matrix.npy")
    meta_ref = pd.read_parquet(PROCESSED / "feature_meta.parquet")

    meta_pred, X_pred = _vectorize_predicted(pred_domains, feature_names)

    scaler = StandardScaler(with_mean=False)
    Xs_ref = scaler.fit_transform(X_ref)
    Xs_pred = scaler.transform(X_pred)

    n_comp = min(PCA_N_COMPONENTS, Xs_ref.shape[0] - 1, Xs_ref.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    Z_ref = pca.fit_transform(Xs_ref)
    Z_pred = pca.transform(Xs_pred)

    nn = NearestNeighbors(n_neighbors=min(k, len(Z_ref)), metric="euclidean")
    nn.fit(Z_ref)
    dists, idxs = nn.kneighbors(Z_pred)
    knn_mean = dists.mean(axis=1)
    nearest_idx = idxs[:, 0]
    nearest_dist = dists[:, 0]

    ref_nn = NearestNeighbors(n_neighbors=min(k + 1, len(Z_ref)), metric="euclidean")
    ref_nn.fit(Z_ref)
    ref_dists, _ = ref_nn.kneighbors(Z_ref)
    ref_knn = ref_dists[:, 1:].mean(axis=1)
    novelty = np.array([(ref_knn < d).mean() for d in knn_mean], dtype=float)

    out = meta_pred.copy()
    out["novelty"] = novelty
    out["knn_mean_dist"] = knn_mean
    out["nearest_dist"] = nearest_dist
    out["nearest_mibig"] = meta_ref["bgc_id"].iloc[nearest_idx].to_numpy()
    out["neighbor_class"] = meta_ref["biosynth_class"].iloc[nearest_idx].to_numpy()
    out["source"] = source
    out = out.sort_values("novelty", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / "predicted_novelty_ranking.csv"
    out.to_csv(out_path, index=False)
    LOG.info(
        "Predicted BGC novelty ranking (%d BGCs from %s):\n%s",
        len(out),
        source,
        out.head(20).to_string(index=False),
    )
    LOG.info("Wrote %s", out_path)
    return out
