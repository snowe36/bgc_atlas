"""Homolog90 clustering (UniRef90-style) for candidate identity — Level 2.

Expectation / gaps / calibration must never import this module.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from bgcatlas.paths import PROCESSED

LOG = logging.getLogger(__name__)

HOMOLOG_IDENTITY = 0.90  # fixed — no threshold sweeps
CLUSTER_MAP = PROCESSED / "protein_homolog_clusters.csv"
PROTEINS = PROCESSED / "mibig_proteins.parquet"
REPRESENTATION = "homolog90"


def require_mmseqs() -> str:
    path = shutil.which("mmseqs")
    if not path:
        raise RuntimeError(
            "mmseqs not found on PATH. Install MMseqs2 "
            "(e.g. `brew install mmseqs2` or conda) for homolog90 clustering."
        )
    return path


def protein_id(bgc_id: str, locus_tag: str) -> str:
    """Stable FASTA header / join key (no spaces)."""
    return f"{bgc_id}__{locus_tag}"


def parse_protein_id(pid: str) -> tuple[str, str]:
    bgc_id, _, locus = pid.partition("__")
    return bgc_id, locus


def load_proteins(path: Path | None = None) -> pd.DataFrame:
    p = path or PROTEINS
    if not p.exists():
        raise FileNotFoundError(f"Missing protein table: {p}")
    df = pd.read_parquet(p)
    need = {"bgc_id", "locus_tag", "translation"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing columns: {sorted(missing)}")
    df = df.copy()
    df["bgc_id"] = df["bgc_id"].astype(str)
    df["locus_tag"] = df["locus_tag"].astype(str)
    df["translation"] = df["translation"].astype(str)
    df = df[df["translation"].str.len() >= 20]
    df = df[df["locus_tag"].str.len() > 0]
    df = df.drop_duplicates(subset=["bgc_id", "locus_tag"], keep="first")
    return df.reset_index(drop=True)


def write_fasta(proteins: pd.DataFrame, fasta_path: Path) -> int:
    n = 0
    with fasta_path.open("w", encoding="utf-8") as fh:
        for r in proteins.itertuples(index=False):
            seq = "".join(str(r.translation).split())
            if len(seq) < 20:
                continue
            pid = protein_id(str(r.bgc_id), str(r.locus_tag))
            fh.write(f">{pid}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")
            n += 1
    return n


def _run_mmseqs_cluster(fasta: Path, out_dir: Path, *, min_seq_id: float) -> Path:
    mmseqs = require_mmseqs()
    db = out_dir / "seqDB"
    clu = out_dir / "clu"
    tsv = out_dir / "clusters.tsv"
    tmp = out_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [mmseqs, "createdb", str(fasta), str(db)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            mmseqs,
            "easy-cluster",
            str(fasta),
            str(clu),
            str(tmp),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            "0.8",
            "--cov-mode",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # easy-cluster writes <prefix>_cluster.tsv
    cluster_tsv = Path(str(clu) + "_cluster.tsv")
    if not cluster_tsv.exists():
        # fallback: createtsv from clu DB
        subprocess.run(
            [mmseqs, "createtsv", str(db), str(db), str(clu), str(tsv)],
            check=True,
            capture_output=True,
            text=True,
        )
        return tsv
    return cluster_tsv


def _parse_cluster_tsv(tsv: Path) -> pd.DataFrame:
    """MMseqs cluster TSV: rep_id\\tmember_id."""
    rows = []
    with tsv.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            rows.append({"rep_id": rep, "member_id": member})
    if not rows:
        return pd.DataFrame(
            columns=[
                "bgc_id",
                "locus_tag",
                "cluster_id",
                "family_key",
                "n_members",
                "rep_locus",
                "rep_bgc_id",
            ]
        )
    raw = pd.DataFrame(rows)
    # Assign dense cluster indices by representative
    reps = sorted(raw["rep_id"].unique())
    rep_to_clu = {r: f"clu_{i:05d}" for i, r in enumerate(reps)}
    raw["cluster_id"] = raw["rep_id"].map(rep_to_clu)
    sizes = raw.groupby("cluster_id")["member_id"].transform("count")
    raw["n_members"] = sizes.astype(int)

    out_rows = []
    for r in raw.itertuples(index=False):
        bgc_id, locus = parse_protein_id(str(r.member_id))
        rep_bgc, rep_locus = parse_protein_id(str(r.rep_id))
        cid = str(r.cluster_id)
        out_rows.append(
            {
                "bgc_id": bgc_id,
                "locus_tag": locus,
                "cluster_id": cid,
                "family_key": f"{REPRESENTATION}__{cid}",
                "n_members": int(r.n_members),
                "rep_locus": rep_locus,
                "rep_bgc_id": rep_bgc,
            }
        )
    return pd.DataFrame(out_rows)


def build_homolog_clusters(
    *,
    proteins_path: Path | None = None,
    out_path: Path | None = None,
    min_seq_id: float = HOMOLOG_IDENTITY,
    force: bool = False,
) -> pd.DataFrame:
    """Cluster MIBiG proteins at fixed identity; write cache CSV."""
    out = out_path or CLUSTER_MAP
    if out.exists() and not force:
        LOG.info("Loading cached homolog clusters from %s", out)
        return pd.read_csv(out)

    proteins = load_proteins(proteins_path)
    require_mmseqs()
    LOG.info(
        "Clustering %d proteins at min-seq-id=%.2f (homolog90)",
        len(proteins),
        min_seq_id,
    )
    with tempfile.TemporaryDirectory(prefix="bgc_homolog_") as td:
        td_path = Path(td)
        fasta = td_path / "proteins.faa"
        n = write_fasta(proteins, fasta)
        LOG.info("Wrote %d FASTA records", n)
        tsv = _run_mmseqs_cluster(fasta, td_path / "mmseqs", min_seq_id=min_seq_id)
        clusters = _parse_cluster_tsv(tsv)

    out.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(out, index=False)
    LOG.info(
        "Wrote %s (%d members, %d clusters)",
        out,
        len(clusters),
        clusters["cluster_id"].nunique() if len(clusters) else 0,
    )
    return clusters


def load_or_build_clusters(*, rebuild: bool = False) -> pd.DataFrame:
    return build_homolog_clusters(force=rebuild)
