"""Ingest antiSMASH outputs into the predicted-domain table schema used by apply.

Supported inputs:
  - directory of antiSMASH region GenBank files (`*.region*.gbk`, `*.gbk`)
  - antiSMASH JSON (`records[*].areas` product list + optional CDS annotations)
  - already-normalized domains CSV (genome, bgc_id, predicted_class, gene_order, domain_id, n_genes)

The GenBank path is preferred for domain richness; JSON is useful for region
inventory when full GBK isn't available.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from bgcatlas.data.curate import _coarsen_classes

LOG = logging.getLogger(__name__)

DOMAIN_COLS = [
    "genome",
    "bgc_id",
    "predicted_class",
    "gene_order",
    "domain_id",
    "n_genes",
]

_REGION_GBK_RE = re.compile(r"\.region\d+\.gbk$", re.IGNORECASE)


def _normalize_product_class(products: list[str]) -> str:
    """Map antiSMASH product labels onto the MIBiG coarse classes used elsewhere."""
    if not products:
        return "other"
    # antiSMASH often uses T1PKS / NRPS / terpene / RiPP-like labels
    mapped: list[str] = []
    for p in products:
        pl = p.lower()
        if "nrps" in pl:
            mapped.append("NRPS")
        elif "pks" in pl or "polyketide" in pl:
            mapped.append("PKS")
        elif "terpene" in pl:
            mapped.append("Terpene")
        elif "ripp" in pl or "lantipeptide" in pl or "lanthipeptide" in pl or "bacteriocin" in pl:
            mapped.append("RiPP")
        else:
            mapped.append(p)
    return _coarsen_classes(mapped)


def _domains_from_gbk_record(record, genome: str, bgc_id: str) -> list[dict]:
    products: list[str] = []
    for feat in record.features:
        if feat.type in {"region", "cluster", "protocluster", "cand_cluster"}:
            prods = feat.qualifiers.get("product") or feat.qualifiers.get("category") or []
            products.extend(str(p) for p in prods)

    predicted_class = _normalize_product_class(products)
    cds_features = [f for f in record.features if f.type == "CDS"]
    n_genes = len(cds_features) or 1

    rows: list[dict] = []
    order = 0

    # Prefer explicit domain features from antiSMASH
    domain_feats = [
        f
        for f in record.features
        if f.type in {"aSDomain", "PFAM_domain", "CDS_motif", "antismash_domain"}
    ]
    if domain_feats:
        for feat in sorted(domain_feats, key=lambda f: int(f.location.start)):
            dom = (
                (feat.qualifiers.get("domain") or feat.qualifiers.get("aSDomain") or feat.qualifiers.get("description") or [""])[0]
            )
            dom = str(dom).strip() or (
                (feat.qualifiers.get("label") or feat.qualifiers.get("note") or ["domain"])[0]
            )
            order += 1
            rows.append(
                {
                    "genome": genome,
                    "bgc_id": bgc_id,
                    "predicted_class": predicted_class,
                    "gene_order": order,
                    "domain_id": str(dom).split()[0][:80],
                    "n_genes": n_genes,
                }
            )
        return rows

    # Fallback: CDS products as architecture tokens (same spirit as MIBiG featurize)
    for feat in cds_features:
        prod = (feat.qualifiers.get("product") or ["hypothetical"])[0]
        # also harvest sec_met / gene_functions domain hints when present
        extras = []
        for key in ("sec_met_domain", "sec_met", "gene_functions", "domain"):
            extras.extend(feat.qualifiers.get(key) or [])
        tokens = [str(prod)] + [str(x).split(":")[-1].strip() for x in extras]
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            order += 1
            rows.append(
                {
                    "genome": genome,
                    "bgc_id": bgc_id,
                    "predicted_class": predicted_class,
                    "gene_order": order,
                    "domain_id": tok[:80],
                    "n_genes": n_genes,
                }
            )
    if not rows:
        rows.append(
            {
                "genome": genome,
                "bgc_id": bgc_id,
                "predicted_class": predicted_class,
                "gene_order": 1,
                "domain_id": "unknown",
                "n_genes": n_genes,
            }
        )
    return rows


def load_antismash_gbk(path: Path, genome: str | None = None) -> pd.DataFrame:
    """Parse one or many antiSMASH GenBank region files into a domains table."""
    files: list[Path]
    if path.is_dir():
        files = sorted(path.rglob("*.gbk")) + sorted(path.rglob("*.gbff"))
        # prefer region files when present
        region_files = [f for f in files if _REGION_GBK_RE.search(f.name)]
        if region_files:
            files = region_files
    else:
        files = [path]

    if not files:
        raise FileNotFoundError(f"No GenBank files found under {path}")

    genome_label = genome or path.stem
    rows: list[dict] = []
    for fpath in files:
        for i, record in enumerate(SeqIO.parse(str(fpath), "genbank"), start=1):
            bgc_id = fpath.stem if len(files) > 1 or i == 1 else f"{fpath.stem}_{i}"
            # sanitize id
            bgc_id = re.sub(r"[^A-Za-z0-9_.-]", "_", bgc_id)[:60]
            rows.extend(_domains_from_gbk_record(record, genome_label, bgc_id))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Parsed zero domains from {path}")
    LOG.info("antiSMASH GBK: %d domain tokens across %d BGCs from %s", len(df), df["bgc_id"].nunique(), path)
    return df[DOMAIN_COLS]


def load_antismash_json(path: Path, genome: str | None = None) -> pd.DataFrame:
    """Parse antiSMASH JSON areas into a coarse domains table (products as tokens)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    genome_label = genome or path.stem
    rows: list[dict] = []
    records = data.get("records") or []
    for ri, rec in enumerate(records):
        rec_name = str(rec.get("name") or rec.get("id") or f"record{ri}")
        areas = rec.get("areas") or rec.get("regions") or []
        for ai, area in enumerate(areas, start=1):
            products = [str(p) for p in (area.get("products") or [])]
            predicted_class = _normalize_product_class(products)
            bgc_id = f"{rec_name}_r{ai}"
            bgc_id = re.sub(r"[^A-Za-z0-9_.-]", "_", bgc_id)[:60]
            # Prefer nested CDS/domain annotations when exporters include them
            cds_list = area.get("cds") or area.get("orfs") or []
            order = 0
            if cds_list:
                for cds in cds_list:
                    for tok in _tokens_from_json_cds(cds):
                        order += 1
                        rows.append(
                            {
                                "genome": genome_label,
                                "bgc_id": bgc_id,
                                "predicted_class": predicted_class,
                                "gene_order": order,
                                "domain_id": tok[:80],
                                "n_genes": max(len(cds_list), 1),
                            }
                        )
            if order == 0:
                tokens = products or ["unknown"]
                for order, tok in enumerate(tokens, start=1):
                    rows.append(
                        {
                            "genome": genome_label,
                            "bgc_id": bgc_id,
                            "predicted_class": predicted_class,
                            "gene_order": order,
                            "domain_id": str(tok)[:80],
                            "n_genes": max(len(tokens), 1),
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Parsed zero regions/domains from JSON {path}")
    LOG.info("antiSMASH JSON: %d tokens across %d BGCs from %s", len(df), df["bgc_id"].nunique(), path)
    return df[DOMAIN_COLS]


def _tokens_from_json_cds(cds: dict) -> list[str]:
    tokens: list[str] = []
    for key in ("product", "gene_function", "function"):
        val = cds.get(key)
        if isinstance(val, str) and val:
            tokens.append(val)
        elif isinstance(val, list):
            tokens.extend(str(v) for v in val if v)
    for key in ("domains", "pfams", "sec_met_domains"):
        val = cds.get(key) or []
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    tokens.append(item)
                elif isinstance(item, dict):
                    tokens.append(str(item.get("name") or item.get("domain") or item.get("id") or ""))
    return [t for t in tokens if t]


def load_predicted_domains(path: Path, genome: str | None = None) -> pd.DataFrame:
    """Dispatch loader by file type / directory contents."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        missing = [c for c in DOMAIN_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Domains CSV missing columns {missing}; need {DOMAIN_COLS}")
        return df[DOMAIN_COLS]

    if path.is_file() and path.suffix.lower() == ".json":
        return load_antismash_json(path, genome=genome)

    if path.is_dir():
        json_files = list(path.glob("*.json"))
        gbk_files = list(path.rglob("*.gbk")) + list(path.rglob("*.gbff"))
        if gbk_files:
            return load_antismash_gbk(path, genome=genome)
        if len(json_files) == 1:
            return load_antismash_json(json_files[0], genome=genome)
        if json_files:
            frames = [load_antismash_json(j, genome=genome or j.stem) for j in json_files]
            return pd.concat(frames, ignore_index=True)[DOMAIN_COLS]
        raise FileNotFoundError(f"No antiSMASH .gbk/.json found under {path}")

    if path.is_file() and path.suffix.lower() in {".gbk", ".gbff"}:
        return load_antismash_gbk(path, genome=genome)

    raise FileNotFoundError(f"Unsupported antiSMASH / predicted input: {path}")
