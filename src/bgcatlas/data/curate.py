"""Parse MIBiG JSON + GenBank into tidy BGC / domain tables."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO

from bgcatlas.paths import PROCESSED, RAW, ensure_dirs

LOG = logging.getLogger(__name__)

CLASS_ALIASES = {
    "nrps": "NRPS",
    "nrp": "NRPS",
    "pks": "PKS",
    "polyketide": "PKS",
    "terpene": "terpene",
    "ripp": "RiPP",
    "ribosomal": "RiPP",
    "alkaloid": "other",
    "saccharide": "other",
    "other": "other",
}


def _find_json_dir() -> Path:
    for base in (RAW / "mibig_json_4.0", RAW / "mibig_json"):
        if base.exists():
            return base
    for p in RAW.rglob("BGC*.json"):
        return p.parent
    raise FileNotFoundError("No MIBiG JSON directory found under data/raw/")


def _find_gbk_dir() -> Path:
    for base in (RAW / "mibig_gbk_4.0", RAW / "mibig_gbk"):
        if base.exists():
            return base
    for p in RAW.rglob("BGC*.gbk"):
        return p.parent
    raise FileNotFoundError("No MIBiG GenBank directory found under data/raw/")


def _coarsen_classes(classes: list[str]) -> str:
    mapped: list[str] = []
    for c in classes:
        key = c.strip().lower()
        hit = "other"
        for pattern, label in CLASS_ALIASES.items():
            if pattern == key or pattern in key:
                hit = label
                break
        mapped.append(hit)
    uniq = sorted(set(mapped))
    if not uniq:
        return "other"
    if len(uniq) == 1:
        return uniq[0]
    majors = [u for u in uniq if u in {"NRPS", "PKS", "terpene", "RiPP"}]
    if len(majors) >= 2:
        return "hybrid"
    return majors[0] if majors else "other"


def _parse_one_json(path: Path) -> dict[str, Any] | None:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    bgc_id = data.get("accession") or path.stem.split(".")[0]
    m = re.search(r"(BGC\d+)", str(bgc_id))
    if m:
        bgc_id = m.group(1)

    biosynthesis = data.get("biosynthesis") or {}
    raw_classes = []
    for entry in biosynthesis.get("classes") or []:
        if isinstance(entry, dict):
            raw_classes.append(str(entry.get("class") or entry.get("subclass") or ""))
        else:
            raw_classes.append(str(entry))
    raw_classes = [c for c in raw_classes if c]

    taxonomy = data.get("taxonomy") or {}
    organism = taxonomy.get("name") if isinstance(taxonomy, dict) else "unknown"
    tax_id = taxonomy.get("ncbiTaxId") if isinstance(taxonomy, dict) else None

    compounds = data.get("compounds") or []
    compound_names = []
    for c in compounds:
        if isinstance(c, dict):
            compound_names.append(str(c.get("name") or c.get("compound") or ""))
        else:
            compound_names.append(str(c))

    genes = data.get("genes") or {}
    n_genes_json = 0
    if isinstance(genes, list):
        n_genes_json = len(genes)
    elif isinstance(genes, dict):
        n_genes_json = len(genes.get("annotations") or []) + len(genes.get("to_add") or [])

    date_added, date_latest = _changelog_dates(data.get("changelog") or {})

    return {
        "bgc_id": str(bgc_id),
        "biosynth_classes_raw": ";".join(raw_classes),
        "biosynth_class": _coarsen_classes(raw_classes),
        "organism": str(organism or "unknown"),
        "ncbi_tax_id": tax_id,
        "n_compounds": len(compound_names),
        "compounds": ";".join(compound_names[:10]),
        "n_genes_json": n_genes_json,
        "status": data.get("status") or "",
        "completeness": data.get("completeness") or "",
        "quality": data.get("quality") or "",
        "date_added": date_added,
        "date_latest": date_latest,
        "json_path": str(path),
    }


def _changelog_dates(changelog: dict[str, Any]) -> tuple[str | None, str | None]:
    """Earliest ("Submitted") and most recent revision dates from MIBiG's changelog."""
    dates: list[str] = []
    for release in changelog.get("releases") or []:
        d = release.get("date")
        if isinstance(d, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            dates.append(d)
        for entry in release.get("entries") or []:
            d = entry.get("date")
            if isinstance(d, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                dates.append(d)
    if not dates:
        return None, None
    dates.sort()
    return dates[0], dates[-1]


_DOMAIN_PATTERNS = re.compile(
    r"(?:PF\d{5}|PKS_[A-Za-z0-9]+|NRPS_[A-Za-z0-9]+|AMP-binding|"
    r"Condensation(?:_[A-Za-z0-9]+)?|Thioesterase|ACP_domain|PCP|"
    r"\bKS\b|\bAT\b|\bDH\b|\bER\b|\bKR\b|\bMT\b|\bTE\b|\bACP\b|"
    r"Terpene_synth|LANC_like|YcaO|DUF\d+|"
    r"polyketide\s+synthase|nonribosomal\s+peptide|"
    r"terpene\s+synthase|lanthipeptide|bacteriocin)",
    re.IGNORECASE,
)


def _product_to_domain_tokens(product: str) -> list[str]:
    """Map free-text CDS products to coarse biosynthetic domain tokens."""
    p = product.lower()
    tokens: list[str] = []
    rules = [
        (r"nonribosomal|nrps|peptide synthetase|adenylation", "NRPS_module"),
        (r"polyketide|pks|ketosynthase|acyltransferase", "PKS_module"),
        (r"terpene|terpenoid|squalene|phytoene", "Terpene_synth"),
        (r"lanthipeptide|lantibiotic|lanC|lanM", "RiPP_lanthipeptide"),
        (r"bacteriocin|ripp|thiazole|oxazole", "RiPP_other"),
        (r"glycosyltransferase", "Glycosyltransferase"),
        (r"methyltransferase", "Methyltransferase"),
        (r"cytochrome p450|p450", "P450"),
        (r"abc transporter|transporter", "Transporter"),
        (r"transcription|regulator|saras-family|luxr", "Regulator"),
        (r"thioesterase", "Thioesterase"),
        (r"acyl carrier|phosphopantetheine", "ACP"),
        (r"condensation", "Condensation"),
        (r"halogenase", "Halogenase"),
        (r"oxygenase|dehydrogenase|reductase", "Redox"),
        (r"hydrolase|esterase|protease", "Hydrolase"),
        (r"kinase", "Kinase"),
        (r"hypothetical|unknown", "Hypothetical"),
    ]
    for pattern, token in rules:
        if re.search(pattern, p):
            tokens.append(token)
    # also raw regex hits
    for m in _DOMAIN_PATTERNS.findall(product):
        tokens.append(re.sub(r"\s+", "_", m.strip())[:48])
    return tokens or (["CDS_other"] if product else [])


def _domains_from_gbk(
    record,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    protein_rows: list[dict[str, Any]] = []
    bgc_id = record.id
    m = re.search(r"(BGC\d+)", record.name or record.id or "")
    if m:
        bgc_id = m.group(1)

    gene_order = 0
    for feat in record.features:
        if feat.type not in {"CDS", "aSDomain", "PFAM_domain", "misc_feature"}:
            continue
        quals = feat.qualifiers

        if feat.type == "CDS":
            gene_order += 1
            locus = (quals.get("locus_tag") or quals.get("gene") or [f"cds_{gene_order}"])[0]
            product = (quals.get("product") or [""])[0]
            seq = quals.get("translation", [""])[0]
            rows.append(
                {
                    "bgc_id": bgc_id,
                    "feature_type": "CDS",
                    "gene_order": gene_order,
                    "locus_tag": locus,
                    "product": product,
                    "domain_id": "",
                    "aa_length": len(seq),
                }
            )
            if len(seq) >= 10:
                protein_rows.append(
                    {
                        "bgc_id": bgc_id,
                        "gene_order": gene_order,
                        "locus_tag": locus,
                        "product": product,
                        "translation": seq,
                        "aa_length": len(seq),
                    }
                )
            for d in _product_to_domain_tokens(product):
                rows.append(
                    {
                        "bgc_id": bgc_id,
                        "feature_type": "domain",
                        "gene_order": gene_order,
                        "locus_tag": locus,
                        "product": product,
                        "domain_id": d,
                        "aa_length": 0,
                    }
                )
            continue

        labels: list[str] = []
        for key in ("aSDomain", "domain", "note", "product", "PFAM"):
            if key in quals:
                labels.extend(str(x) for x in quals[key])
        for lab in labels:
            for tok in _DOMAIN_PATTERNS.findall(lab):
                rows.append(
                    {
                        "bgc_id": bgc_id,
                        "feature_type": "domain",
                        "gene_order": gene_order,
                        "locus_tag": "",
                        "product": "",
                        "domain_id": re.sub(r"\s+", "_", str(tok))[:64],
                        "aa_length": 0,
                    }
                )
    return rows, protein_rows


def _parse_gbk_summary(
    gbk_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    record = next(SeqIO.parse(str(gbk_path), "genbank"))
    bgc_id = record.id
    m = re.search(r"(BGC\d+)", gbk_path.name)
    if m:
        bgc_id = m.group(1)

    cds = [f for f in record.features if f.type == "CDS"]
    aa_lengths = []
    for f in cds:
        if "translation" in f.qualifiers:
            aa_lengths.append(len(f.qualifiers["translation"][0]))
        else:
            aa_lengths.append(max(0, int(f.location.end - f.location.start) // 3))

    domain_rows, protein_rows = _domains_from_gbk(record)
    domain_ids = [r["domain_id"] for r in domain_rows if r["domain_id"]]

    summary = {
        "bgc_id": bgc_id,
        "n_genes": len(cds),
        "cluster_nt_length": len(record.seq),
        "mean_aa_length": float(sum(aa_lengths) / len(aa_lengths)) if aa_lengths else 0.0,
        "total_aa_length": int(sum(aa_lengths)),
        "n_domain_annotations": len(domain_ids),
        "domain_types": ";".join(sorted(Counter(domain_ids).keys())[:80]),
        "gbk_path": str(gbk_path),
    }
    return summary, domain_rows, protein_rows


def curate_mibig() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse JSON + GBK into processed parquet/CSV tables."""
    ensure_dirs()
    json_dir = _find_json_dir()
    gbk_dir = _find_gbk_dir()

    json_paths = [p for p in sorted(json_dir.rglob("*.json")) if "BGC" in p.name]
    LOG.info("Parsing %d MIBiG JSON entries from %s", len(json_paths), json_dir)

    meta_rows = []
    for path in json_paths:
        try:
            row = _parse_one_json(path)
            if row:
                meta_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Failed JSON %s: %s", path.name, exc)

    meta = pd.DataFrame(meta_rows).drop_duplicates(subset=["bgc_id"])

    gbk_paths = sorted(gbk_dir.rglob("*.gbk"))
    LOG.info("Parsing %d GenBank files from %s", len(gbk_paths), gbk_dir)

    gbk_rows = []
    domain_rows: list[dict[str, Any]] = []
    protein_rows: list[dict[str, Any]] = []
    for path in gbk_paths:
        try:
            summary, domains, proteins = _parse_gbk_summary(path)
            gbk_rows.append(summary)
            domain_rows.extend(domains)
            protein_rows.extend(proteins)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Failed GBK %s: %s", path.name, exc)

    gbk_df = pd.DataFrame(gbk_rows).drop_duplicates(subset=["bgc_id"])
    domains_df = pd.DataFrame(domain_rows)
    proteins_df = pd.DataFrame(protein_rows)

    bgcs = meta.merge(gbk_df, on="bgc_id", how="outer")
    bgcs["biosynth_class"] = bgcs["biosynth_class"].fillna("other")
    bgcs["organism"] = bgcs["organism"].fillna("unknown")
    if "n_genes" not in bgcs.columns:
        bgcs["n_genes"] = 0
    bgcs["n_genes"] = bgcs["n_genes"].fillna(bgcs.get("n_genes_json", 0)).fillna(0).astype(int)

    out_bgc = PROCESSED / "mibig_bgcs.parquet"
    out_dom = PROCESSED / "mibig_domains.parquet"
    out_prot = PROCESSED / "mibig_proteins.parquet"
    bgcs.to_parquet(out_bgc, index=False)
    domains_df.to_parquet(out_dom, index=False)
    proteins_df.to_parquet(out_prot, index=False)
    bgcs.to_csv(PROCESSED / "mibig_bgcs.csv", index=False)
    domains_df.to_csv(PROCESSED / "mibig_domains.csv", index=False)

    LOG.info(
        "Wrote %d BGCs (%s), %d domain rows (%s), %d protein sequences (%s)",
        len(bgcs),
        out_bgc,
        len(domains_df),
        out_dom,
        len(proteins_df),
        out_prot,
    )
    LOG.info("Class counts:\n%s", bgcs["biosynth_class"].value_counts().to_string())
    n_dated = bgcs["date_added"].notna().sum() if "date_added" in bgcs.columns else 0
    LOG.info("BGCs with changelog date_added: %d / %d", n_dated, len(bgcs))
    return bgcs, domains_df
