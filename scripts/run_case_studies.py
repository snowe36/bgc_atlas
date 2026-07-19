#!/usr/bin/env python3
"""Biological case studies: where high architecture-novelty lands in phylogeny & chemistry.

Writes:
  reports/biological_case_studies.json
  reports/figures/biological_case_studies.png
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, norm

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MYXO_GENERA = {
    "Sorangium",
    "Chondromyces",
    "Stigmatella",
    "Myxococcus",
    "Corallococcus",
    "Cystobacter",
    "Melittangium",
    "Archangium",
    "Pyxidicoccus",
}

# Hand-picked neighborhoods: each demonstrates a different scientific point.
CASE_STUDIES = [
    {
        "bgc_id": "BGC0000103",
        "title": "Mycolactone — tiny cluster, alien PKS architecture",
        "demonstrates": "Small cluster can still be architecturally unusual",
        "why": (
            "Only 9 genes, yet rank-2 architecture novelty. The locus packs rare PFAM "
            "tokens and insertion-element transposases beside modular PKS machinery — "
            "a pathogen toxin neighborhood far from typical Streptomyces type-I islands."
        ),
    },
    {
        "bgc_id": "BGC0002490",
        "title": "Yersinopine — plague metallophore with almost unique domains",
        "demonstrates": "Rare domain vocabulary / unexplored architectural space",
        "why": (
            "A 6-gene 'other' cluster from Yersinia pestis whose nearest MIBiG neighbor "
            "is a Pseudomonas PKS (class mismatch). DUF6 appears in only three MIBiG "
            "entries; the architecture looks like a transport/metallophore island, not "
            "a classical NRPS siderophore."
        ),
    },
    {
        "bgc_id": "BGC0001313",
        "title": "Arabidiol–baruol — plant terpene island in a microbial atlas",
        "demonstrates": "Cross-kingdom chemical vocabulary in the same atlas",
        "why": (
            "Arabidopsis CYP702/CYP705 P450s plus cellulose synthase-like proteins and "
            "pentacyclic triterpene synthases — a eukaryotic domain vocabulary with "
            "almost no bacterial analogue, so the neighborhood sits at the atlas edge."
        ),
    },
    {
        "bgc_id": "BGC0001884",
        "title": "Aranazole — architecture says weird; ESM says familiar",
        "demonstrates": "Representation disagreement reveals hidden biology",
        "why": (
            "Fischerella NRPS–PKS hybrid with halogenase + P450 tailoring. Architecture "
            "novelty is extreme (~0.99) while ESM novelty is only moderate (~0.59): "
            "sequence space recognizes related enzymes; domain organization does not. "
            "That disagreement is often where interesting discovery lives."
        ),
    },
]


def _genus(organism: str) -> str:
    parts = str(organism).split()
    return parts[0] if parts else ""


def _domain_freq(dom: pd.DataFrame) -> tuple[dict[str, list[str]], Counter]:
    dom_only = dom[dom["domain_id"].notna() & (dom["domain_id"] != "")]
    bgc_domains = dom_only.groupby("bgc_id")["domain_id"].apply(lambda s: sorted(set(s))).to_dict()
    freq: Counter = Counter()
    for ds in bgc_domains.values():
        for d in ds:
            freq[d] += 1
    return bgc_domains, freq


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    # P(a>b) - P(a<b); fine at n≈84×826
    gt = lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    n = len(a) * len(b)
    return float((gt - lt) / n) if n else 0.0


def _myxo_vs_strep(nov: pd.DataFrame, size_max: int | None) -> dict:
    df = nov if size_max is None else nov[nov["n_genes"] <= size_max].copy()
    df = df.copy()
    df["genus"] = df["organism"].map(_genus)
    df["group"] = np.where(
        df["genus"].isin(MYXO_GENERA),
        "Myxobacteria",
        np.where(df["genus"].eq("Streptomyces"), "Streptomyces", "Other"),
    )
    thr = df["novelty"].quantile(0.9)
    myxo = df.loc[df["group"] == "Myxobacteria", "novelty"].to_numpy()
    strep = df.loc[df["group"] == "Streptomyces", "novelty"].to_numpy()
    n_myxo_top = int((myxo >= thr).sum())
    n_myxo_rest = int(len(myxo) - n_myxo_top)
    n_strep_top = int((strep >= thr).sum())
    n_strep_rest = int(len(strep) - n_strep_top)
    # Odds ratio for top-decile membership (Myxo vs Streptomyces)
    # Woolf logit CI; Haldane–Anscombe +0.5 if any cell is zero
    a_ci, b_ci, c_ci, d_ci = float(n_myxo_top), float(n_myxo_rest), float(n_strep_top), float(n_strep_rest)
    if min(a_ci, b_ci, c_ci, d_ci) == 0:
        a_ci, b_ci, c_ci, d_ci = a_ci + 0.5, b_ci + 0.5, c_ci + 0.5, d_ci + 0.5
    odds_ratio = float((a_ci / b_ci) / (c_ci / d_ci))
    se_log_or = float(np.sqrt(1 / a_ci + 1 / b_ci + 1 / c_ci + 1 / d_ci))
    z = float(norm.ppf(0.975))
    or_ci_low = float(np.exp(np.log(odds_ratio) - z * se_log_or))
    or_ci_high = float(np.exp(np.log(odds_ratio) + z * se_log_or))
    enrich_ratio = float((n_myxo_top / len(myxo)) / (n_strep_top / len(strep)))
    u_stat, p_value = mannwhitneyu(myxo, strep, alternative="greater")

    groups = []
    for gname, gdf in df.groupby("group"):
        groups.append(
            {
                "group": gname,
                "n": int(len(gdf)),
                "mean_novelty": float(gdf["novelty"].mean()),
                "median_novelty": float(gdf["novelty"].median()),
                "frac_top_decile": float((gdf["novelty"] >= thr).mean()),
            }
        )

    genus_top = df[df["novelty"] >= thr]["genus"].value_counts()
    genus_all = df["genus"].value_counts()
    enrich_rows = []
    for genus, top_n in genus_top.items():
        all_n = int(genus_all.get(genus, 0))
        if top_n < 3 or all_n < 5:
            continue
        frac_top = top_n / genus_top.sum()
        frac_all = all_n / genus_all.sum()
        enrich_rows.append(
            {
                "genus": genus,
                "n_top_decile": int(top_n),
                "n_total": all_n,
                "enrichment": float(frac_top / frac_all) if frac_all else None,
            }
        )
    enrich_rows.sort(key=lambda r: r["enrichment"] or 0, reverse=True)

    return {
        "size_filter_n_genes_max": size_max,
        "n_myxo": int(len(myxo)),
        "n_streptomyces": int(len(strep)),
        "median_myxo": float(np.median(myxo)),
        "median_streptomyces": float(np.median(strep)),
        "frac_top_decile_myxo": float(n_myxo_top / len(myxo)),
        "frac_top_decile_streptomyces": float(n_strep_top / len(strep)),
        "top_decile_enrichment_ratio": enrich_ratio,
        "odds_ratio_top_decile": odds_ratio,
        "odds_ratio_top_decile_ci95": [or_ci_low, or_ci_high],
        "contingency_top_decile": {
            "myxo_top": n_myxo_top,
            "myxo_rest": n_myxo_rest,
            "streptomyces_top": n_strep_top,
            "streptomyces_rest": n_strep_rest,
        },
        "cohens_d": _cohens_d(myxo, strep),
        "cliffs_delta": _cliffs_delta(myxo, strep),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(p_value),
        "groups": groups,
        "genus_enrichment_top_decile": enrich_rows[:12],
        "_df": df,
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    nov = pd.read_csv(REPORTS / "novelty_ranking.csv")
    meta = pd.read_csv(PROCESSED / "feature_meta.csv")
    coords = pd.read_csv(PROCESSED / "atlas_coords.csv")
    dom = pd.read_csv(PROCESSED / "mibig_domains.csv")
    cmp = pd.read_csv(REPORTS / "novelty_representation_comparison.csv")

    bgc_domains, domain_freq = _domain_freq(dom)
    n_bgcs = len(bgc_domains)
    nov_i = nov.set_index("bgc_id")
    meta_i = meta.set_index("bgc_id")
    cmp_i = cmp.set_index("bgc_id")

    unrestricted = _myxo_vs_strep(nov, size_max=None)
    size_ctrl = _myxo_vs_strep(nov, size_max=60)
    nov2 = size_ctrl.pop("_df")
    unrestricted.pop("_df")

    cases_out = []
    for case in CASE_STUDIES:
        bid = case["bgc_id"]
        r = nov_i.loc[bid]
        m = meta_i.loc[bid]
        nn = r["nearest_mibig"]
        nn_compounds = nov_i.loc[nn, "compounds"] if nn in nov_i.index else None
        nn_org = nov_i.loc[nn, "organism"] if nn in nov_i.index else None
        uniq = bgc_domains.get(bid, [])
        rare = sorted(
            [{"domain": d, "n_bgcs": int(domain_freq[d]), "frac": float(domain_freq[d] / n_bgcs)} for d in uniq],
            key=lambda x: x["n_bgcs"],
        )[:6]
        esm_nov = float(cmp_i.loc[bid, "novelty_esm"]) if bid in cmp_i.index else None
        cases_out.append(
            {
                **case,
                "organism": m["organism"],
                "biosynth_class": m["biosynth_class"],
                "compounds": m["compounds"],
                "n_genes": int(m["n_genes"]),
                "novelty": float(r["novelty"]),
                "rank": int(r["rank"]),
                "nearest_mibig": nn,
                "neighbor_class": r["neighbor_class"],
                "nearest_compounds": nn_compounds,
                "nearest_organism": nn_org,
                "novelty_esm": esm_nov,
                "rarest_domains": rare,
                "domain_types": m["domain_types"],
            }
        )

    report = {
        "observed": (
            "Myxobacterial BGCs occupy regions of architectural space with higher "
            "novelty scores than Streptomyces BGCs; the enrichment persists after "
            "restricting to n_genes≤60."
        ),
        "interpretation": (
            "This suggests lineage-specific expansion of biosynthetic design patterns "
            "(a 'biosynthetic grammar'), measured relative to the architecture "
            "representation — not a claim of evolutionary innovation per se."
        ),
        "size_filter_n_genes_max": 60,
        "phylogeny": {
            "unrestricted": unrestricted,
            "size_controlled_n_genes_le_60": size_ctrl,
        },
        "case_studies": cases_out,
    }
    out_json = REPORTS / "biological_case_studies.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Figure: size-control survival + violin + atlas callouts
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    # Panel A — top-decile % before/after size control
    bar_df = pd.DataFrame(
        [
            {
                "filter": "all sizes",
                "group": "Myxobacteria",
                "frac": unrestricted["frac_top_decile_myxo"],
            },
            {
                "filter": "all sizes",
                "group": "Streptomyces",
                "frac": unrestricted["frac_top_decile_streptomyces"],
            },
            {
                "filter": "≤60 genes",
                "group": "Myxobacteria",
                "frac": size_ctrl["frac_top_decile_myxo"],
            },
            {
                "filter": "≤60 genes",
                "group": "Streptomyces",
                "frac": size_ctrl["frac_top_decile_streptomyces"],
            },
        ]
    )
    palette = {"Myxobacteria": "#c44e52", "Streptomyces": "#4c72b0"}
    sns.barplot(
        data=bar_df,
        x="filter",
        y="frac",
        hue="group",
        palette=palette,
        ax=axes[0],
        order=["all sizes", "≤60 genes"],
    )
    axes[0].set_ylabel("fraction in top novelty decile")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 0.45)
    ci_lo, ci_hi = size_ctrl["odds_ratio_top_decile_ci95"]
    axes[0].set_title(
        f"Enrichment survives size control\n"
        f"(OR={size_ctrl['odds_ratio_top_decile']:.1f}, "
        f"95% CI {ci_lo:.1f}–{ci_hi:.1f}; ≤60-gene filter)"
    )
    axes[0].legend(title="", loc="upper right", fontsize=9)

    # Panel B — size-controlled score distributions
    order = ["Myxobacteria", "Streptomyces", "Other"]
    full_palette = {**palette, "Other": "#cccccc"}
    plot_df = nov2[nov2["group"].isin(order)]
    sns.violinplot(
        data=plot_df,
        x="group",
        y="novelty",
        order=order,
        hue="group",
        palette=full_palette,
        legend=False,
        cut=0,
        inner="quartile",
        ax=axes[1],
    )
    axes[1].set_title(
        f"Score distributions (n_genes≤60)\n"
        f"Cliff δ={size_ctrl['cliffs_delta']:.2f}, "
        f"d={size_ctrl['cohens_d']:.2f}, "
        f"p={size_ctrl['mannwhitney_p']:.2g}"
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("novelty score")

    # Panel C — atlas callouts
    atlas = coords.merge(nov[["bgc_id", "novelty"]], on="bgc_id", how="left")
    atlas["genus"] = atlas["organism"].map(_genus)
    atlas["is_myxo"] = atlas["genus"].isin(MYXO_GENERA)
    labels = {
        "BGC0000103": ("mycolactone", (8, 12)),
        "BGC0002490": ("yersinopine", (8, -14)),
        "BGC0001313": ("arabidiol", (-8, 12)),
        "BGC0001884": ("aranazole", (8, 8)),
    }
    case_pts = atlas[atlas["bgc_id"].isin(labels)]
    x, y = atlas["dim1"].to_numpy(), atlas["dim2"].to_numpy()
    qx0, qx1 = np.nanpercentile(x, [1, 99])
    qy0, qy1 = np.nanpercentile(y, [1, 99])
    qx0 = min(qx0, float(case_pts["dim1"].min()))
    qx1 = max(qx1, float(case_pts["dim1"].max()))
    qy0 = min(qy0, float(case_pts["dim2"].min()))
    qy1 = max(qy1, float(case_pts["dim2"].max()))
    pad_x, pad_y = 0.12 * (qx1 - qx0), 0.12 * (qy1 - qy0)
    axes[2].scatter(atlas["dim1"], atlas["dim2"], s=8, c="#dddddd", alpha=0.7, linewidths=0)
    myxo_pts = atlas[atlas["is_myxo"]]
    axes[2].scatter(
        myxo_pts["dim1"],
        myxo_pts["dim2"],
        s=22,
        c="#c44e52",
        alpha=0.85,
        label="Myxobacteria",
        linewidths=0,
    )
    for bid, (label, offset) in labels.items():
        row = atlas.loc[atlas["bgc_id"] == bid].iloc[0]
        axes[2].scatter([row["dim1"]], [row["dim2"]], s=90, c="#222222", zorder=5)
        axes[2].annotate(
            label,
            (row["dim1"], row["dim2"]),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
            fontweight="bold",
            ha="left" if offset[0] >= 0 else "right",
            arrowprops={"arrowstyle": "-", "color": "#444444", "lw": 0.8},
        )
    axes[2].set_xlim(qx0 - pad_x, qx1 + pad_x)
    axes[2].set_ylim(qy0 - pad_y, qy1 + pad_y)
    axes[2].set_xlabel("embed-1")
    axes[2].set_ylabel("embed-2")
    axes[2].set_title("Case-study neighborhoods on the atlas")
    axes[2].legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig_path = FIGURES / "biological_case_studies.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_json}")
    print(f"Wrote {fig_path}")
    ci_lo, ci_hi = size_ctrl["odds_ratio_top_decile_ci95"]
    print(
        f"Size-controlled: OR={size_ctrl['odds_ratio_top_decile']:.2f} "
        f"(95% CI {ci_lo:.2f}–{ci_hi:.2f}) "
        f"d={size_ctrl['cohens_d']:.3f} δ={size_ctrl['cliffs_delta']:.3f} "
        f"p={size_ctrl['mannwhitney_p']:.3g}"
    )


if __name__ == "__main__":
    main()
