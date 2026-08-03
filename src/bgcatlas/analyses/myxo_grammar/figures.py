"""Figure 5 — myxobacterial architecture grammar."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from bgcatlas.viz.palette import CLADE_COLORS, GRID, MUTED, PALETTE, TEXT


def make_figure5(
    cohort: pd.DataFrame,
    coords: pd.DataFrame | None,
    enrichment: pd.DataFrame,
    *,
    out_png: Path,
    out_pdf: Path | None = None,
    top_n: int = 10,
) -> Path:
    """Four-panel figure: phenotype map, volcano, top grammars, schematic."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    fig.patch.set_facecolor("white")

    _panel_a_phenotype(axes[0, 0], cohort, coords)
    _panel_b_volcano(axes[0, 1], enrichment)
    _panel_c_top_tokens(axes[1, 0], enrichment, top_n=top_n)
    _panel_d_schematic(axes[1, 1])

    fig.suptitle(
        "Myxobacterial BGCs occupy unusual architecture neighborhoods for "
        "interpretable assembly-grammar reasons",
        fontsize=12,
        color=TEXT,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def _panel_a_phenotype(ax, cohort: pd.DataFrame, coords: pd.DataFrame | None) -> None:
    ax.set_title("A. Matched cohort on the atlas", loc="left", fontsize=11, color=TEXT)
    if coords is None or coords.empty:
        ax.text(0.5, 0.5, "atlas coords unavailable", ha="center", va="center", color=MUTED)
        ax.axis("off")
        return
    df = cohort.merge(coords[["bgc_id", "dim1", "dim2"]], on="bgc_id", how="left")
    df = df.dropna(subset=["dim1", "dim2"])
    ctrl = df[df["role"] == "control"]
    myxo = df[df["role"] == "myxo"]
    ax.scatter(ctrl["dim1"], ctrl["dim2"], s=18, c=CLADE_COLORS["Streptomyces"], alpha=0.65, label="matched Streptomyces", linewidths=0)
    ax.scatter(myxo["dim1"], myxo["dim2"], s=28, c=CLADE_COLORS["Myxobacteria"], alpha=0.9, label="Myxobacteria", linewidths=0)
    ax.set_xlabel("embed-1")
    ax.set_ylabel("embed-2")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(MUTED)


def _panel_b_volcano(ax, enrichment: pd.DataFrame) -> None:
    ax.set_title("B. Token enrichment volcano", loc="left", fontsize=11, color=TEXT)
    if enrichment.empty:
        ax.text(0.5, 0.5, "no enrichment rows", ha="center", va="center", color=MUTED)
        ax.axis("off")
        return
    df = enrichment.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2_or", "q_bh"])
    y = -np.log10(df["q_bh"].clip(lower=1e-300))
    colors = []
    for _, r in df.iterrows():
        if r["q_bh"] <= 0.05 and r["log2_or"] >= 1:
            colors.append(PALETTE["coral"])
        elif r["q_bh"] <= 0.05 and r["log2_or"] <= -1:
            colors.append(PALETTE["teal"])
        else:
            colors.append(GRID)
    ax.scatter(df["log2_or"], y, c=colors, s=22, alpha=0.85, linewidths=0)
    ax.axhline(-np.log10(0.05), color=MUTED, ls="--", lw=0.8)
    ax.axvline(0, color=MUTED, ls="-", lw=0.6)
    # label top hits
    hits = df[(df["q_bh"] <= 0.05) & (df["log2_or"] >= 1)].nsmallest(8, "q_bh")
    for _, r in hits.iterrows():
        label = r["token"].split("::", 1)[-1]
        ax.annotate(
            label[:28],
            (r["log2_or"], -np.log10(max(r["q_bh"], 1e-300))),
            fontsize=7,
            color=TEXT,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("log2(odds ratio)  myxo vs matched control")
    ax.set_ylabel("−log10(BH q)")
    ax.tick_params(colors=MUTED)


def _panel_c_top_tokens(ax, enrichment: pd.DataFrame, *, top_n: int) -> None:
    ax.set_title("C. Top enriched architectural grammars", loc="left", fontsize=11, color=TEXT)
    if enrichment.empty:
        ax.axis("off")
        return
    # Prefer grammar/capacity; fall back to any significant
    pref = enrichment[enrichment["level"].isin(["grammar", "capacity"])]
    pref = pref[pref["q_bh"] <= 0.1]
    if len(pref) < 3:
        pref = enrichment[enrichment["q_bh"] <= 0.1]
    if pref.empty:
        pref = enrichment.head(top_n)
    top = pref.head(top_n).iloc[::-1]
    y = np.arange(len(top))
    ax.barh(y - 0.15, top["myxo_fraction"], height=0.3, color=CLADE_COLORS["Myxobacteria"], label="myxo")
    ax.barh(y + 0.15, top["control_fraction"], height=0.3, color=CLADE_COLORS["Streptomyces"], label="control")
    labels = [t.split("::", 1)[-1] for t in top["token"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("fraction of BGCs with token")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, min(1.0, float(max(top["myxo_fraction"].max(), top["control_fraction"].max()) * 1.25) + 0.05))
    ax.tick_params(colors=MUTED)


def _panel_d_schematic(ax) -> None:
    ax.set_title("D. Interpretation", loc="left", fontsize=11, color=TEXT)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    boxes = [
        (1.2, 7.2, 7.6, 1.8, "Myxobacteria", PALETTE["coral"]),
        (1.2, 4.4, 7.6, 1.8, "Preferential assembly grammar\n(domains · motifs · combos)", PALETTE["mustard"]),
        (1.2, 1.6, 7.6, 1.8, "Shift in hashed-architecture\nneighborhood (size-matched)", PALETTE["teal"]),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor=fc,
                edgecolor=TEXT,
                linewidth=1.2,
                alpha=0.4,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=TEXT)
    for y0, y1 in ((7.2, 6.2), (4.4, 3.4)):
        ax.annotate(
            "",
            xy=(5, y1),
            xytext=(5, y0),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.4),
        )
    ax.text(
        5,
        0.45,
        "Novelty ≠ discovery · grammar explains the neighborhood",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
        style="italic",
    )
