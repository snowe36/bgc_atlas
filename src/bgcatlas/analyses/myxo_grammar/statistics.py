"""Clean per-token enrichment statistics (OR, CI, FDR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # monotone from the back
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def odds_ratio_ci(a: float, b: float, c: float, d: float) -> tuple[float, float, float]:
    """Haldane–Anscombe corrected OR + Woolf 95% CI."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a / b) / (c / d)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = float(norm.ppf(0.975))
    lo = float(np.exp(np.log(or_) - z * se))
    hi = float(np.exp(np.log(or_) + z * se))
    return float(or_), lo, hi


def enrich_tokens(
    mat: pd.DataFrame,
    myxo_ids: list[str],
    control_ids: list[str],
    *,
    idf: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-token contingency enrichment: myxo vs matched controls."""
    myxo = set(map(str, myxo_ids))
    ctrl = set(map(str, control_ids))
    # Unique controls only (a control matched to multiple myxo counts once).
    m_idx = [i for i in mat.index.astype(str) if i in myxo]
    c_idx = [i for i in mat.index.astype(str) if i in ctrl]
    if not m_idx or not c_idx:
        return pd.DataFrame()

    rows = []
    n_m, n_c = len(m_idx), len(c_idx)
    for token in mat.columns:
        m_pos = int(mat.loc[m_idx, token].sum())
        c_pos = int(mat.loc[c_idx, token].sum())
        m_neg = n_m - m_pos
        c_neg = n_c - c_pos
        table = np.array([[m_pos, m_neg], [c_pos, c_neg]], dtype=float)
        # Fisher exact: alternative greater → myxo enriched
        _, p_greater = fisher_exact(table, alternative="greater")
        _, p_two = fisher_exact(table, alternative="two-sided")
        or_, lo, hi = odds_ratio_ci(float(m_pos), float(m_neg), float(c_pos), float(c_neg))
        level = (
            "capacity"
            if token.startswith("capacity::")
            else "grammar"
            if token.startswith("grammar::")
            else "motif"
            if token.startswith("motif")
            else "domain"
        )
        rows.append(
            {
                "token": token,
                "level": level,
                "myxo_positive": m_pos,
                "myxo_n": n_m,
                "control_positive": c_pos,
                "control_n": n_c,
                "myxo_fraction": m_pos / n_m,
                "control_fraction": c_pos / n_c if n_c else np.nan,
                "odds_ratio": or_,
                "or_ci95_lo": lo,
                "or_ci95_hi": hi,
                "log2_or": float(np.log2(or_)) if or_ > 0 else np.nan,
                "p_fisher_greater": float(p_greater),
                "p_fisher_twosided": float(p_two),
                "idf": float(idf.get(token, np.nan)) if idf else np.nan,
                "interpretation": _interpret(token, or_),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh"] = _bh_fdr(out["p_fisher_greater"].to_numpy())
    out = out.sort_values(["q_bh", "odds_ratio"], ascending=[True, False]).reset_index(drop=True)
    return out


def _interpret(token: str, or_: float) -> str:
    direction = "enriched in myxo" if or_ >= 1 else "depleted in myxo"
    if token.startswith("grammar::"):
        return f"Assembly combo ({token.split('::', 1)[1]}) {direction}"
    if token.startswith("capacity::"):
        return f"Functional capacity ({token.split('::', 1)[1]}) {direction}"
    if token.startswith("motif"):
        return f"Ordered motif {direction}"
    if token.startswith("dom::"):
        return f"Domain {token.split('::', 1)[1]} {direction}"
    return direction


def rarity_contrast(
    rarity: pd.Series,
    myxo_ids: list[str],
    control_ids: list[str],
) -> dict:
    """Do myxo BGCs carry rarer architectural vocabulary?"""
    m = rarity.reindex([str(i) for i in myxo_ids]).dropna()
    c = rarity.reindex([str(i) for i in control_ids]).dropna()
    if m.empty or c.empty:
        return {"n_myxo": 0, "n_control": 0}
    from scipy.stats import mannwhitneyu

    u, p = mannwhitneyu(m.to_numpy(), c.to_numpy(), alternative="greater")
    return {
        "n_myxo": int(len(m)),
        "n_control": int(len(c)),
        "median_idf_myxo": float(m.median()),
        "median_idf_control": float(c.median()),
        "mean_idf_myxo": float(m.mean()),
        "mean_idf_control": float(c.mean()),
        "mannwhitney_u": float(u),
        "mannwhitney_p_myxo_greater": float(p),
    }
