"""Atlas figure color palette (see atlas-figure-palette skill)."""

from __future__ import annotations

PALETTE = {
    "sage": "#8FB996",
    "teal": "#5FA8A8",
    "peach": "#F4A261",
    "coral": "#E76F51",
    "lavender": "#B8A9E8",
    "mustard": "#E9C46A",
    "navy": "#264653",
}

TEXT = PALETTE["navy"]
MUTED = "#5A6F74"
GRID = "#D9E2E0"
FACE = "#FFFFFF"
SPINE = "#A8B8B5"

CLASS_COLORS = {
    "NRPS": PALETTE["teal"],
    "PKS": PALETTE["sage"],
    "RiPP": PALETTE["mustard"],
    "terpene": PALETTE["peach"],
    "hybrid": PALETTE["coral"],
    "other": PALETTE["lavender"],
}

CLADE_COLORS = {
    "Myxobacteria": PALETTE["coral"],
    "Streptomyces": PALETTE["teal"],
    "Other": GRID,
}

SERIES = [
    PALETTE["teal"],
    PALETTE["sage"],
    PALETTE["mustard"],
    PALETTE["peach"],
    PALETTE["coral"],
    PALETTE["lavender"],
]
