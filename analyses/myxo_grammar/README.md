# Myxobacterial architecture grammar

**Question.** Given that myxobacterial BGCs occupy unusual hashed-architecture neighborhoods after size matching, which architectural primitives account for that separation?

**Not the question.** Which BGCs are novel?

Matching isolates architectural composition from size and coarse class. The rarity control (unchanged mean token-IDF) supports arrangement grammar over rarer parts. Temporal myxo contrasts stay exploratory (small held-out *n*).

Implementation lives in the package:

`src/bgcatlas/analyses/myxo_grammar/`

| Module | Role |
|--------|------|
| `cohort.py` | Phenotype gate (OR≈2.9) + size/class-matched cohort |
| `tokens.py` | Domain / motif / capacity / combo tokens from existing annotations |
| `statistics.py` | OR, 95% CI, Fisher, BH–FDR |
| `enrichment.py` | Matched enrichment + temporal myxo sneak |
| `figures.py` | Figure 5 panels |
| `run.py` | Orchestrator |

## Run

```bash
uv run --no-sync bgc-myxo-grammar
# or:
uv run --no-sync python scripts/run_myxo_grammar.py
```

## Outputs

- `reports/myxo_grammar_report.md`
- `reports/myxo_grammar/token_enrichment.csv`
- `reports/figures/fig_myxo_architecture_grammar.png`
