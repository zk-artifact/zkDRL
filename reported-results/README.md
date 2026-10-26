# Reported campaign summaries

These files are sanitized aggregates from the paper's single unified campaign
(`full-023-unified`, schemas 202/203). They are included so reviewers can
cross-check the paper's tables and run the plotting code without downloading
raw proof bundles or telemetry.

| File | Contents |
| --- | --- |
| `proof-configuration-summary.csv` | 22 proof configurations; median/min/max over 3 timing requests; bundle size and verification timing |
| `PROOF_ANALYSIS.json` | Configuration-level proof analysis consumed by `plot_results.py` |
| `numerical-condition-summary.csv` | 22 condition summaries (5 seeds each); observed min/max and available n |
| `numerical-endpoints.csv` | 110 final-checkpoint rows with training/evaluation status |
| `numerical-checkpoint-curves.csv` | 550 stored checkpoint rows (0/1000/2000/3000/4000) |
| `NUMERICAL_ANALYSIS.json` | Diagnostics and condition summaries consumed by the analysis/plots |

PDF/SVG figures are intentionally not shipped: renderer metadata can carry
timestamps or tool fingerprints. They are regenerated from the CSV/JSON files
with `plot_results.py` (for example, copy these files into
`runs/full-023-unified/analysis/` and run `python3 -B plot_results.py all`).

The summary is not raw evidence and does not include proof bundles, auditor
databases, host telemetry, cloud instance records, or per-request logs. Missing
evaluations are `NA` with explicit available `n`; no zero imputation or
best-checkpoint selection was applied. The 22 configurations are not 66
independent learning samples: the three requests per configuration are timing
repetitions on the same committed input and challenge.
