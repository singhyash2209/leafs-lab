# Validation assessment: share with explicit caveats

The project is a reproducible portfolio experiment. It is not evidence of superior or reliable Stanley Cup forecasting.

## Data and arithmetic

All 5,256 eligible games downloaded and parsed, producing 1,278,937 regulation observations. Source checksums, season coverage, final-score reconciliation, missingness, game weights and temporal partitions were checked. No source or parser exclusions occurred.

11 numerical/regression tests passed: clock parsing, Elo probability, same-day outcome isolation, goal and empty-net states, missing situation, duplicate/schema rejection, temporal training pipeline, seeded bracket totals, nonempty schedule coverage, overtime-point conservation and invalid overtime score rejection.

Python and SQLite dashboard values independently reconcile for Brier scores, classification accuracy, the complete game curve, preseason point MAE and playoff scores. Prediction export readback covers all 321,124 held-out states and 1,312 games after recovery of an incomplete intermediate CSV.

## Statistical assessment

Advanced features and calibration did not beat the score/time baseline on held-out Brier score. The paired interval includes zero; no improvement claim is supported. The selected probability band shows overconfidence.

The season model's uncertainty revision followed baseline review. Three retrospective seasons do not establish Cup calibration. Latest playoff Brier 0.27146 exceeds the constant 50% baseline's 0.25; Cup Brier 0.03088 also exceeds uniform-team baseline 0.03027. Current season percentages are exploratory model outputs.

## Artifact checks

- Notebook: all eight code cells executed in order with in-process IPython; rich outputs saved; nbformat validates. Standard network/IPC Jupyter kernel execution was blocked by this environment.
- Dashboard: canonical payload validation, packaging and structural verification passed. No installed Chromium meant enhanced-reader/browser interaction QA and automatic chart SVG extraction could not run. The portable reader includes semantic chart tables as fallback.
- Scientific figures: game curve and calibration PNGs visually inspected; bounded 0–100% / 0–1 scales and readable labels confirmed.
- Refresh: a fast update path preserves historical model/evaluation and refreshes only current source results, ratings and simulation. Activation of a scheduled task is reported separately from providing a workflow template.

## Scope limits

No explicit injury, lineup, goalie starter, trade, fatigue or expected-goals model. Independent Poisson regulation goals. Simulated tied games resolve as OT. Official head-to-head tie-breaks approximated. Future season-strength SD is an empirical drift proxy. Confidence intervals for Cup numbers concern Monte Carlo error only. Historical source snapshots and a future API schema change can affect reproducibility.
