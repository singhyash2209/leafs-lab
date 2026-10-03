# Leafs Lab v0.1 — model card

**Status:** real-data game modeling executed; season forecasts experimental.

## Evidence

- Official NHL source: 5,256 completed regular-season games; 1,278,937 nonterminal regulation states; 100% coverage for each selected historical season; no parser errors or missing required features.
- Training: 2,624 games (2022–23/2023–24). Tuning: 639 games (early 2024–25). Calibration: 673 games (later 2024–25). Chronological test: 1,312 games and 321,124 states (2025–26).
- Validation selected logistic regression with C=1. Five alternatives were compared, including the score/time baseline; extra features were not forced to win.
- Test calibrated Brier 0.178507 vs baseline 0.176329. Paired game-bootstrap difference 0.002178, interval −0.000792 to 0.005057. **Improvement over baseline not established.**
- Test calibrated classification accuracy 71.3%, AUC 0.8073, log loss 0.5209. Accuracy alone does not validate probability estimates.
- Held-out home-leading 80–90% cohort: 679 games, mean prediction 85.1%, observed wins 80.1%, game-bootstrap observed-rate interval 77.2–82.9%. **Overconfidence is visible.**
- Toronto's September 30, 2026 2–1 state at 54:09 elapsed: model estimate 85.9%. This retrospective reconstruction does not use the final outcome as a predictor.

## Season forecast

10,000 seeded simulations as of October 1, 2026. Toronto: playoffs 13.98%, Final 0.67%, Cup 0.25%. Cup Monte Carlo interval 0.1694–0.3688%; this measures numerical sampling error, not full forecast uncertainty.

Future strength receives one latent draw per club per simulated season, with SD 70.34 Elo estimated from training-season rating drift. This proxy is not a calibrated posterior. No explicit injuries, starting goalies, roster/trades, fatigue or expected-goals data enter the model.

The fixed-strength baseline was reviewed first and retained. The uncertainty revision followed that review, so its three retrospective season checks are **exploratory, not a pristine untouched holdout**. Latest playoff Brier 0.27146 vs naive 0.25. Latest Cup Brier 0.03088 vs uniform-team 0.03027. **Reliable championship odds have not been demonstrated.**

## Method and safeguards

Features: score, remaining time, score/time interaction, past-only pregame Elo, cumulative shots, observed skaters, goalie-pulled indicators, missing-situation flag. Final outcomes are labels only. All game states stay in one chronological split. Each game receives equal total evaluation weight. Same-day Elo outcome updates are batched.

Candidate selection occurs on tuning dates; logistic probability calibration uses separate later dates; game test metrics describe a frozen model. Permutation importance and one-feature contrasts are noncausal and nonadditive. Observations occur at recorded events, not uniform seconds.

Source responses preserve URLs, timestamps and SHA-256 checksums. Parser reconciles source IDs, scores, regulation/OT outcomes, clock order and event uniqueness. Dashboard probabilities and errors are recomputed in SQLite, with model inference through a Python UDF; they reconcile against the frozen Python outputs. A partial prediction CSV was regenerated from the unchanged model and verified feature matrix, with full-row readback; `prediction_export_repair.json` records this.

Season simulator uses schedule/standings reconciliation, overtime-loss points, division/wildcard qualification and four best-of-seven rounds. Stage totals must be 16/8/4/2/1. Official head-to-head standings tie-breaks and shootout-specific simulation are approximated. Postseason updates are blocked after the regular-season finish until an engine that conditions on completed playoff games exists.

## Delivery and validation limits

Notebook code cells executed top-to-bottom with in-process IPython and real-data rich outputs. Standard nbclient/Jupyter kernel transport was blocked by sandbox socket restrictions. Notebook structure validates with nbformat.

Dashboard artifact and self-contained reader passed canonical validation and structural verification. Interactive browser checks were unavailable because Chromium was not installed. Scientific PNG figures were visually inspected. These software checks do not establish forecasting accuracy.
