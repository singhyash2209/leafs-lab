# Leafs Lab - From a 2-1 Lead to Cup Chances

[Open the dashboard](https://singhyash2209.github.io/leafs-lab/)

A game-night win probability study with model validation and an experimental Stanley Cup comparison.

## What the dashboard includes

- Toronto Maple Leafs vs New York Islanders game reconstruction.
- An 85.9% estimated win probability at the 2-1 state with 5:51 remaining.
- Calibration and held-out model comparisons, with readable prediction approach labels.
- Projected points, playoff and Stanley Cup probabilities, using full team names.
- A league forecast table, season backtests, model descriptions and limitations.
- A saved 2026-2027 season archive.

## Publishing and freshness

The forecast snapshot uses results through October 3, 2026. It is experimental and is not currently refreshed automatically. The working GitHub Actions workflow only publishes saved static files on changes or manual dispatch. It does not call NHL APIs, run forecasts or use ChatGPT credits. The last saved dashboard remains available independently of API access.

## Method and limits

The game model uses score, time remaining, prior team strength, shots, skaters and goalie-pulled indicators. The reconstructed game was excluded from training.

On held-out games, the calibrated model's Brier score was 0.1785, compared with 0.1763 for the score/time baseline (lower is better). The experimental season model also did not beat its playoff baseline in the latest retrospective check. These findings are displayed beside the charts. Simulation intervals measure sampling error, not all model uncertainty.

## Rebuild the full saved dashboard

```sh
python3 scripts/readable_dashboard.py --artifact outputs/artifact.json --forecast outputs/forecast.json
node scripts/render_dashboard.mjs
python3 scripts/archive_dashboard.py --html outputs/dashboard.html --forecast outputs/forecast.json --docs docs
```

No API request or retraining is required to restore or publish this saved analysis. Source extracts, methodology, notebook and validation results remain in the repository.
