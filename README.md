# Leafs Lab - Game Night Win Probability

[Open the dashboard](https://singhyash2209.github.io/leafs-lab/)

A retrospective win-probability dashboard for **Toronto Maple Leafs vs New York Islanders on September 30, 2026**.

At Toronto's 2-1 lead with 5:51 remaining, the model estimated an **85.9% chance of winning**. The chart follows the probability through that one game.

## Scope

The public page is a saved one-game analysis. It does not refresh daily, fetch new seasons or publish Cup forecasts. GitHub Actions only deploys the static files when those files change, or when manually triggered. It does not use ChatGPT tasks or AI API calls.

## Method and limits

The underlying Python model uses score, time remaining, prior team strength, shots, skaters and goalie-pulled indicators. This game was excluded from training. Probabilities are estimates, not guarantees.

On held-out games, the calibrated model's Brier score was 0.1785, compared with 0.1763 for the simpler score/time baseline (lower is better). This project illustrates game-state modeling; it does not establish superior predictive performance.

The saved analysis and source provenance are in `outputs/artifact.json` and `outputs/game_explanation.json`. Historical research files remain for reproducibility; they are not part of the public dashboard.

## Rebuild the saved page

No new API request or model training is required:

```sh
python3 scripts/game_night_dashboard.py
node scripts/render_dashboard.mjs outputs/game_night_artifact.json
cp outputs/dashboard.html docs/index.html
```

The portable renderer is packaged in the repository. GitHub Pages serves `docs/index.html` through the static deployment workflow.
