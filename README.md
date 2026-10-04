# Leafs Lab - NHL analytics portfolio

Python analysis by Yash Singh: in-game win probabilities, historical EDA, Elo ratings, season simulation and validation.

Explore the dashboard: https://singhyash2209.github.io/leafs-lab/

Source code: https://github.com/singhyash2209/leafs-lab

The public dashboard has a stable home and a [season archive](https://singhyash2209.github.io/leafs-lab/archive.html). The initial published season is **2026-27**, snapshot **October 3, 2026**, using completed results through October 2. GitHub Actions independently runs the Python API refresh daily at **6:17 a.m. America/Toronto**. The former ChatGPT task is paused. Failed runs retain the last good snapshot; inspect the displayed date. Future seasons require explicit configuration and model checks before publication. Hosting availability depends on GitHub and maintaining this repository.

To add or update a season after generating its verified portable dashboard:

```bash
python scripts/archive_dashboard.py --html outputs/dashboard.html --forecast outputs/forecast.json --docs docs
```

Commit the changed `docs` files to `main`; Pages serves `main:/docs`. The script preserves other season directories and updates the archive registry. The public archive is a snapshot; the separate private Site also has direct NHL standings polling while open.

## What the evidence supports

- At Toronto's first observed 2-1 state, with 5:51 remaining, the model estimated **85.9%** eventual win probability. This is a retrospective model estimate, not proof of accuracy from one win.
- Game evaluation held out **1,312 whole games** from 2025-26, with **321,124 regulation states**.
- Calibrated game-weighted Brier score: **0.1785**, versus **0.1763** for the score/time baseline. Lower is better; improvement was not established.
- Season and Cup forecasts remain **experimental**. Their validation is separate from the game model, and the latest playoff backtest did not beat the naive baseline.

## Python project documentation

Author/project owner: Yash Singh. Python portfolio project inspired by Toronto's September 30, 2026 game.

## Execution status

The initial direct API requests returned HTTP 403. The pipeline subsequently obtained official NHL data. **All 5,256 completed regular-season games parsed successfully**, with **1,278,937 regulation states**, 100% per-season coverage and no parser errors. Training/validation/test results, fitted models and figures are included after execution.

The game model has a chronological test season. The season model remains **experimental**: its strength-uncertainty revision was made after inspecting the initial fixed-strength baseline, and its latest retrospective playoff Brier score still exceeds the naive 0.25 baseline. Do not market the Cup percentages as independently validated odds. See `outputs/` and `MODEL_CARD.md` for evidence and limits.

## Run on your laptop in VS Code

Use Python 3.11 or 3.12. Open the extracted `leafs-lab` directory as the VS Code folder.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m leafslab run
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m leafslab run
```

The first run downloads several thousand NHL games. It may take substantial time. Cache is reused, schedule downloads use four workers and PBP downloads default to eight workers (configurable via LEAFSLAB_DOWNLOAD_WORKERS), retry transient failures, and preserve source URLs, timestamps, and SHA-256 checksums. HTTP 403 is a hard source-access failure, not a retryable empty dataset. No API key is required by the adapter, but endpoint availability/access is not guaranteed.

This GitHub package includes source code, the executed notebook, selected evaluation outputs and a portable dashboard export. Large processed datasets, raw sources, the fitted binary model and source database are omitted; rebuild these with the pipeline. Extract `leafs-lab-sources.zip` beside it to restore the `leafs-lab/data/raw/` cache. This permits a full `run --offline` without another historical download. Public endpoints may change; preserve the `.meta.json` provenance records.

## Commands and outputs

| Command | Purpose | Main outputs |
|---|---|---|
| `python -m leafslab fetch` | Download all configured regular/postseason schedules and completed regular-season PBP | `data/raw/`, `data/games.csv`, source-error logs |
| `python -m leafslab prepare` | Validate PBP, reconstruct known state, build past-only Elo and game weights | `data/states.csv`, `rated_games.csv`, quality/EDA exports |
| `python -m leafslab train` | Train candidates, select on tuning period, calibrate on later validation period, score held-out season | `model.joblib`, `metrics.json`, `test_predictions.csv` |
| `python -m leafslab forecast` | Simulate remaining league schedule and playoffs | `forecast.json`, `forecast.csv`, Toronto strength sensitivity |
| `python -m leafslab game --date 2026-09-30` | Reconstruct the game and explain the first 2-1 state | `game_curve.csv`, `game_explanation.json` |
| `python -m leafslab backtest` | Score three retrospective preseason forecasts | `season_backtest_metrics.json` |
| `python -m leafslab report` | Generate figures, a readable report, and canonical dashboard payload | `figures/`, `report.md`, `artifact.json`; `dashboard.html` where renderer exists |
| `python -m leafslab run --as-of YYYY-MM-DD` | Entire pipeline with explicit forecast date | All of the above |
| `python -m leafslab update --as-of YYYY-MM-DD` | Fast current forecast with frozen game model | Updated forecast, report, and dashboard payload |
| `python -m leafslab run --offline` | Use previously downloaded sources without network requests | Same outputs; fails when a required cache is absent |

`as_of` is the start of the specified calendar day in Toronto. Standings are taken from the previous day's snapshot. It is not an intraday live forecast. Config defaults to October 1, 2026. As the season progresses, pass the new date. The 2026-27 schedule uses 84 games per club; the held-out 2025-26 season uses 82. Schedule appearance counts must reconcile.

### Dashboard delivery

The dashboard is defined in a portable canonical Data Analytics `artifact.json`, with source metadata and native charts/tables. Its HTML renderer is supplied by the Data Analytics plugin in Work Mode, not by pip. In Work Mode, the report command discovers the packaged renderer. For another installed copy, set `LEAFSLAB_HTML_RENDERER` to its `deliver_portable_artifact.mjs` path and ensure Node is available.

On a laptop without that renderer, Python still creates PNG charts, `report.md`, and the dashboard payload; it **does not claim to create interactive HTML**. Upload the output payload with source files to Work Mode for packaging. No dashboard is rendered from fake data in this handoff.

## Notebook

`analysis.ipynb` provides EDA, missingness checks, modeling methodology, calibration, evaluation, team ranking, game reconstruction, season probabilities, and limitations. Open it in VS Code/Jupyter after `run`. The accompanying `analysis.py` has the same `# %%` cells for VS Code's interactive window.

All notebook code cells were executed top-to-bottom with in-process IPython and saved rich outputs. Standard nbclient kernel execution was blocked by sandbox socket restrictions. The Python companion provides the same analysis cells.

## Models and evaluation

- **Outcome:** eventual home-team victory, including OT/shootout; features cover regulation only.
- **Observation grain:** last post-event state for each game-second, before regulation ends. Equal total weight per game, not per event. Sparse/quiet clock periods have no observations; calibration describes event-observed states rather than uniform elapsed time.
- **Features:** score difference, time remaining, score/time interaction, pregame Elo log-odds, cumulative shot differential, contemporaneous skater difference, goalie-pulled indicators, situation-missing flag.
- **Baseline:** regularized logistic score/time model; also pregame-only and 50% baselines.
- **Candidates:** regularized logistic models (`C=0.1/1`), histogram gradient boosting (`7/15` leaves), and the score/time baseline itself. More features are not forced to win.
- **Training:** 2022-23 and 2023-24. **Tuning:** first half of chronological dates in 2024-25. **Calibration:** second half. **Test:** entire 2025-26 season. No shuffled row split and no game in multiple partitions.
- **Calibration:** logistic calibration of model log-odds, fit on the calibration partition only. Its fit may worsen held-out scores; raw and calibrated metrics are both shown, without retroactive test-based selection.
- **Metrics:** game-weighted Brier score/log loss, AUC, threshold accuracy; reliability bins; paired game-bootstrap Brier-difference interval. Terminal states are excluded.
- **Instinct check:** among held-out games containing a home-leading state predicted in [80%, 90%), compare equal-game mean prediction with eventual win frequency and a game bootstrap interval. A single game's victory does not validate an 85% forecast.
- **Explainability:** permutation feature importance and one-feature scenario contrasts. These are noncausal, nonadditive diagnostics. Correlated features can distort importance and hypothetical states can be uncommon.

The model is selected only from tuning scores. A result is not marketed as "better" unless the paired uncertainty interval supports that statement. Software tests cannot establish forecast accuracy. Cup probabilities are separate from in-game accuracy.

## Team strength and season forecast

Elo starts at 1500, has fixed K=20 and home advantage=45, and regresses 25% toward average at a new season. Updates are batched by calendar date; no same-day final outcome enters another game's pregame rating. These parameters are explicit assumptions, not claimed to have been optimized. A new franchise code receives an average prior; predecessor franchise continuity is not modeled.

One latent strength draw per team per simulated season represents uncertainty; its standard deviation is the empirical training-season Elo drift. This is a proxy, not a calibrated posterior. Regulation goal rates use Poisson maximum likelihood on training-season scores, with the deciding OT/SO goal removed. Simulated regulation ties award the loser one point and the winner two. Current standings initialize the season. Each iteration simulates every remaining scheduled game, top-three division qualification, two wildcards per conference, and four best-of-seven rounds with 2-2-1-1-1 home ice. Stage totals reconcile to 16, 8, 4, 2, and 1 clubs.

Default: 10,000 seeded simulations. Expected points and 10th/90th percentiles, probabilities for each playoff stage, Monte Carlo intervals, and ±50 Toronto Elo sensitivity are exported.

**Limitations:** each sampled latent strength is fixed within its simulated future; no injury, goalie-starter, roster/trade, fatigue, travel, or expected-goals model; independent Poisson goals; all simulated tied regular-season games resolved as OT rather than shootout; official head-to-head standings tie-break not implemented; regular-season Elo reused for playoff games; no independently calibrated playoff/Cup model. Monte Carlo intervals measure simulation error only. Strength sensitivity is not a comprehensive model confidence interval.

Three rolling-origin preseason checks cover 2023-24, 2024-25, and 2025-26. They report points MAE/RMSE and stage Brier scores against constant-rate baselines. These season checks are exploratory: the uncertainty variant followed review of the initial baseline, so they are not claimed as a pristine untouched holdout. A few seasons with dependent club outcomes cannot establish championship calibration.

## Automation

The public dashboard uses the independent GitHub Actions workflow described below. The old Work Mode task is paused; the separate private Site is not refreshed by this GitHub workflow. GitHub Actions runs Python and publishes validated snapshots. GitHub Pages serves the HTML. Failed refreshes retain the last good publication. Future seasons need explicit configuration and model checks.

Before validating or rendering a new dashboard payload, preserve readable labels and precise percentage tooltips:

```bash
python scripts/readable_dashboard.py --artifact outputs/artifact.json --forecast outputs/forecast.json
```

The presentation script preserves numeric forecast and evaluation values, spells out all 32 team names, formats season labels, and recomputes Toronto's explanation from that run's forecast. Then use the canonical renderer and archive script described above.

## LinkedIn draft

Use the executed results and preserve the experimental status when writing a LinkedIn post. The Cup model's experimental status must stay visible.

Use the actual held-out Brier score, baseline comparison, number of games, forecast date, and limitations. Never post test-fixture metrics as hockey results.

## Sources and access record

- NHL schedule JSON: `https://api-web.nhle.com/v1/club-schedule-season/{TEAM}/{SEASON}`
- NHL play-by-play JSON: `https://api-web.nhle.com/v1/gamecenter/{GAME_ID}/play-by-play`
- NHL dated standings JSON: `https://api-web.nhle.com/v1/standings/{YYYY-MM-DD}`
- Endpoint reference: https://github.com/Zmalski/NHL-API-Reference
- Official playoff structure: https://www.nhl.com/info/standings-info/playoff-format
- Official 2026-27 schedule announcement: https://www.nhl.com/news/nhl-announces-2026-27-regular-season-schedule
- Toronto game confirmation: https://www.nhl.com/mapleleafs/video/easton-cowan-post-game-vs-new-york-islanders-6405998358112

The API is public-facing but lacks a versioned schema contract here. The parser was reconciled against all downloaded sources; future schema changes may require repair. Respect the source terms; the separate raw-source archive preserves the downloaded NHL responses and provenance for reproducibility.

## Postseason update boundary
This version forecasts the Cup during the regular season and simulates the entire playoff path. It does not yet condition on completed playoff games. Updates after the regular-season finish fail explicitly; they do not quietly restart the playoffs from scratch.

## SQL audit layer
`queries/` contains the actual SQLite queries producing dashboard metrics. `outputs/dashboard_sources.sqlite` contains the underlying evaluation, simulation-count and season-result tables. `leafslab/dashboard_sql.py` registers the frozen model UDF for game probabilities. These computations independently reconcile with Python outputs. The source modal uses this auditable SQL provenance.

## Independent daily refresh

The public dashboard is refreshed by `.github/workflows/daily-refresh.yml` at
6:17 AM America/Toronto, plus a manual Run workflow button. It runs Python
against NHL JSON endpoints, validates standings/schedules, recomputes 10,000
season simulations, preserves the frozen game/evaluation datasets, commits the
verified snapshot and deploys `docs` directly to GitHub Pages. It does not call
ChatGPT or an OpenAI API, and does not require an API key or personal token.

Run locally from a clean checkout: `pip install -r requirements.txt`, install
Node 22, then `python scripts/scheduled_refresh.py`. Historical schedule API
responses are cached with their checksums and retrieval dates; current-season
schedules and dated standings are requested fresh on each online run. If late
games are still unfinished at midnight, the job uses the latest fully completed
calendar cutoff and displays it. It refuses to regress a published date.

The current model supports the configured regular season only. The existing
postseason cutoff stops publishing unsupported forecasts; the latest verified
dashboard remains available. New seasons require a reviewed `config.json` and
validation before publication. Earlier published seasons remain in the archive.
An API/schema/validation error fails the job instead of replacing good data.
GitHub schedules may be delayed and public-repository schedules can be disabled
after 60 days without repository activity. Check Actions and the displayed date
if the dashboard is stale. No promise of permanent third-party service uptime.

GitHub Pages must use **GitHub Actions** as its publishing source. A bot commit
alone does not trigger a branch-based Pages build, so the workflow explicitly
uploads and deploys the Pages artifact. The Actions cache is only an optimization;
a cache miss downloads the historical schedules again.

The portable renderer in `scripts/portable` is adapted from the same Data
Analytics portable builder used for the original report; its packaged runtime
and semantic fallback are self-contained, with no CDN dependency.
