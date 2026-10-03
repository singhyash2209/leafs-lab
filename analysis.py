# %% [markdown]
# # Leafs Lab — an honest forecasting workflow
# ## Executed results
# Official NHL data: 5,256 games and 1,278,937 regulation states.
# The September 30 2–1 state received an 85.9% modeled win probability.
# The calibrated model did not beat the simpler baseline on held-out Brier
# score. The season forecasts are exploratory, with retrospective limitations.
# This notebook reads saved real-data outputs and exposes the caveats.

# %%
from pathlib import Path
import json
import pandas as pd
import numpy as np
try:
    from IPython.display import display, Markdown, Image
except ImportError:
    def display(value): print(value)
    def Markdown(text): return text
    def Image(filename): return 'Image artifact: '+filename
ROOT = Path.cwd()
if not (ROOT / 'config.json').exists():
    raise RuntimeError('Open the extracted leafs-lab folder as your working directory.')
config = json.loads((ROOT / 'config.json').read_text())
ready = (ROOT / 'data/states.csv').exists()
print('Verified dataset available:', ready)
if not ready:
    print('BLOCKED: run python -m leafslab run with NHL source access. No data/results substituted.')

# %% [markdown]
# ## Context and methods
# Outcome: eventual home-team win, including OT/shootout. Observations are
# nonterminal regulation post-event states. Training and test are separated by
# season, with distinct tuning and calibration dates. Equal total game weights
# prevent a high-event game dominating the evaluation.
#
# ### Key assumptions
# Elo uses past dates only. Goalies/skaters are observed situation codes,
# with explicit missingness. Future season strength is fixed. Simulation error
# intervals are not full model confidence intervals.

# %%
display(pd.DataFrame([
    {'partition': 'Training', 'seasons': str(config['train_seasons'])},
    {'partition': 'Tuning', 'seasons': str(config['validation_season'])+' first half of dates'},
    {'partition': 'Calibration', 'seasons': str(config['validation_season'])+' second half of dates'},
    {'partition': 'Untouched test', 'seasons': str(config['test_season'])}
]))

# %% [markdown]
# ## Data and EDA
# Check season coverage before interpreting distributions. Missingness and
# exclusions are part of the result. Final labels are not input features.

# %%
if ready:
    states = pd.read_csv(ROOT / 'data/states.csv')
    quality = json.loads((ROOT / 'outputs/data_quality.json').read_text())
    display(pd.DataFrame(quality['coverage_by_season']).T)
    display(states.head(10))
    display(states.groupby('season').agg(games=('game_id','nunique'),states=('game_id','size')))
    display(states.isna().sum().rename('missing').to_frame())
    assert not states[['game_id','elapsed']].duplicated().any()
    assert states.groupby('game_id').weight.sum().between(.999999,1.000001).all()
    assert states.elapsed.lt(3600).all()
else:
    print('EDA not executed: official historical observations unavailable.')

# %%
if ready:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    states.goal_diff.value_counts().sort_index().plot.bar(ax=axes[0], color='#003e7e')
    axes[0].set(title='Observed score differences', xlabel='Home minus away goals', ylabel='States')
    states.groupby(pd.cut(states.elapsed/60, range(0,61,5), include_lowest=True), observed=True).size().plot.bar(ax=axes[1], color='#003e7e')
    axes[1].set(title='Observation coverage over regulation', xlabel='Elapsed minutes', ylabel='States')
    figure.tight_layout()
    display(figure)
    plt.close(figure)

# %% [markdown]
# ## Model results and validation
# Logistic regression and histogram gradient boosting are selected by tuning
# Brier score, then calibrated on a later independent period. Test metrics
# describe a frozen pipeline. Never select a model using these test results.
# AUC measures discrimination; Brier/log loss assess probability quality.

# %%
if (ROOT / 'outputs/metrics.json').exists():
    metrics = json.loads((ROOT / 'outputs/metrics.json').read_text())
    assert metrics['status'] == 'real_data_evaluated', 'Fixture results are not analysis'
    display(pd.DataFrame(metrics['test']).T)
    display(pd.DataFrame(metrics['calibration']))
    display(pd.DataFrame(metrics['feature_importance']).sort_values('brier_increase', ascending=False))
    print('Paired game-bootstrap comparison:', metrics['paired_bootstrap'])
    print('Held-out 80–90% home-leading cohort:', metrics['instinct_cohort'])
else:
    print('Model training/evaluation not executed on real NHL observations.')

# %% [markdown]
# ## Game reconstruction and explanations
# Examine the first verified 2–1 post-goal state, rather than searching for a
# convenient high probability. Scenario contrasts are noncausal/nonadditive.

# %%
if (ROOT / 'outputs/game_explanation.json').exists():
    game = json.loads((ROOT / 'outputs/game_explanation.json').read_text())
    print(game['date'], game['home'], 'vs', game['away'])
    print('First 2–1 state:', game['first_2_1_state'])
    if game['explanation']:
        display(pd.DataFrame(game['explanation']['contrasts']))
    if (ROOT / 'outputs/figures/game_probability.png').exists():
        display(Image(filename=str(ROOT / 'outputs/figures/game_probability.png')))
else:
    print('Game reconstruction awaiting official PBP and a trained model.')

# %% [markdown]
# ## Team ratings and season forecast
# Simulation includes current standings, the remaining schedule, division and
# wildcard qualification, overtime points, and four best-of-seven rounds.
# Forecast stage probabilities are unconditional. ±50 Elo is a strength
# sensitivity analysis; it is not a comprehensive uncertainty interval.

# %%
if (ROOT / 'outputs/forecast.json').exists():
    forecast = json.loads((ROOT / 'outputs/forecast.json').read_text())
    assert forecast['status'] == 'real_data_simulated'
    teams = pd.DataFrame(forecast['teams']).sort_values('cup', ascending=False)
    display(teams)
    display(teams[teams.team.eq('TOR')])
    print('As of:', forecast['as_of'], 'Simulations:', forecast['simulations'])
    print('Toronto sensitivity:', forecast.get('sensitivity'))
    assert np.isclose(teams.cup.sum(), 1)
    assert np.isclose(teams.playoffs.sum(), 16)
    print('\n'.join(forecast['assumptions']))
else:
    print('Season forecast not executed: no Cup percentage available.')

# %% [markdown]
# ## Season backtest and takeaways
# Game win-probability accuracy is not championship forecast accuracy.
# One held-out season supplies a diagnostic, not reliable Cup calibration.
# A defensible LinkedIn claim includes the held-out baseline comparison,
# observed data coverage, forecast date, and relevant assumptions.

# %%
if (ROOT / 'outputs/season_backtest_metrics.json').exists():
    season_test = json.loads((ROOT / 'outputs/season_backtest_metrics.json').read_text())
    for result in season_test['seasons']:
        print('Backtest season:', result['season'])
        display(pd.DataFrame(result['probability_scores']).T)
        print('Points MAE:', result['points_mae'], 'RMSE:', result['points_rmse'])
    print('Average points MAE:', season_test['mean_points_mae'])
    print(season_test['warning'])
else:
    print('No held-out season simulation evaluation available yet.')
if (ROOT / 'outputs/report.md').exists():
    display(Markdown((ROOT / 'outputs/report.md').read_text().replace('](figures/', '](outputs/figures/')))
else:
    print('Conclusion: implementation exists; forecasting claims remain unverified until real-data execution.')
