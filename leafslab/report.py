"""Reader artifacts from executed, verified results only. Never fabricate cards."""
import json, os, subprocess, runpy
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_RENDERER='/root/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.10-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs'

def chart_spec(identifier,title,dataset,x,y,kind='line',percent=False,source='model'):
    return {'id':identifier,'title':title,'type':kind,'dataset':dataset,'sourceId':source,
            'valueFormat':'percent' if percent else 'number',
            'encodings':{'x':{'field':x,'type':'nominal' if kind=='bar' else 'quantitative','label':x.replace('_',' ').title()},
                         'y':{'field':y,'type':'quantitative','label':y.replace('_',' ').title()}}}

def build_artifact(config,metrics,forecast,curve,quality,season_test=None):
    season_test=season_test or {}
    now=datetime.now(timezone.utc).isoformat(); teams=pd.DataFrame(forecast['teams'])
    tor=teams[teams.team=='TOR'].iloc[0]
    cards=[{'id':stage,'dataset':'toronto','sourceId':'forecast','description':'Experimental season model. Unconditional share of simulations; not established as better than a naive playoff baseline.',
            'metrics':[{'label':label,'field':stage,'format':'percent'}]} for stage,label in [('playoffs','Make playoffs'),('final','Reach the Final'),('cup','Win the Cup')]]
    key=curve[(curve.home_goals==2)&(curve.away_goals==1)].iloc[0]
    game_cards=[{'id':'game_p','dataset':'game_summary','sourceId':'game','description':'September 30 retrospective state: Toronto leads 2–1 with 5:51 remaining. Excludes that game from training.',
                 'metrics':[{'label':'Toronto win estimate at 2–1','field':'toronto_probability','format':'percent'}]},
                {'id':'accuracy','dataset':'game_summary','sourceId':'model','description':'Equal weight per held-out game, regulation event states, threshold 50%.',
                 'metrics':[{'label':'Held-out classification accuracy','field':'accuracy','format':'percent'}]},
                {'id':'brier','dataset':'game_summary','sourceId':'model','description':'Probability error; lower is better. Selected calibrated model compared with score/time baseline.',
                 'metrics':[{'label':'Calibrated Brier score','field':'brier','format':'number'},
                            {'label':'Score/time baseline','field':'baseline_brier','format':'number'}]}]
    cards=game_cards+cards
    sources=[{'id':'model','label':'Held-out NHL game-model evaluation','path':'outputs/metrics.json',
              'query':{'language':'python','engine':'scikit-learn','sql':'python -m leafslab train',
                       'description':'Frozen test-season evaluation. Equal total weight per game.',
                       'tables_used':['data/states.csv','data/rated_games.csv'],
                       'filters':{'test_season':config['test_season'],'regulation_only':True,'terminal_states_excluded':True},
                       'executed_at':now}},
             {'id':'forecast','label':'Schedule and standings Monte Carlo','path':'outputs/forecast.json',
              'query':{'language':'python','engine':'numpy/scipy','sql':'python -m leafslab forecast',
                       'description':'Fixed-strength remaining-season and playoff simulation.',
                       'tables_used':['data/games.csv','NHL dated standings'],
                       'filters':{'as_of':forecast['as_of'],'simulations':forecast['simulations']},'executed_at':now}},
             {'id':'game','label':'Toronto game reconstruction','path':'outputs/game_curve.csv',
              'query':{'language':'python','engine':'scikit-learn','sql':'python -m leafslab game --date 2026-09-30',
                       'description':'Post-event win probabilities from the frozen calibrated model.',
                       'tables_used':['outputs/game_curve.csv','NHL play-by-play'],'executed_at':now}}]
    sources.append({'id':'season_eval','label':'Retrospective preseason checks','path':'outputs/season_backtest_metrics.json',
                    'query':{'language':'shell','engine':'Python','sql':'python -m leafslab backtest',
                             'description':'Three rolling-origin season checks; revision followed baseline review. Outcomes within each season are dependent.',
                             'tables_used':['outputs/season_backtest_metrics.json','data/games.csv'],'executed_at':now}})
    season_rows=[{'season':str(x['season']), 'points_mae':x['points_mae'],
                  'playoff_brier':x['probability_scores']['playoffs']['brier'],'playoff_baseline':.25,
                  'cup_brier':x['probability_scores']['cup']['brier'],'cup_baseline':31/1024} for x in season_test.get('seasons',[])]
    comparison=[{'model':name,**row} for name,row in metrics['test'].items()]
    curve=curve.copy(); curve['minute']=curve.elapsed/60
    curves=curve[['minute','toronto_probability','home_goals','away_goals']].iloc[::max(1,len(curve)//300)].to_dict('records')
    # Include last state regardless of downsampling.
    if curves[-1]['minute']!=float(curve.iloc[-1].minute): curves.append(curve[['minute','toronto_probability','home_goals','away_goals']].iloc[-1].to_dict())
    charts=[chart_spec('game_curve','Toronto win probability during the game','game','minute','toronto_probability',percent=True,source='game'),
            chart_spec('reliability','Predicted versus observed win frequency','calibration','mean_prediction','observed_rate',kind='scatter',percent=True),
            chart_spec('brier','Held-out Brier score (lower is better)','models','model','brier',kind='bar'),
            chart_spec('cup','Cup probabilities: top 10 and Toronto','cup_comparison','team','cup',kind='bar',percent=True,source='forecast')]
    manifest={'version':1,'surface':'dashboard','title':'Leafs Lab — From a 2–1 Lead to Cup Chances',
              'description':'Verified historical game probabilities and a model-based season forecast.',
              'generatedAt':now,'filters':[], 'cards':cards,'charts':charts,
              'tables':[{'id':'teams','title':'League forecast','dataset':'teams','sourceId':'forecast',
                         'defaultSort':{'field':'cup','direction':'desc'},
                         'columns':[{'field':'team','label':'Team','type':'text'},
                                    {'field':'elo','label':'Elo','format':'number'},
                                    {'field':'mean_points','label':'Expected points','format':'number'},
                                    *[{'field':f,'label':label,'format':'percent'} for f,label in [('playoffs','Playoffs'),('final','Final'),('cup','Cup')]]]}],
              'sources':sources,
              'blocks':[{'id':'intro','type':'markdown','body':f"# Leafs Lab\n\nFrom the September 30 game to a testable forecasting project. Season snapshot: {forecast['as_of']}."},
                        {'id':'game_headline','type':'metric-strip','cardIds':['game_p','accuracy','brier']},
                        {'id':'game_guardrail','type':'markdown','sourceId':'model','body':'The calibrated model did not establish an improvement over the score/time baseline on the held-out season. Its 80–90% home-leading cohort also showed overconfidence. A correct prediction in one game does not validate a probability.'},
                        *[{'id':'block_'+c['id'],'type':'chart','chartId':c['id']} for c in charts if c['id']!='cup'],
                        {'id':'season_guardrail','type':'markdown','body':'## Experimental season forecast\n\nThe revised season model did not outperform the 50% playoff baseline in the latest retrospective season check. Cup intervals describe simulation error, not full model uncertainty.'},
                        {'id':'headline','type':'metric-strip','cardIds':['playoffs','final','cup']},
                        {'id':'block_cup','type':'chart','chartId':'cup'},
                        {'id':'league_table','type':'table','tableId':'teams'},
                        {'id':'season_table','type':'table','tableId':'season_checks'},
                        {'id':'limits','type':'markdown','body':'## Model limits\n\n'+'\n'.join('- '+x for x in forecast['assumptions'])+'\n\nGame features: score, time remaining, prior team strength, shots, skaters and goalie-pulled indicators. Calibration and feature importance describe predictions; they do not establish causality.'}]}
    manifest['tables'].append({'id':'season_checks','title':'Retrospective season checks','dataset':'season_checks','sourceId':'season_eval',
                             'defaultSort':{'field':'season','direction':'asc'},
                             'columns':[{'field':'season','label':'Season','type':'text'},
                                        {'field':'points_mae','label':'Points MAE','format':'number'},
                                        {'field':'playoff_brier','label':'Playoff Brier','format':'number'},
                                        {'field':'playoff_baseline','label':'Naive playoff Brier','format':'number'},
                                        {'field':'cup_brier','label':'Cup Brier','format':'number'},
                                        {'field':'cup_baseline','label':'Naive Cup Brier','format':'number'}]})
    bounded_teams=teams.drop(columns='cup_mc_95_interval').sort_values('cup',ascending=False).to_dict('records')
    return {'surface':'dashboard','manifest':manifest,
            'snapshot':{'version':1,'generatedAt':now,'status':'ready','datasets':{'toronto':[tor.drop(labels='cup_mc_95_interval').to_dict()],
                        'game_summary':[{'toronto_probability':float(key.toronto_probability),'accuracy':metrics['test']['selected_calibrated']['accuracy'],'brier':metrics['test']['selected_calibrated']['brier'],'baseline_brier':metrics['test']['baseline']['brier']}],'season_checks':season_rows,
                        'teams':bounded_teams,'cup_comparison':teams.sort_values('cup',ascending=False).head(10).drop(columns='cup_mc_95_interval').to_dict('records')+([tor.drop(labels='cup_mc_95_interval').to_dict()] if 'TOR' not in teams.sort_values('cup',ascending=False).head(10).team.tolist() else []),'game':curves,'models':comparison,'calibration':metrics['calibration']},'accessIssues':[]},
            'sources':sources}

def build(config):
    out=Path('outputs'); metrics=json.loads((out/'metrics.json').read_text()); forecast=json.loads((out/'forecast.json').read_text())
    quality=json.loads((out/'data_quality.json').read_text()); season_test=json.loads((out/'season_backtest_metrics.json').read_text()); curve=pd.read_csv(out/'game_curve.csv')
    if metrics['status']!='real_data_evaluated' or forecast['status']!='real_data_simulated': raise ValueError('Only verified real-data results may be rendered')
    plots=out/'figures'; plots.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(curve.elapsed/60,100*curve.toronto_probability,color='#003e7e',lw=2)
    for row in curve[curve.goal_at_second.eq(1)].to_dict('records'): ax.axvline(row['elapsed']/60,color='#999999',alpha=.35)
    ax.set(xlabel='Elapsed regulation minutes',ylabel='Toronto win probability (%)',ylim=(0,100),xlim=(0,60),title='Toronto game probabilities — goals marked'); fig.tight_layout(); fig.savefig(plots/'game_probability.png'); plt.close(fig)
    cal=pd.DataFrame(metrics['calibration']); fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],'--',color='gray'); ax.plot(cal.mean_prediction,cal.observed_rate,'o-',color='#003e7e')
    ax.set(xlabel='Mean predicted probability',ylabel='Observed home win frequency',xlim=(0,1),ylim=(0,1),title='Held-out calibration'); fig.tight_layout(); fig.savefig(plots/'calibration.png'); plt.close(fig)
    comp=pd.DataFrame(metrics['test']).T.sort_values('brier'); fig,ax=plt.subplots(figsize=(8,4)); ax.barh(comp.index,comp.brier,color='#003e7e'); ax.invert_yaxis()
    ax.set(xlabel='Game-weighted Brier score — lower is better',title='Untouched test season'); fig.tight_layout(); fig.savefig(plots/'model_comparison.png'); plt.close(fig)
    all_teams=pd.DataFrame(forecast['teams']); teams=all_teams.sort_values('cup',ascending=False).head(10); teams=pd.concat([teams,all_teams[all_teams.team.eq('TOR')]]).drop_duplicates('team').sort_values('cup'); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(teams.team,100*teams.cup,color=['#003e7e' if t=='TOR' else '#8fa5b9' for t in teams.team]); ax.set(xlabel='Unconditional Cup probability (%)',title=f"Cup forecast as of {forecast['as_of']}"); fig.tight_layout(); fig.savefig(plots/'cup_forecast.png'); plt.close(fig)
    tor=next(t for t in forecast['teams'] if t['team']=='TOR'); test=metrics['test']; interval=metrics['paired_bootstrap']['95_percent_game_bootstrap_interval']
    verdict='Evidence favors the selected model.' if interval[1]<0 else 'Improvement over baseline is not established by the bootstrap interval.'
    text=f"""# Leafs Lab: executed results

Snapshot: {forecast['as_of']}. Experimental model-based season forecast. The latest retrospective playoff Brier score is above the 0.25 naive baseline; these percentages are exploratory.

Toronto: playoffs **{tor['playoffs']:.1%}**, Final **{tor['final']:.1%}**, Cup **{tor['cup']:.1%}**.
Cup Monte Carlo interval: {tor['cup_mc_95_interval'][0]:.1%}–{tor['cup_mc_95_interval'][1]:.1%}. This interval measures simulation error only.

## Game-model validation
Selected model: {metrics['selected_model']}. Untouched test season: {config['test_season']}.
Game-weighted Brier: selected/calibrated **{test['selected_calibrated']['brier']:.4f}**, score/time baseline **{test['baseline']['brier']:.4f}**.
Paired Brier-difference interval: {interval[0]:.4f} to {interval[1]:.4f}. {verdict}

![Game reconstruction](figures/game_probability.png)
![Calibration](figures/calibration.png)
![Model comparison](figures/model_comparison.png)
![Cup forecast](figures/cup_forecast.png)

## Season-model evaluation
Retrospective preseason checks cover three seasons. Model parameters were revised after reviewing the initial fixed-strength baseline; these checks are exploratory, not a pristine untouched holdout. Details are saved in season_backtest_metrics.json and fixed_strength_baseline_backtests.json.

## Data quality
{quality['games']} parsed games; {quality['states']} regulation states. Last complete game date: {quality['last_game_date']}.
Situation missingness: {quality['situation_missing_fraction']:.1%}. See data_quality.json for season coverage and parser_errors.json for exclusions.

## Limits
"""+'\n'.join('- '+s for s in forecast['assumptions'])+'\n\nSources: NHL JSON responses retained with URL, retrieval timestamp, and SHA-256. Implementation: leafslab/data.py, model.py, season.py. Exact results: metrics.json, forecast.json, game_explanation.json. The season backtest must be considered separately from game-model validation.\n'
    (out/'report.md').write_text(text)
    artifact=build_artifact(config,metrics,forecast,curve,quality,season_test)
    from .dashboard_sql import materialize
    sql_data,queries=materialize(config,metrics,forecast,curve,season_test)
    datasets=artifact['snapshot']['datasets']
    for name in ['models','calibration','game','teams','season_checks']: datasets[name]=sql_data[name]
    tor_sql=next(t for t in sql_data['teams'] if t['team']=='TOR');datasets['toronto']=[tor_sql]
    key_sql=next(g for g in sql_data['game'] if g['home_goals']==2 and g['away_goals']==1)
    selected_sql=next(m for m in sql_data['models'] if m['model']=='selected_calibrated')
    baseline_sql=next(m for m in sql_data['models'] if m['model']=='baseline')
    datasets['game_summary']=[{'toronto_probability':key_sql['toronto_probability'],'accuracy':selected_sql['accuracy'],
                               'brier':selected_sql['brier'],'baseline_brier':baseline_sql['brier']}]

    datasets['cup_comparison']=sql_data['teams'][:10]+([tor_sql] if 'TOR' not in [t['team'] for t in sql_data['teams'][:10]] else [])
    for source in artifact['manifest']['sources']:
        source['path']='queries/'+source['id']+'.sql'
        source['query']['sql']=queries[source['id']]
        source['query']['language']='sql'; source['query']['engine']='SQLite'+(' with Python model UDF' if source['id']=='game' else '')
        source['query']['tables_used']={'model':['test_predictions'],'game':['game_states','outputs/model.joblib'],'forecast':['forecast_counts'],'season_eval':['season_predictions']}[source['id']]
        source['query']['description']+=' Database: outputs/dashboard_sources.sqlite; reproduce through leafslab/dashboard_sql.py.'
    artifact['sources']=artifact['manifest']['sources']

    presentation = Path(__file__).resolve().parents[1]/'scripts'/'readable_dashboard.py'
    artifact = runpy.run_path(str(presentation))['readable'](artifact, forecast)
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2,allow_nan=False))
    renderer=os.environ.get('LEAFSLAB_HTML_RENDERER',DEFAULT_RENDERER)
    if Path(renderer).exists():
        subprocess.run([os.environ.get('CODEX_PRIMARY_RUNTIME_NODE','node'),renderer,'--input',str(out/'artifact.json'),'--output',str(out/'dashboard.html')],check=True)
    else:
        print('Charts, report.md and canonical artifact.json created. HTML packaging requires the Data Analytics portable renderer; open this project in Work Mode to render it.')
    print('Saved verified results under outputs/',flush=True)
