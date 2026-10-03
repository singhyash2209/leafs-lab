"""Auditable SQLite calculations for dashboard metrics and model inference.
The saved SQL materially computes displayed probabilities/errors; Python
continues to own ingestion, fitting, simulations and source validation.
"""
from pathlib import Path
import sqlite3, math, json
import pandas as pd
import numpy as np
import joblib
from .model import FEATURES, predict

MODEL_SQL = '''SELECT 'selected_calibrated' AS model,
 SUM(weight*(prediction-home_win)*(prediction-home_win))/SUM(weight) AS brier,
 SUM(weight*CASE WHEN (prediction>=0.5)=home_win THEN 1.0 ELSE 0.0 END)/SUM(weight) AS accuracy
 FROM test_predictions
 UNION ALL SELECT 'baseline',SUM(weight*(baseline-home_win)*(baseline-home_win))/SUM(weight),
 SUM(weight*CASE WHEN (baseline>=0.5)=home_win THEN 1.0 ELSE 0.0 END)/SUM(weight) FROM test_predictions
 UNION ALL SELECT 'selected_raw',SUM(weight*(raw_prediction-home_win)*(raw_prediction-home_win))/SUM(weight),
 SUM(weight*CASE WHEN (raw_prediction>=0.5)=home_win THEN 1.0 ELSE 0.0 END)/SUM(weight) FROM test_predictions
 UNION ALL SELECT 'pregame_only',SUM(weight*(pregame_probability-home_win)*(pregame_probability-home_win))/SUM(weight),
 SUM(weight*CASE WHEN (pregame_probability>=0.5)=home_win THEN 1.0 ELSE 0.0 END)/SUM(weight) FROM test_predictions
 UNION ALL SELECT 'constant_50_percent',SUM(weight*(0.5-home_win)*(0.5-home_win))/SUM(weight),
 SUM(weight*home_win)/SUM(weight) FROM test_predictions'''
CALIBRATION_SQL = '''SELECT MIN(9,CAST(prediction*10 AS INTEGER)) AS bin,
 SUM(weight*prediction)/SUM(weight) AS mean_prediction,
 SUM(weight*home_win)/SUM(weight) AS observed_rate,
 COUNT(DISTINCT game_id) AS games, COUNT(*) AS states
 FROM test_predictions GROUP BY bin ORDER BY bin'''
GAME_SQL = '''SELECT elapsed/60.0 AS minute,
 CASE WHEN home='TOR' THEN
 home_win_probability(goal_diff,remaining,score_clock,pregame_logit,shot_diff,skater_diff,home_empty,away_empty,situation_missing)
 ELSE 1.0-home_win_probability(goal_diff,remaining,score_clock,pregame_logit,shot_diff,skater_diff,home_empty,away_empty,situation_missing)
 END AS toronto_probability, home_goals, away_goals
 FROM game_states ORDER BY elapsed'''
FORECAST_SQL = '''SELECT team,elo,points_sum*1.0/simulations AS mean_points,
 points_p10,points_p90,playoffs_count*1.0/simulations AS playoffs,
 round_2_count*1.0/simulations AS round_2,
 conference_final_count*1.0/simulations AS conference_final,
 final_count*1.0/simulations AS final,
 cup_count*1.0/simulations AS cup, cup_count AS cup_simulation_wins
 FROM forecast_counts ORDER BY cup DESC,team'''
SEASON_SQL = '''SELECT CAST(season AS TEXT) AS season,
 AVG(ABS(expected_points-actual_points)) AS points_mae,
 AVG((p_playoffs-actual_playoffs)*(p_playoffs-actual_playoffs)) AS playoff_brier,
 0.25 AS playoff_baseline,
 AVG((p_cup-actual_cup)*(p_cup-actual_cup)) AS cup_brier,
 31.0/1024.0 AS cup_baseline
 FROM season_predictions GROUP BY season ORDER BY season'''

def materialize(config,metrics,forecast,curve,season_test):
    out=Path('outputs'); query_dir=Path('queries'); query_dir.mkdir(exist_ok=True)
    bundle=joblib.load(out/'model.joblib')
    pred=pd.read_csv(out/'test_predictions.csv')
    # Older exports can be enriched without changing any frozen predictions.
    incomplete=len(pred)!=metrics['partitions']['test']['states'] or pred.game_id.nunique()!=metrics['partitions']['test']['games']
    if incomplete or 'raw_prediction' not in pred or 'pregame_probability' not in pred:
        states=pd.read_csv('data/states.csv'); test=states[states.season.eq(config['test_season'])].copy()
        pred=test[['game_id','date','elapsed','home_win','weight']].copy()
        pred['prediction']=predict(bundle,test)
        pred['baseline']=bundle['baseline'].predict_proba(test[['goal_diff','remaining','score_clock']])[:,1]
        pred['raw_prediction']=bundle['model'].predict_proba(test[bundle['features']])[:,1]
        pred['pregame_probability']=1/(1+np.exp(-test.pregame_logit))
        tmp=out/'test_predictions.tmp.csv'; pred.to_csv(tmp,index=False)
        verified=pd.read_csv(tmp)
        if len(verified)!=len(test) or verified.game_id.nunique()!=test.game_id.nunique():
            raise ValueError('Prediction export failed full-row readback')
        tmp.replace(out/'test_predictions.csv')
        if incomplete: (out/'prediction_export_repair.json').write_text(json.dumps({'reason':'Prior CSV did not cover all held-out observations','regenerated_from':'frozen model and verified data/states.csv','states':len(pred),'games':int(pred.game_id.nunique()),'model_and_metrics_changed':False},indent=2))
    connection=sqlite3.connect(out/'dashboard_sources.sqlite')
    connection.create_function('home_win_probability',len(FEATURES),lambda *values:float(predict(bundle,pd.DataFrame([dict(zip(FEATURES,values))]))[0]))
    pred.to_sql('test_predictions',connection,index=False,if_exists='replace')
    # Features only are used by the UDF. Final outcome and saved p are excluded.
    curve.drop(columns=['home_probability','toronto_probability','home_win'],errors='ignore').to_sql('game_states',connection,index=False,if_exists='replace')
    counts=[]
    for t in forecast['teams']:
        row={k:t[k] for k in ['team','elo','points_p10','points_p90']}
        row.update(simulations=forecast['simulations'],points_sum=int(round(t['mean_points']*forecast['simulations'])))
        for stage in ['playoffs','round_2','conference_final','final','cup']:
            row[stage+'_count']=t.get(stage+'_count',int(round(t[stage]*forecast['simulations'])))
        counts.append(row)
    pd.DataFrame(counts).to_sql('forecast_counts',connection,index=False,if_exists='replace')
    games=pd.read_csv('data/games.csv'); season_rows=[]
    for backtest in season_test['seasons']:
        season=backtest['season']; result=json.loads((out/f'backtest_{season}_forecast.json').read_text())
        reg=games[(games.season.eq(season))&(games.game_type.eq(2))&games.status.isin(['OFF','FINAL'])]
        playoffs=games[(games.season.eq(season))&(games.game_type.eq(3))&games.status.isin(['OFF','FINAL'])]
        entrants=set(playoffs.home)|set(playoffs.away); wins={}; pts={}
        for g in playoffs.to_dict('records'):
            winner=g['home'] if g['home_score']>g['away_score'] else g['away'];wins[winner]=wins.get(winner,0)+1
        for g in reg.to_dict('records'):
            hw=g['home_score']>g['away_score'];extra=g['last_period'] in ('OT','SO')
            pts[g['home']]=pts.get(g['home'],0)+(2 if hw else int(extra));pts[g['away']]=pts.get(g['away'],0)+(int(extra) if hw else 2)
        for t in result['teams']:
            season_rows.append({'season':season,'team':t['team'],'expected_points':t['mean_points'],
                                'actual_points':pts[t['team']],'p_playoffs':t['playoffs'],'p_cup':t['cup'],
                                'actual_playoffs':int(t['team'] in entrants),'actual_cup':int(wins.get(t['team'],0)==16)})
    pd.DataFrame(season_rows).to_sql('season_predictions',connection,index=False,if_exists='replace')
    queries={'model':MODEL_SQL+';\n\n'+CALIBRATION_SQL+';','game':GAME_SQL+';',
             'forecast':FORECAST_SQL+';','season_eval':SEASON_SQL+';'}
    for name,sql in queries.items(): (query_dir/f'{name}.sql').write_text(sql+'\n')
    data={'models':pd.read_sql_query(MODEL_SQL,connection).to_dict('records'),
          'calibration':pd.read_sql_query(CALIBRATION_SQL,connection).to_dict('records'),
          'game':pd.read_sql_query(GAME_SQL,connection).to_dict('records'),
          'teams':pd.read_sql_query(FORECAST_SQL,connection).to_dict('records'),
          'season_checks':pd.read_sql_query(SEASON_SQL,connection).to_dict('records')}
    # Independent reconciliation of newly computed dashboard values.
    for row in data['models']:
        assert abs(row['brier']-metrics['test'][row['model']]['brier'])<1e-10
        assert abs(row['accuracy']-metrics['test'][row['model']]['accuracy'])<1e-10
    assert np.max(np.abs(np.array([r['toronto_probability'] for r in data['game']])-curve.toronto_probability.to_numpy()))<1e-10
    for row in data['season_checks']:
        original=next(s for s in season_test['seasons'] if str(s['season'])==row['season'])
        assert abs(row['points_mae']-original['points_mae'])<1e-10
        assert abs(row['playoff_brier']-original['probability_scores']['playoffs']['brier'])<1e-10
    connection.close()
    return data,queries
